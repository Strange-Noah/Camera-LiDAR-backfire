"""Canonical tables, constructed scene by scene without loading sensor blobs."""
import json
from pathlib import Path
import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
from .core import differences, yaw, wrap, relative_components, corners_world, project_boxes
from .store import Store

CAMERAS = ['CAM_FRONT','CAM_FRONT_LEFT','CAM_FRONT_RIGHT','CAM_BACK','CAM_BACK_LEFT','CAM_BACK_RIGHT']


class Writers:
    def __init__(self, root):
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)
        self.writers = {}
    def put(self, name, frame):
        if frame.empty:
            return
        # Explicit string dtype prevents null-only first scene schema traps.
        for c in frame.select_dtypes('object'):
            frame[c] = frame[c].astype('string')
        table = pa.Table.from_pandas(frame, preserve_index=False)
        if name not in self.writers:
            self.writers[name] = pq.ParquetWriter(self.root / (name + '.parquet'), table.schema, compression='zstd')
        writer = self.writers[name]
        writer.write_table(table.cast(writer.schema))
    def close(self):
        for writer in self.writers.values():
            writer.close()


def construct(root, version, output, cache, config):
    store = Store(root, version, cache)
    out = Path(output)
    out.mkdir(parents=True, exist_ok=True)
    writers = Writers(out)
    categories = {r['token']: r['name'] for r in store.all('category')}
    instances = {r['token']: r for r in store.all('instance')}
    calibrations = {r['token']: r for r in store.all('calibrated_sensor')}
    sensors = {r['token']: r for r in store.all('sensor')}
    channel = {k: sensors[v['sensor_token']]['channel'] for k,v in calibrations.items()}
    scenes = store.all('scene')
    seen_instances = {}
    validation = {'instance_cross_scene': [], 'scenes': len(scenes), 'annotation_links_checked': 0,
                  'sensor_links_checked': 0, 'sample_links_checked': 0, 'all_camera_channels_checked': True}
    for scene_i, scene in enumerate(scenes):
        st = scene['token']
        samples = sorted(store.children('sample', st), key=lambda r:r['timestamp'])
        assert len(samples) == scene['nbr_samples']
        assert samples[0]['token'] == scene['first_sample_token'] and samples[-1]['token'] == scene['last_sample_token']
        sample_map = {s['token']:s for s in samples}
        for i,s in enumerate(samples):
            assert s['prev'] == (samples[i-1]['token'] if i else '')
            assert s['next'] == (samples[i+1]['token'] if i+1<len(samples) else '')
            validation['sample_links_checked'] += 1
        sd = store.scene_rows('sample_data', st)
        sd_map = {r['token']:r for r in sd}
        relevant = [r for r in sd if channel[r['calibrated_sensor_token']] in CAMERAS + ['LIDAR_TOP']]
        poses = {r['ego_pose_token']:store.get('ego_pose', r['ego_pose_token']) for r in relevant}
        key = {(r['sample_token'], channel[r['calibrated_sensor_token']]):r for r in relevant if r['is_key_frame']}
        lidar = sorted([r for r in relevant if channel[r['calibrated_sensor_token']]=='LIDAR_TOP'], key=lambda r:r['timestamp'])
        # Interpolate ego position/yaw to exact annotation sample time; never extrapolate.
        lt = np.array([r['timestamp'] for r in lidar], dtype=np.int64)
        assert (np.diff(lt)>0).all()
        ep = np.array([poses[r['ego_pose_token']]['translation'] for r in lidar])
        ey = np.unwrap([yaw(poses[r['ego_pose_token']]['rotation']) for r in lidar])
        ts = np.array([s['timestamp'] for s in samples], dtype=np.int64)
        origin = int(lt[0]); tx=(ts-origin)/1e6; lx=(lt-origin)/1e6
        position = np.column_stack([np.interp(tx,lx,ep[:,j],left=np.nan,right=np.nan) for j in range(3)])
        ego_yaw=np.interp(tx,lx,ey,left=np.nan,right=np.nan)
        ev, esp, enn, em = differences(ts, position, config['ego_neighbor_max_gap_s'])
        ew, _, _, _ = differences(ts, ego_yaw, config['ego_neighbor_max_gap_s'])
        ego = pd.DataFrame({'sample_token':[s['token'] for s in samples], 'timestamp':ts,
            'ego_x':position[:,0], 'ego_y':position[:,1], 'ego_z':position[:,2], 'ego_yaw':wrap(ego_yaw),
            'ego_vx':ev[:,0], 'ego_vy':ev[:,1], 'ego_speed':np.linalg.norm(ev[:,:2],axis=1),
            'ego_yaw_rate':ew, 'ego_velocity_span_s':esp, 'ego_velocity_neighbors':enn, 'ego_velocity_method':em,
            'ego_pose_valid':np.isfinite(position).all(axis=1), 'ego_velocity_valid':np.isfinite(ev[:,:2]).all(axis=1),
            'ego_yaw_rate_valid':np.isfinite(ew)})
        ego['scene_token']=st
        writers.put('ego_samples',ego.copy())
        intervals=[]; timing=[]
        def linked_dt(r, direction):
            token=r[direction]
            if not token:
                return np.nan
            other=sd_map.get(token) or store.get('sample_data',token)
            assert channel[other['calibrated_sensor_token']]==channel[r['calibrated_sensor_token']]
            assert other['next' if direction=='prev' else 'prev']==r['token']
            other_sample=sample_map.get(other['sample_token']) or store.get('sample',other['sample_token'])
            assert other_sample['scene_token']==st
            delta=(r['timestamp']-other['timestamp']) if direction=='prev' else (other['timestamp']-r['timestamp'])
            assert delta>0
            validation['sensor_links_checked']+=1
            return float(delta)
        for r in relevant:
            intervals.append({'scene_token':st,'sample_token':r['sample_token'],'sample_data_token':r['token'],
                'channel':channel[r['calibrated_sensor_token']], 'timestamp_us':r['timestamp'],
                'is_key_frame':r['is_key_frame'], 'prev_dt_us':linked_dt(r,'prev'), 'next_dt_us':linked_dt(r,'next')})
        intervals=pd.DataFrame(intervals)
        intervals['prev_dt_s']=intervals.prev_dt_us/1e6; intervals['next_dt_s']=intervals.next_dt_us/1e6
        writers.put('sensor_intervals', intervals.copy())
        for s in samples:
            l=key[(s['token'],'LIDAR_TOP')]
            for c in CAMERAS:
                r=key[(s['token'],c)]
                row={'scene_token':st,'sample_token':s['token'],'camera_channel':c,
                     'sample_timestamp':s['timestamp'],'camera_timestamp':r['timestamp'],'lidar_timestamp':l['timestamp'],
                     'camera_sample_data_token':r['token'],'lidar_sample_data_token':l['token'],
                     'camera_minus_lidar_dt_us':float(r['timestamp']-l['timestamp']),
                     'abs_camera_lidar_dt_us':float(abs(r['timestamp']-l['timestamp']))}
                for prefix,record in [('camera',r),('lidar',l)]:
                    for direction in ['prev','next']:
                        row[f'{prefix}_{direction}_dt_us']=linked_dt(record,direction)
                for name in list(row):
                    if name.endswith('_us'):
                        row[name[:-3]+'_s']=row[name]/1e6
                row['timing_valid']=True
                timing.append(row)
        writers.put('sensor_timing',pd.DataFrame(timing))
        anns=store.scene_rows('sample_annotation',st)
        rows=[]
        annmap={a['token']:a for a in anns}
        for a in anns:
            inst=instances[a['instance_token']]
            if a['instance_token'] in seen_instances and seen_instances[a['instance_token']]!=st:
                validation['instance_cross_scene'].append(a['instance_token'])
                raise ValueError('Instance spans scenes; trajectory model invalid')
            seen_instances[a['instance_token']]=st
            for direction in ['prev','next']:
                if a[direction]:
                    other=annmap[a[direction]]
                    assert other['instance_token']==a['instance_token']
                    assert other['next' if direction=='prev' else 'prev']==a['token']
            validation['annotation_links_checked']+=1
            rows.append({'scene_token':st,'sample_token':a['sample_token'],'annotation_token':a['token'],
                'instance_token':a['instance_token'],'category':categories[inst['category_token']],
                'visibility_token':a['visibility_token'],'visibility':float(a['visibility_token']) if a['visibility_token'] in ['1','2','3','4'] else np.nan,
                'num_lidar_pts':a['num_lidar_pts'],'num_radar_pts':a['num_radar_pts'],
                'x_world':a['translation'][0],'y_world':a['translation'][1],'z_world':a['translation'][2],
                'width':a['size'][0],'length':a['size'][1],'height':a['size'][2], 'yaw_object':yaw(a['rotation']),
                'prev_annotation_token':a['prev'],'next_annotation_token':a['next']})
        obj=pd.DataFrame(rows).merge(ego.drop(columns='scene_token'),on='sample_token',validate='many_to_one')
        obj=obj.sort_values(['instance_token','timestamp']).reset_index(drop=True)
        for col in ['object_vx','object_vy','object_speed','velocity_span_s','annotation_prev_dt_s','annotation_next_dt_s']:
            obj[col]=np.nan
        obj['velocity_neighbors']=0;obj['velocity_method']='unavailable';obj['track_position']=0;obj['track_length']=0
        for token,group in obj.groupby('instance_token',sort=False):
            inst=instances[token]
            assert len(group)==inst['nbr_annotations']
            assert group.iloc[0].annotation_token==inst['first_annotation_token']
            assert group.iloc[-1].annotation_token==inst['last_annotation_token']
            velocity,span,neighbors,method=differences(group.timestamp.to_numpy(),group[['x_world','y_world']].to_numpy(),config['object_neighbor_max_gap_s'])
            ix=group.index
            tokens=group.annotation_token.tolist()
            assert group.prev_annotation_token.tolist()==['']+tokens[:-1]
            assert group.next_annotation_token.tolist()==tokens[1:]+['']
            obj.loc[ix,['object_vx','object_vy']]=velocity
            obj.loc[ix,'object_speed']=np.linalg.norm(velocity,axis=1)
            obj.loc[ix,'velocity_span_s']=span; obj.loc[ix,'velocity_neighbors']=neighbors;obj.loc[ix,'velocity_method']=method
            obj.loc[ix,'track_position']=np.arange(len(ix));obj.loc[ix,'track_length']=len(ix)
            dt=np.diff(group.timestamp.to_numpy())/1e6
            obj.loc[ix,'annotation_prev_dt_s']=np.r_[np.nan,dt];obj.loc[ix,'annotation_next_dt_s']=np.r_[dt,np.nan]
        obj['velocity_valid']=np.isfinite(obj.object_speed)
        d=obj[['x_world','y_world','z_world']].to_numpy()-obj[['ego_x','ego_y','ego_z']].to_numpy()
        obj['distance_3d']=np.linalg.norm(d,axis=1);obj['distance_xy']=np.linalg.norm(d[:,:2],axis=1)
        rel,radial,lateral=relative_components(obj[['x_world','y_world']].to_numpy(),obj[['ego_x','ego_y']].to_numpy(),obj[['object_vx','object_vy']].to_numpy(),obj[['ego_vx','ego_vy']].to_numpy())
        obj['relative_vx']=rel[:,0];obj['relative_vy']=rel[:,1];obj['relative_speed']=np.linalg.norm(rel,axis=1)
        obj['relative_radial_velocity']=radial;obj['relative_lateral_velocity']=abs(lateral);obj['relative_lateral_velocity_signed']=lateral
        obj['relative_velocity_valid']=np.isfinite(obj.relative_speed)&(obj.distance_xy>0)
        obj['relative_orientation']=wrap(obj.yaw_object.to_numpy()-np.arctan2(d[:,1],d[:,0]))
        obj['abs_ego_yaw_rate']=abs(obj.ego_yaw_rate)
        obj['bearing_rate_world_rad_s']=lateral/obj.distance_xy
        obj['bearing_rate_ego_rad_s']=obj.bearing_rate_world_rad_s-obj.ego_yaw_rate
        obj['visibility_valid']=obj.visibility.notna();obj['size_valid']=(obj[['width','length','height']]>0).all(axis=1)
        obj['distance_valid']=np.isfinite(obj.distance_xy)&(obj.distance_xy>0)
        obj['lidar_support_valid']=obj.num_lidar_pts>=0
        projections=[]
        for sample_token,group in obj.groupby('sample_token',sort=False):
            corners=corners_world(group[['x_world','y_world','z_world']].to_numpy(),group[['width','length','height']].to_numpy(),[annmap[t]['rotation'] for t in group.annotation_token])
            for c in CAMERAS:
                r=key[(sample_token,c)]
                fields=project_boxes(corners,poses[r['ego_pose_token']],calibrations[r['calibrated_sensor_token']],r['width'],r['height'],config['projection_near_plane_m'])
                pf=group[['scene_token','sample_token','annotation_token','instance_token','category']].copy()
                pf['camera_channel']=c
                pf['camera_minus_sample_dt_s']=(r['timestamp']-int(group.timestamp.iloc[0]))/1e6
                for name,val in fields.items():pf[name]=val
                projections.append(pf)
        proj=pd.concat(projections,ignore_index=True)
        writers.put('camera_projections',proj.copy())
        front=proj[proj.camera_channel==config['canonical_camera']].drop(columns=['scene_token','sample_token','instance_token','category'])
        obj=obj.merge(front,on='annotation_token',validate='one_to_one')
        writers.put('objects',obj.copy())
        trajectory=[]
        for token,g in obj.groupby('instance_token',sort=False):
            row={'instance_token':token,'scene_token':st,'category':g.category.iloc[0],'n_annotations':len(g),
                'duration':float((g.timestamp.max()-g.timestamp.min())/1e6),'n_scenes':g.scene_token.nunique(),
                'mean_speed':g.object_speed.mean(),'max_speed':g.object_speed.max(),
                'mean_relative_speed':g.relative_speed.mean(),'max_relative_speed':g.relative_speed.max(),
                'mean_distance':g.distance_xy.mean(),'min_distance':g.distance_xy.min(),'max_distance':g.distance_xy.max(),
                'visibility_range':g.visibility.max()-g.visibility.min(),'visibility_min':g.visibility.min(),'visibility_max':g.visibility.max(),
                'fraction_velocity_valid':g.velocity_valid.mean(),'n_valid_velocity':int(g.velocity_valid.sum()),
                'valid_temporal_neighbors':int(g.velocity_neighbors.sum()),'median_annotation_dt_s':g.annotation_next_dt_s.median(),
                'max_annotation_dt_s':g.annotation_next_dt_s.max()}
            trajectory.append(row)
        traj=pd.DataFrame(trajectory);writers.put('trajectories',traj)
        sc={'scene_token':st,'scene_name':scene['name'],'duration':float((ts[-1]-ts[0])/1e6),'n_samples':len(samples),
            'n_annotations':len(obj),'n_instances':len(traj),'category_counts':json.dumps(obj.category.value_counts().to_dict(),sort_keys=True),
            'ego_distance_travelled':float(np.linalg.norm(np.diff(position,axis=0),axis=1).sum()),
            'ego_mean_speed':ego.ego_speed.mean(),'ego_max_speed':ego.ego_speed.max(),
            'ego_yaw_change':float(ego_yaw[-1]-ego_yaw[0]),'ego_yaw_total_variation':float(abs(np.diff(ego_yaw)).sum()),
            'ego_max_yaw_rate':float(abs(ego.ego_yaw_rate).max()),
            'moving_object_fraction':float((obj.loc[obj.velocity_valid,'object_speed']>config['moving_speed_m_s']).mean()),
            'n_moving_trajectories':int((traj.max_speed>config['moving_speed_m_s']).sum()),
            'relative_speed_mean':obj.relative_speed.mean(),'relative_speed_max':obj.relative_speed.max(),
            'lateral_velocity_p95':obj.relative_lateral_velocity.quantile(.95),'distance_min':obj.distance_xy.min(),
            'distance_p05':obj.distance_xy.quantile(.05),'distance_p95':obj.distance_xy.quantile(.95),'distance_max':obj.distance_xy.max(),
            'visibility_coverage':json.dumps(obj.visibility_token.value_counts().to_dict(),sort_keys=True),
            'motion_coverage':json.dumps({'valid':int(obj.velocity_valid.sum()),'moving':int((obj.object_speed>config['moving_speed_m_s']).sum())})}
        writers.put('scenes',pd.DataFrame([sc]))
        print(f'Constructed scene {scene_i+1}/{len(scenes)}: {len(obj)} annotations',flush=True)
    writers.close()
    validation['source_tables']=store.source_manifest()
    (out/'construction_validation.json').write_text(json.dumps(validation,indent=2)+'\n')
    return validation
