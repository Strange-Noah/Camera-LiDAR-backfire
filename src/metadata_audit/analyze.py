"""Hierarchical descriptive support diagnostics; no inferential p-values or score."""
import itertools
import json
from pathlib import Path
import numpy as np
import pandas as pd
import pyarrow.parquet as pq
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

MOTION=['object_speed','ego_speed','ego_yaw_rate','relative_speed','relative_radial_velocity','relative_lateral_velocity']
CONT=['distance_xy','width','length','height','object_speed','ego_speed','relative_speed','relative_radial_velocity','relative_lateral_velocity','abs_ego_yaw_rate','num_lidar_pts','projected_width_px']
BINV=['distance_xy','width','relative_lateral_velocity','num_lidar_pts','abs_ego_yaw_rate','projected_width_px']
PAIRS=[('distance_xy','relative_lateral_velocity','distance_x_lateral_velocity'),('width','relative_lateral_velocity','size_x_lateral_velocity'),('distance_xy','visibility','distance_x_visibility'),('distance_xy','num_lidar_pts','distance_x_lidar_support'),('relative_lateral_velocity','abs_ego_yaw_rate','lateral_motion_x_ego_yaw_rate'),('projected_width_px','relative_lateral_velocity','projected_width_x_lateral_velocity')]


def clean(value):
    if isinstance(value,dict):return {str(k):clean(v) for k,v in value.items()}
    if isinstance(value,(list,tuple)):return [clean(v) for v in value]
    if isinstance(value,np.generic):return clean(value.item())
    if isinstance(value,float) and not np.isfinite(value):return None
    return value


def save_json(path,value):
    Path(path).write_text(json.dumps(clean(value),indent=2,allow_nan=False)+'\n')


def stats(series):
    s=pd.to_numeric(series,errors='coerce').replace([np.inf,-np.inf],np.nan)
    q=s.quantile([.05,.25,.5,.75,.95])
    return {'count':int(s.notna().sum()),'missing_count':int(s.isna().sum()),'missing_rate':float(s.isna().mean()),
        'p05':q.loc[.05],'p25':q.loc[.25],'p50':q.loc[.5],'p75':q.loc[.75],'p95':q.loc[.95],
        'IQR':q.loc[.75]-q.loc[.25],'R90':q.loc[.95]-q.loc[.05],'min':s.min(),'max':s.max()}


def support(g,cfg):
    na=g.annotation_token.nunique() if 'annotation_token' in g else len(g)
    nt=g.instance_token.nunique() if 'instance_token' in g else 0
    ns=g.scene_token.nunique()
    scene=g.groupby('scene_token',observed=True).size().sort_values(ascending=False)
    traj=g.groupby('instance_token',observed=True).size().sort_values(ascending=False) if 'instance_token' in g else pd.Series(dtype=float)
    denom=len(g)
    row={'N_ann':int(na),'N_traj':int(nt),'N_scene':int(ns)}
    for k in [1,3,10]:row[f'top{k}_scene_fraction']=float(scene.head(k).sum()/denom) if denom else np.nan
    for k in [1,10]:row[f'top{k}_trajectory_fraction']=float(traj.head(k).sum()/denom) if denom else np.nan
    s=cfg['support']
    row['status']='SUPPORTED' if nt>=s['supported_min_trajectories'] and ns>=s['supported_min_scenes'] and row['top1_scene_fraction']<=s['supported_max_top1_scene_fraction'] else ('WEAK' if nt>=s['weak_min_trajectories'] and ns>=s['weak_min_scenes'] else 'SPARSE')
    return row


def quantile_bin(series,cfg):
    s=series.replace([np.inf,-np.inf],np.nan)
    edges=np.unique(s.dropna().quantile(cfg['quantiles']).to_numpy())
    edges=edges[np.isfinite(edges)]
    if len(edges)<2:
        labels=pd.Series(np.where(s.notna(),'constant','missing'),index=s.index).mask(s.isna())
        return labels, {'edges':edges.tolist(),'labels':['constant'] if len(edges) else ['unavailable'],'collapsed':True}
    names=['low','middle','high'] if len(edges)==4 else [f'q{i+1}' for i in range(len(edges)-1)]
    bins=pd.cut(s,edges,labels=names,include_lowest=True,right=True)
    return bins, {'edges':edges.tolist(),'labels':names,'collapsed':len(edges)!=4,'boundary':'right-closed; first includes minimum'}


def grouped_stats(frame, groups, variables):
    rows=[]
    for keys,g in frame.groupby(groups,observed=True,dropna=False):
        if not isinstance(keys,tuple):keys=(keys,)
        base=dict(zip(groups,keys))
        for var in variables:
            rows.append({**base,'variable':var,**stats(g[var])})
    return pd.DataFrame(rows)


def hist(series,path,label):
    x=pd.to_numeric(series,errors='coerce').replace([np.inf,-np.inf],np.nan).dropna()
    fig,ax=plt.subplots(figsize=(7,4))
    if len(x):ax.hist(x,bins=60,color='#287d8e')
    ax.set(xlabel=label,ylabel='Descriptive observations (not independent trials)')
    ax.set_title(f'n valid={len(x):,}; missing={len(series)-len(x):,}')
    fig.tight_layout();fig.savefig(path,dpi=140);plt.close(fig)


def boxplot(frame,category,variables,path):
    cats=sorted(frame[category].dropna().unique())
    fig,axes=plt.subplots(len(variables),1,figsize=(max(9,len(cats)*.42),4*len(variables)),squeeze=False)
    for ax,var in zip(axes.flat,variables):
        data=[frame.loc[frame[category]==c,var].dropna().to_numpy() for c in cats]
        ax.boxplot(data,tick_labels=cats,showfliers=False)
        ax.tick_params(axis='x',labelrotation=75);ax.set_ylabel(var);ax.set_title('Descriptive quartiles; repeated observations are dependent')
    fig.tight_layout();fig.savefig(path,dpi=120);plt.close(fig)


def matrix_plot(frame,x,y,path):
    fig,axes=plt.subplots(1,3,figsize=(12,4))
    for ax,metric in zip(axes,['N_ann','N_traj','N_scene']):
        pivot=frame.pivot(index=x,columns=y,values=metric).reindex(index=frame[x].drop_duplicates(),columns=frame[y].drop_duplicates())
        im=ax.imshow(pivot.fillna(0).to_numpy(),aspect='auto',cmap='viridis',vmin=0)
        ax.set_xticks(range(len(pivot.columns)),pivot.columns,rotation=45)
        ax.set_yticks(range(len(pivot.index)),pivot.index);ax.set_title(metric)
        ax.set_xlabel(y);ax.set_ylabel(x)
        for i in range(len(pivot.index)):
            for j in range(len(pivot.columns)):
                val=pivot.iloc[i,j]
                ax.text(j,i,f'{int(val):,}',ha='center',va='center',fontsize=8,color='white' if val < .5*pivot.max().max() else 'black')
        fig.colorbar(im,ax=ax,shrink=.7)
    fig.tight_layout();fig.savefig(path,dpi=140);plt.close(fig)


def analyze(tables,output,version,cfg):
    tables=Path(tables);out=Path(output);out.mkdir(parents=True,exist_ok=True)
    for name in ['motion','geometry','visibility','independence','timing','joint','identifiability','mismatch_proxy','gates']:(out/name).mkdir(exist_ok=True)
    obj=pd.read_parquet(tables/'objects.parquet')
    for c in ['scene_token','sample_token','instance_token','category','visibility_token']:obj[c]=obj[c].astype('category')
    traj=pd.read_parquet(tables/'trajectories.parquet');scenes=pd.read_parquet(tables/'scenes.parquet')
    ego=pd.read_parquet(tables/'ego_samples.parquet');timing=pd.read_parquet(tables/'sensor_timing.parquet')
    intervals=pd.read_parquet(tables/'sensor_intervals.parquet')
    specs={}
    for var in BINV+['object_speed','ego_speed','relative_speed','relative_radial_velocity']:
        obj[var+'_bin'],specs[var]=quantile_bin(obj[var],cfg)
    obj['visibility_bin']=obj.visibility.map({1.0:'v0-40',2.0:'v40-60',3.0:'v60-80',4.0:'v80-100'})
    specs['visibility']={'labels':['v0-40','v40-60','v60-80','v80-100'],'ordinal':True}
    save_json(out/'bin_definitions.json',specs)
    # Constructability: timing is a sensor-pair population, not duplicated onto objects.
    critical=['distance_xy','width','object_speed','ego_speed','relative_lateral_velocity','ego_yaw_rate','visibility','num_lidar_pts']
    rows=[]
    for var in critical+['projected_width_px']:
        mask=np.isfinite(obj[var])
        if var in ['distance_xy','width','projected_width_px']:mask &= obj[var]>0
        if var=='num_lidar_pts':mask &= obj[var]>=0
        rows.append({'variable':var,'population':'annotations','valid_count':int(mask.sum()),'missing_count':int((~mask).sum()),'valid_fraction':float(mask.mean()),'N_trajectories_valid':obj.loc[mask,'instance_token'].nunique(),'N_scenes_valid':obj.loc[mask,'scene_token'].nunique()})
    valid_t=timing.timing_valid & np.isfinite(timing.camera_minus_lidar_dt_s)
    rows.append({'variable':'camera_minus_lidar_dt_s','population':'keyframe camera-lidar pairs','valid_count':int(valid_t.sum()),'missing_count':int((~valid_t).sum()),'valid_fraction':float(valid_t.mean()),'N_trajectories_valid':None,'N_scenes_valid':timing.loc[valid_t,'scene_token'].nunique()})
    constructability=pd.DataFrame(rows);constructability.to_csv(out/'gates/constructability.csv',index=False)
    summary={v:stats(obj[v]) for v in CONT+['ego_yaw_rate','relative_orientation','visibility']}
    pd.DataFrame.from_dict(summary,orient='index').rename_axis('variable').to_csv(out/'gates/dynamic_range.csv')
    # Every marginal regime includes scene/trajectory support and concentration.
    regimes=[];scene_regimes=[]
    for var,spec in specs.items():
        col=var+'_bin'
        for label in spec['labels']:
            g=obj[obj[col]==label]
            regimes.append({'variable':var,'regime':label,**support(g,cfg)})
            for st,h in g.groupby('scene_token',observed=True):scene_regimes.append({'scene_token':st,'variable':var,'regime':label,**support(h,cfg)})
    regime=pd.DataFrame(regimes)
    regime.to_csv(out/'independence/regime_concentration.csv',index=False)
    pd.DataFrame(scene_regimes).to_csv(out/'independence/scene_regime_coverage.csv',index=False)
    regime[regime.variable.isin(MOTION+['abs_ego_yaw_rate'])].to_csv(out/'motion/motion_coverage.csv',index=False)
    regime[regime.variable.isin(['distance_xy','width','projected_width_px'])].to_csv(out/'geometry/geometry_coverage.csv',index=False)
    names={'object_speed':'object_speed','ego_speed':'ego_speed','ego_yaw_rate':'ego_yaw_rate','relative_speed':'relative_speed','relative_radial_velocity':'radial_velocity','relative_lateral_velocity':'lateral_velocity'}
    for var,name in names.items():
        data=ego[var] if var in ['ego_speed','ego_yaw_rate'] else obj[var]
        hist(data,out/f'motion/{name}.png',f'{var} ({"rad/s" if "yaw" in var else "m/s"})')
    boxplot(obj,'category',['object_speed','relative_lateral_velocity'],out/'motion/motion_by_class.png')
    grouped_stats(obj,['category'],MOTION).to_csv(out/'motion/motion_by_class.csv',index=False)
    grouped_stats(obj,['instance_token'],['object_speed','relative_lateral_velocity']).to_csv(out/'motion/motion_by_trajectory.csv',index=False)
    grouped_stats(ego,['scene_token'],['ego_speed','ego_yaw_rate']).to_csv(out/'motion/ego_by_scene.csv',index=False)
    yaw_edges=cfg['yaw_regime_edges_rad_s']
    physical=[]
    for lo,hi,label in [(0,yaw_edges[0],'straight'),(yaw_edges[0],yaw_edges[1],'moderate_turn'),(yaw_edges[1],np.inf,'high_yaw')]:
        g=obj[(obj.abs_ego_yaw_rate>=lo)&(obj.abs_ego_yaw_rate<hi)]
        physical.append({'regime':label,'lower_rad_s':lo,'upper_rad_s':hi,**support(g,cfg)})
    pd.DataFrame(physical).to_csv(out/'motion/physical_yaw_regimes.csv',index=False)
    save_json(out/'motion/motion_summary.json',{'annotation_weighted':{v:summary[v] for v in MOTION},'sample_weighted_ego':{v:stats(ego[v]) for v in ['ego_speed','ego_yaw_rate']},'trajectory_mean_speed':stats(traj.mean_speed),'physical_thresholds_are_design_choices':True})
    for var,name,unit in [('distance_xy','distance','m'),('relative_orientation','relative_orientation','rad'),('projected_width_px','projected_width','px')]:hist(obj[var],out/f'geometry/{name}.png',f'{var} ({unit})')
    boxplot(obj,'category',['distance_xy'],out/'geometry/distance_by_class.png')
    boxplot(obj,'category',['width','length','height'],out/'geometry/physical_size_by_class.png')
    grouped_stats(obj,['category'],['distance_xy','width','length','height']).to_csv(out/'geometry/geometry_by_class.csv',index=False)
    grouped_stats(obj,['scene_token'],['distance_xy']).to_csv(out/'geometry/distance_by_scene.csv',index=False)
    traj[['instance_token','category','min_distance','max_distance','mean_distance']].to_csv(out/'geometry/trajectory_distance_range.csv',index=False)
    folded=np.minimum(abs(obj.relative_orientation),np.pi-abs(obj.relative_orientation))
    orientation=[]
    for name,mask in [('frontal_or_rear',folded<np.pi/6),('oblique',(folded>=np.pi/6)&(folded<np.pi/3)),('lateral',folded>=np.pi/3)]:orientation.append({'regime':name,**support(obj[mask],cfg)})
    pd.DataFrame(orientation).to_csv(out/'geometry/orientation_coverage.csv',index=False)
    save_json(out/'geometry/geometry_summary.json',{'canonical_distance':'distance_xy (ego origin, sample timestamp)','canonical_camera':cfg['canonical_camera'],'statistics':{v:summary[v] for v in ['distance_xy','width','length','height','projected_width_px']},'projection':'Unclipped 3D-box envelope at camera ego pose; box remains at keyframe annotation time, no occlusion measurement or object-time compensation.'})
    hist(obj.visibility,out/'visibility/visibility_distribution.png','nuScenes ordinal visibility token (not continuous fraction)')
    boxplot(obj,'distance_xy_bin',['visibility'],out/'visibility/visibility_by_distance.png')
    hist(obj.num_lidar_pts,out/'visibility/lidar_points.png','num_lidar_pts')
    fig,ax=plt.subplots(figsize=(7,4));ax.hexbin(obj.distance_xy,obj.num_lidar_pts,gridsize=50,bins='log',mincnt=1);ax.set(xlabel='distance_xy (m)',ylabel='num_lidar_pts');fig.tight_layout();fig.savefig(out/'visibility/lidar_points_vs_distance.png',dpi=140);plt.close(fig)
    grouped_stats(obj,['category','distance_xy_bin'],['num_lidar_pts','visibility']).to_csv(out/'visibility/lidar_points_by_class_distance.csv',index=False)
    grouped_stats(obj,['scene_token'],['visibility']).to_csv(out/'visibility/visibility_by_scene.csv',index=False)
    # Projection file is streamed by scene row-group; don't materialize six copies of all annotations.
    camera_rows=[]
    for batch in pq.ParquetFile(tables/'camera_projections.parquet').iter_batches(batch_size=65536):
        frame=batch.to_pandas()
        for keys,g in frame.groupby(['scene_token','camera_channel'],observed=True):
            camera_rows.append({'scene_token':keys[0],'camera_channel':keys[1],'N_ann':len(g),'N_valid_projection':int(g.projection_valid.sum()),'N_geometric_fov':int(g.camera_visible.sum()),'N_partial':int(g.partially_outside_image.sum()),'N_near_plane_crossing':int(g.near_plane_crossing.sum())})
    camera=pd.DataFrame(camera_rows).groupby(['scene_token','camera_channel'],as_index=False).sum()
    camera.to_csv(out/'visibility/camera_observability.csv',index=False)
    save_json(out/'visibility/visibility_summary.json',{'visibility_counts':obj.visibility_token.value_counts().to_dict(),'lidar_support':stats(obj.num_lidar_pts),'geometric_projection_counts':camera.groupby('camera_channel').sum(numeric_only=True).to_dict(),'limits':'Visibility is object-level categorical metadata, not per-camera occlusion. Image content, blur, real detection support and actual image-box widths need sensor blobs.'})
    for var,name in [('n_annotations','trajectory_length'),('duration','trajectory_duration'),('n_annotations','annotations_per_trajectory')]:hist(traj[var],out/f'independence/{name}.png',var)
    for var,name in [('n_instances','instances_per_scene'),('n_annotations','observations_per_scene')]:hist(scenes[var],out/f'independence/{name}.png',var)
    traj.to_csv(out/'independence/trajectory_support.csv',index=False)
    grouped_stats(traj,['category'],['n_annotations','duration','fraction_velocity_valid']).to_csv(out/'independence/trajectory_by_class.csv',index=False)
    scenes.to_csv(out/'independence/scene_support.csv',index=False)
    save_json(out/'independence/independence_summary.json',{'n_annotations':len(obj),'n_trajectories':len(traj),'n_scenes':len(scenes),'instances_spanning_multiple_scenes':int((traj.n_scenes!=1).sum()),'trajectory_length':stats(traj.n_annotations),'trajectory_duration_s':stats(traj.duration),'valid_velocity_fraction_by_trajectory':stats(traj.fraction_velocity_valid),'warning':'Unique trajectories and scenes are cluster counts, not a proven statistical effective sample size. Instances in the same scene may remain dependent. C10_scene equals 1 for mini by construction.'})
    # Timing: paired keyframes for offsets, complete sample_data chains for sampling intervals.
    for var,name in [('camera_minus_lidar_dt_s','native_camera_lidar_offset'),('abs_camera_lidar_dt_s','absolute_offset')]:hist(timing[var]*1000,out/f'timing/{name}.png',var+' (ms)')
    plot_t=timing.copy();plot_t['offset_ms']=plot_t.camera_minus_lidar_dt_s*1000
    boxplot(plot_t,'camera_channel',['offset_ms'],out/'timing/offset_by_camera.png')
    grouped_stats(timing,['camera_channel','scene_token'],['camera_minus_lidar_dt_s','abs_camera_lidar_dt_s']).to_csv(out/'timing/offset_by_scene.csv',index=False)
    grouped_stats(timing,['camera_channel'],['camera_minus_lidar_dt_s','abs_camera_lidar_dt_s']).to_csv(out/'timing/offset_by_camera.csv',index=False)
    ints=grouped_stats(intervals,['channel'],['next_dt_s']);ints.to_csv(out/'timing/sampling_intervals.csv',index=False)
    interval_flags=[]
    for c,g in intervals.groupby('channel'):
        x=g.next_dt_s.dropna();q1,q3=x.quantile([.25,.75]);iqr=q3-q1
        flags=(x<q1-1.5*iqr)|(x>q3+1.5*iqr)
        interval_flags.append({'channel':c,'n_intervals':len(x),'tukey_outlier_count':int(flags.sum()),'method':'descriptive Tukey 1.5 IQR; not defective sensor classification'})
    pd.DataFrame(interval_flags).to_csv(out/'timing/interval_outliers.csv',index=False)
    hist(intervals.loc[intervals.channel.str.startswith('CAM'),'next_dt_s']*1000,out/'timing/camera_sampling_interval.png','All camera next intervals (ms)')
    hist(intervals.loc[intervals.channel=='LIDAR_TOP','next_dt_s']*1000,out/'timing/lidar_sampling_interval.png','LiDAR next intervals (ms)')
    ac=[]
    for (s,c),g in timing.groupby(['scene_token','camera_channel']):
        x=g.sort_values('camera_timestamp').camera_minus_lidar_dt_s
        for lag in range(1,min(11,len(x)//2)):
            ac.append({'scene_token':s,'camera_channel':c,'lag_keyframes':lag,'autocorrelation':x.autocorr(lag=lag),'n_pairs':len(x)-lag})
    pd.DataFrame(ac).to_csv(out/'timing/offset_autocorrelation.csv',index=False)
    scene_med=timing.groupby(['camera_channel','scene_token']).camera_minus_lidar_dt_s.median()
    camera_med=timing.groupby('camera_channel').camera_minus_lidar_dt_s.median()
    centered=timing.camera_minus_lidar_dt_s-timing.groupby('camera_channel').camera_minus_lidar_dt_s.transform('median')
    save_json(out/'timing/timing_summary.json',{'offset_s':stats(timing.camera_minus_lidar_dt_s),'absolute_offset_s':stats(timing.abs_camera_lidar_dt_s),'camera_median_range_s':float(camera_med.max()-camera_med.min()),'camera_scene_medians':stats(scene_med),'within_camera_residual_s':stats(centered),'sampling_outliers':interval_flags,'pattern_assessment':'Compare per-camera and per-scene medians/ranges; nonzero within-camera spread rules out an exactly fixed offset. Lag-1..10 autocorrelation is diagnostic only: short scene windows do not establish periodicity. No artificial corruption has been applied.'})
    # Joint coverage explicitly enumerates zero cells; distinct counts do not add across bins.
    joint_rows=[];joint_summaries={}
    for x,y,name in PAIRS:
        records=[]
        for bx,by in itertools.product(specs[x]['labels'],specs[y]['labels']):
            g=obj[(obj[x+'_bin']==bx)&(obj[y+'_bin']==by)]
            records.append({x:bx,y:by,**support(g,cfg)})
        frame=pd.DataFrame(records)
        frame.to_csv(out/f'joint/{name}_coverage.csv',index=False)
        for metric,suffix in [('N_ann','annotations'),('N_traj','trajectories'),('N_scene','scenes')]:
            frame.pivot(index=x,columns=y,values=metric).reindex(index=specs[x]['labels'],columns=specs[y]['labels']).to_csv(out/f'joint/{name}_{suffix}.csv')
        matrix_plot(frame,x,y,out/f'joint/{name}.png')
        joint_summaries[name]={'total_bins':len(frame),'occupied_bins':int((frame.N_ann>0).sum()),'supported_bins':int((frame.status=='SUPPORTED').sum()),'weak_bins':int((frame.status=='WEAK').sum()),'sparse_bins':int((frame.status=='SPARSE').sum())}
        joint_rows.extend([{'analysis':name,**r} for r in records])
    pd.DataFrame(joint_rows).to_csv(out/'joint/all_joint_coverage.csv',index=False)
    class_joint=[]
    for c in obj.category.cat.categories:
        g=obj[obj.category==c]
        for db,mb in itertools.product(specs['distance_xy']['labels'],specs['relative_lateral_velocity']['labels']):
            class_joint.append({'category':c,'distance_regime':db,'motion_regime':mb,**support(g[(g.distance_xy_bin==db)&(g.relative_lateral_velocity_bin==mb)],cfg)})
    pd.DataFrame(class_joint).to_csv(out/'joint/class_x_motion_x_distance.csv',index=False)
    save_json(out/'joint/joint_summary.json',joint_summaries)
    print('Marginal, timing and joint diagnostics complete',flush=True)
    # Dependence at multiple weighting levels: no independence-based p-values.
    corr=obj[CONT].corr(method='spearman');corr.to_csv(out/'identifiability/spearman_matrix.csv')
    obj.groupby('instance_token',observed=True)[CONT].median().corr(method='spearman').to_csv(out/'identifiability/spearman_trajectory_medians.csv')
    obj.groupby('scene_token',observed=True)[CONT].median().corr(method='spearman').to_csv(out/'identifiability/spearman_scene_medians.csv')
    within=[]
    for st,g in obj.groupby('scene_token',observed=True):
        for x,y,_ in PAIRS:
            within.append({'scene_token':st,'x':x,'y':y,'spearman':g[[x,y]].corr(method='spearman').iloc[0,1],'n_complete':int(g[[x,y]].notna().all(axis=1).sum())})
    pd.DataFrame(within).to_csv(out/'identifiability/spearman_within_scene.csv',index=False)
    grouped_stats(obj,['category'],['distance_xy','width','relative_lateral_velocity','visibility','num_lidar_pts']).to_csv(out/'identifiability/class_conditioned_statistics.csv',index=False)
    regime.to_csv(out/'identifiability/scene_concentration.csv',index=False)
    dominance=[]
    for var in BINV:
        for label in specs[var]['labels']:
            g=obj[obj[var+'_bin']==label];counts=g.category.value_counts()
            dominance.append({'variable':var,'regime':label,'top_class':str(counts.index[0]) if len(g) else None,'top_class_fraction':float(counts.max()/len(g)) if len(g) else np.nan,**support(g,cfg)})
    pd.DataFrame(dominance).to_csv(out/'identifiability/class_dominance.csv',index=False)
    matches=[]
    for target,controls in [('relative_lateral_velocity',['distance_xy','width']),('distance_xy',['relative_lateral_velocity','width'])]:
        levels=specs[target]['labels']
        if len(levels)<2:continue
        for keys,g in obj.groupby(['category']+[c+'_bin' for c in controls],observed=True):
            low=g[g[target+'_bin']==levels[0]];high=g[g[target+'_bin']==levels[-1]]
            low_s=set(low.scene_token);high_s=set(high.scene_token)
            ls,hs=support(low,cfg),support(high,cfg)
            common_s=low_s&high_s
            # Require the same scenes to support both extremes, within semantic class.
            low_common=low[low.scene_token.isin(common_s)];high_common=high[high.scene_token.isin(common_s)]
            passed=(len(common_s)>=cfg['matched_support_min_scenes_per_extreme'] and low_common.instance_token.nunique()>=cfg['matched_support_min_trajectories_per_extreme'] and high_common.instance_token.nunique()>=cfg['matched_support_min_trajectories_per_extreme'])
            matches.append({'target':target,'category':keys[0],controls[0]+'_bin':keys[1],controls[1]+'_bin':keys[2],
                'target_p05':g[target].quantile(.05),'target_p95':g[target].quantile(.95),'common_scenes_low_high':len(common_s),
                'common_trajectories_low_high':len(set(low.instance_token)&set(high.instance_token)),
                'low_N_traj_in_common_scenes':low_common.instance_token.nunique(),'high_N_traj_in_common_scenes':high_common.instance_token.nunique(),
                **{'low_'+k:v for k,v in ls.items()},**{'high_'+k:v for k,v in hs.items()},'overlap_supported':passed})
    matched=pd.DataFrame(matches);matched.to_csv(out/'identifiability/matched_support_summary.csv',index=False)
    match_counts={target:int(((matched.target==target)&matched.overlap_supported).sum()) if len(matched) else 0 for target in ['relative_lateral_velocity','distance_xy']}
    strong=[]
    for x,y,_ in PAIRS:
        rho=obj[[x,y]].corr(method='spearman').iloc[0,1]
        if np.isfinite(rho) and abs(rho)>=cfg['dependence_warning_abs_spearman']:strong.append({'x':x,'y':y,'rho':rho})
    confounding='SEVERE CONFOUNDING' if any(v==0 for v in match_counts.values()) else ('MANAGEABLE CONFOUNDING' if strong or any(r['top_class_fraction']>cfg['class_dominance_warning_fraction'] for r in dominance) else 'LOW CONFOUNDING')
    save_json(out/'identifiability/identifiability_summary.json',{'classification':confounding,'strong_dependence_flags':strong,'matched_strata_with_same_scene_overlap':match_counts,'interpretation':'Coarse within-class quantile strata are support diagnostics, not exact causal matching. No overlap under these bins does not prove no overlap under all possible designs. No Pearson or independence-based p-values are reported.'})
    # Proxy is evaluated for four separate hypothetical dt values, not corruption runs.
    proxy_writer=None;proxy_rows=[]
    for dt in cfg['temporal_perturbations_s']:
        p=obj[['annotation_token','instance_token','scene_token','width','relative_lateral_velocity']].copy()
        p['delta_t_s']=dt;p['r_physical']=p.relative_lateral_velocity*dt/p.width
        p.loc[~(np.isfinite(p.r_physical)&(p.width>0)),'r_physical']=np.nan
        p['proxy_valid']=p.r_physical.notna()
        p['r_regime']=pd.cut(p.r_physical,[0,.1,.5,1,np.inf],right=False,labels=['r<0.1','0.1<=r<0.5','0.5<=r<1','r>=1']).astype('string')
        import pyarrow as pa
        for c in ['annotation_token','instance_token','scene_token']:p[c]=p[c].astype('string')
        tab=pa.Table.from_pandas(p,preserve_index=False)
        if proxy_writer is None:proxy_writer=pq.ParquetWriter(out/'mismatch_proxy/mismatch_proxy.parquet',tab.schema,compression='zstd')
        proxy_writer.write_table(tab)
        for label in ['r<0.1','0.1<=r<0.5','0.5<=r<1','r>=1']:
            proxy_rows.append({'delta_t_s':dt,'regime':label,**support(p[p.r_regime==label],cfg)})
        hist(p.r_physical,out/f'mismatch_proxy/r_{int(dt*1000)}ms.png','|v_lateral| × dt / physical width (dimensionless proxy)')
    proxy_writer.close();proxy=pd.DataFrame(proxy_rows);proxy.to_csv(out/'mismatch_proxy/r_regime_coverage.csv',index=False)
    save_json(out/'mismatch_proxy/mismatch_summary.json',{'coverage':proxy_rows,'meaning':'First-order translational normalized displacement only; not fusion degradation. Ignores ego rotation, acceleration, occlusion, object rotation and orientation-dependent image extent. 0.5 s extrapolation particularly approximate. r_image deliberately not claimed.'})
    gates=gate_report(obj,traj,scenes,timing,constructability,regime,joint_summaries,match_counts,confounding,proxy,summary,version,cfg)
    save_json(out/'gating_report.json',gates)
    save_json(out/'audit_config.json',cfg)
    report(out,version,scenes,obj,traj,summary,gates,joint_summaries,match_counts,proxy,constructability)
    print('Audit output complete:',out,flush=True)
    return gates


def gate_report(obj,traj,scenes,timing,construct,regime,joint,matched,confounding,proxy,summary,version,cfg):
    final=version=='v1.0-trainval' and len(scenes)==850
    essential=construct[construct.variable!='projected_width_px']
    missing=essential.loc[essential.valid_count==0,'variable'].tolist()
    no_range=[v for v in ['distance_xy','relative_lateral_velocity','width'] if not np.isfinite(summary[v]['R90']) or summary[v]['R90']<=0]
    base_joint=[v for k,v in joint.items() if not k.startswith('projected')]
    weak_joint=any(v['supported_bins']<v['total_bins'] for v in base_joint)
    independent=regime[regime.variable.isin(['distance_xy','relative_lateral_velocity','width','abs_ego_yaw_rate'])]
    weak_ind=bool((independent.status!='SUPPORTED').any())
    proxy_good=any((g.status=='SUPPORTED').all() and len(g)==4 for _,g in proxy.groupby('delta_t_s'))
    any_large=bool(((proxy.regime=='r>=1')&(proxy.N_scene>=cfg['support']['weak_min_scenes'])).any())
    limits=['Observational metadata cannot establish fusion degradation, backfire, causal attribution or a normalization law.',
            'Clusters are nested and dependent; counts are not an effective independent sample size.',
            'Projection uses keyframe box position and camera-time ego pose; native object displacement is not compensated.',
            'Metadata visibility is not per-camera occlusion; LiDAR support is a provided count, not recomputed raw sensor quality.',
            'Velocity is a finite-difference estimate, not instantaneous ground truth; edges, gaps and annotation noise matter.',
            'Physical-width proxy ignores rotational image motion and orientation-dependent image scale.']
    if not final:limits.append('MINI DEVELOPMENT ONLY: final dataset-wide gating is withheld until all 850 trainval scenes pass the same pipeline.')
    if weak_ind:limits.append('One or more marginal quantile regimes fail configured independent-support diagnostics.')
    if weak_joint:limits.append('Some requested joint bins fail configured support; broad comparisons need restriction.')
    if any(v==0 for v in matched.values()):limits.append('At least one comparison has no qualifying within-class, same-scene overlap in the current coarse bins; that comparison is unsupported under this design.')
    if not proxy_good:limits.append('No single tested dt has all four mismatch regimes meeting configured independent-support thresholds.')
    # Lack of a large-r band alone restricts the sweep; it is not a structural failure.
    any_nontrivial=bool(((proxy.regime!='r<0.1') & (proxy.N_ann>0)).any())
    structural=bool(missing or 'distance_xy' in no_range or 'relative_lateral_velocity' in no_range or not any_nontrivial)
    decision='NO-GO' if structural else ('CONDITIONAL GO' if weak_ind or weak_joint or confounding!='LOW CONFOUNDING' or not proxy_good else 'GO')
    return {'scope':version,'final_gating_eligible':final,'n_scenes':len(scenes),'constructability':{'status':'FAIL' if missing else 'PASS','unconstructible_essential':missing,'evidence':'gates/constructability.csv'},
        'motion_diversity':{'status':'FAIL' if 'relative_lateral_velocity' in no_range else 'VARIATION PRESENT','evidence':'motion/motion_summary.json'},
        'geometry_diversity':{'status':'FAIL' if any(v in no_range for v in ['distance_xy','width']) else 'VARIATION PRESENT','evidence':'geometry/geometry_summary.json'},
        'visibility_diversity':{'observed_tokens':int(obj.visibility.nunique()),'evidence':'visibility/visibility_summary.json'},
        'trajectory_independence':{'status':'CONDITIONAL' if weak_ind else 'SUPPORTED','n_trajectories':len(traj),'evidence':'independence/regime_concentration.csv'},
        'scene_diversity':{'n_scenes':len(scenes),'status':'DESIGN SUPPORT CHECKED; NOT PROVEN INDEPENDENCE'},
        'timing_diversity':{'valid_pairs':int(timing.timing_valid.sum()),'evidence':'timing/timing_summary.json'},
        'joint_coverage':{'status':'CONDITIONAL' if weak_joint else 'SUPPORTED','matrices':joint},
        'identifiability':{'status':confounding,'matched_strata':matched,'evidence':'identifiability/matched_support_summary.csv'},
        'mismatch_proxy_coverage':{'status':'SUPPORTED' if proxy_good else 'CONDITIONAL','large_regime_has_minimum_scene_support':any_large,'evidence':'mismatch_proxy/r_regime_coverage.csv'},
        'overall':decision if final else 'NOT ASSESSED — MINI DEVELOPMENT ONLY',
        'hierarchical_logic':'Essential constructability -> physical variation -> independent support -> joint support -> overlap/confounding -> proxy regimes. No aggregate score. Thresholds are declared project choices; final recommendation remains research judgment.',
        'limitations':limits,'supported_claims':['Metadata constructs the listed geometric and kinematic variables on validity-flagged rows.','Observed descriptive variation and overlap can motivate restricted future tests, conditional on scene/trajectory support.'] if not missing else [],
        'unsupported_claims':['Actual fusion degradation/backfire or superiority of a normalization.','Universal cross-scene/semantic-class generalization from observational coverage.','Causal separation based solely on correlations or coarse matching.']+(['Any dataset-wide suitability claim from mini.'] if not final else []),
        'recommended_next_step':('Inspect supported joint/matched regimes, prespecify scene-held-out comparisons and perturbation sweep; do not auto-deploy BEVFusion. Supplement/reformulate unsupported comparisons before acquisition.' if final else 'Complete mini numerical checks, acquire official trainval metadata only, run identical pipeline on exactly 850 scenes; final gating is blocked until then.'),
        'thresholds':cfg,'automatic_recommendation_requires_research_review':True}


def report(out,version,scenes,obj,traj,summary,gates,joint,matched,proxy,construct):
    lines=['# Metadata audit', '', '## 1. Research question being audited',
        'Can natural geometric/motion variation support separating sensor misalignment, geometric mismatch and later fusion degradation? This audit tests metadata support, not model performance.',
        '', '## 2. Dataset and split',f'`{version}`: **{len(scenes)} scenes, {len(traj):,} instances, {len(obj):,} annotations**. Final gating eligible: **{gates["final_gating_eligible"]}**.',
        '', '## 3. Variable construction',
        'See [methods](../../../docs/metadata_audit_methods.md) and `audit_config.json`. Position and velocities use the global/world horizontal frame, meters and seconds. Canonical Z is ego-origin horizontal distance at sample time. Object size order is W,L,H. World velocity uses central secants with actual elapsed time; endpoints are one-sided. Both adjacent gaps must be ≤ configured maximum. Ego positions/yaw are interpolated from LiDAR ego poses at annotation times without extrapolation; yaw is unwrapped before differences.',
        '`v_rel = v_object − v_ego`; `r_hat = (p_object−p_ego)/Z`; radial = dot(v_rel,r_hat); |lateral| = absolute 2D cross product. This relative velocity excludes rotational apparent motion. Ego-frame bearing rate = signed_lateral/Z − ego_yaw_rate. Missing values remain NaN with validity fields.',
        'Camera transform: R_cal^T [R_ego^T(p_world−t_ego)−t_cal]. Projection requires all box corners in front of the near plane and a nonzero convex-hull intersection with the image. Unclipped box-envelope width is not an occlusion-aware object width. Canonical projected columns use CAM_FRONT; all six cameras are in camera_projections.parquet.',
        f'Constructability: see [CSV](gates/constructability.csv); {int(construct.valid_count.eq(0).sum())} variables with zero valid observations.',
        '', '## 4. Motion diversity']
    for v in ['object_speed','ego_speed','relative_lateral_velocity','abs_ego_yaw_rate']:
        s=summary[v];lines.append(f'- `{v}`: P05={s["p05"]:.5g}, median={s["p50"]:.5g}, P95={s["p95"]:.5g}; missing={s["missing_rate"]:.3%}.')
    lines+=['', 'Annotation-weighted summaries are not independent evidence. Sample-weighted ego distributions and trajectory/class summaries are separate in `motion/`.',
        '', '## 5. Geometric diversity',f'Canonical distance median {summary["distance_xy"]["p50"]:.3f} m; P05–P95 {summary["distance_xy"]["p05"]:.3f}–{summary["distance_xy"]["p95"]:.3f} m. Physical dimensions are conditioned on category; orientation is LOS-relative. See `geometry/`.',
        '', '## 6. Visibility / sensing diversity','See `visibility/`: visibility by class/distance/scene, point support by class × distance, and per-camera FOV geometry. No raw sensor quality or per-camera occlusion is inferred from metadata.',
        '', '## 7. Trajectory and scene independence','See `independence/`: annotation counts, trajectory length/duration, temporal neighbors, valid-velocity fraction, moving trajectories per scene and regime concentration. Unique counts across bins are not additive. C10_scene is uninformative when there are only 10 scenes.',
        '', '## 8. Native sensor timing','Offsets pair six camera keyframes to the same sample’s LiDAR keyframe. Full sample_data chains supply sampling intervals (not just ~2 Hz keyframes). Signed/absolute seconds and raw microseconds are retained. Per-camera/scene distributions, Tukey diagnostics and lag-1..10 autocorrelation are saved; short windows do not prove periodicity.',
        '', '## 9. Joint coverage','Each matrix has annotation, trajectory and scene versions, including zero cells. Bins are annotation-derived global quantiles with tied cutpoints collapsed. `bin_definitions.json` records exact edges. Concentration is in the corresponding coverage CSV.']
    for k,v in joint.items():lines.append(f'- `{k}`: {v["occupied_bins"]}/{v["total_bins"]} occupied, {v["supported_bins"]} SUPPORTED, {v["weak_bins"]} WEAK, {v["sparse_bins"]} SPARSE.')
    lines+=['','## 10. Confounding / identifiability',f'Preliminary diagnostic: **{gates["identifiability"]["status"]}**. Qualifying same-scene, within-class coarse matched strata: `{matched}`.',
        'Spearman matrices use annotation, trajectory-median and scene-median levels; within-scene pair correlations are also saved. No iid p-values or causal claims. Coarse quantile overlap is not formal matching; absence only restricts this design.',
        '', '## 11. Hypothetical mismatch coverage','`r_physical=|v_lateral|×dt/W`, dt ∈ {0.05,0.1,0.2,0.5} s. This excludes rotation and acceleration and is not actual degradation. Each dt/regime reports all three cluster counts in [coverage](mismatch_proxy/r_regime_coverage.csv).',
        '', '## 12. Gate-by-gate assessment','Machine-readable decision and evidence: [gating_report.json](gating_report.json). Configurable project support thresholds are diagnostics, not universal laws. No suitability score is computed.']
    for k in ['constructability','motion_diversity','geometry_diversity','trajectory_independence','joint_coverage','identifiability','mismatch_proxy_coverage']:
        lines.append(f'- {k}: **{gates[k]["status"]}**.')
    lines+=['', '## 13. Overall GO / CONDITIONAL GO / NO-GO',f'**{gates["overall"]}**. Automatic hierarchical recommendation; requires research interpretation. Mini results never participate in the final decision.',
        '', '## 14. Supported claims']+['- '+s for s in gates['supported_claims']]
    lines+=['', '## 15. Unsupported / risky claims']+['- '+s for s in gates['unsupported_claims']+gates['limitations']]
    lines+=['', '## 16. Recommended next experiment',gates['recommended_next_step'],'','No sensor-blob acquisition, GPU provisioning, BEVFusion, corruption injection or inference is triggered by this audit.']
    (out/'README.md').write_text('\n'.join(lines)+'\n')
