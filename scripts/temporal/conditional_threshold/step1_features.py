"""Step 1: per-GT-box features for the T1 common subset.

Input : <data>/gt_common.json(.gz), nuScenes v1.0-trainval metadata (<meta>)
Output: <out>/gt_features.csv.gz  (one row per GT box, box_id = order in gt_common)
        <out>/ego_samples.csv     (ego pose / velocity per sample, used for prediction distances)

Feature definitions follow the metadata audit (src/metadata_audit):
  - object velocity: central difference along the instance's annotation track (gaps <= 1.5 s, else NaN)
  - ego velocity / yaw rate: same difference on the key-frame LIDAR_TOP ego poses
  - v_rel = v_obj - v_ego; radial = v_rel . r_hat (positive = receding); lateral = |r_hat x v_rel|
  - distance_xy: horizontal ego-to-box distance; width = size[0]
  - r500 = lateral relative speed * 0.5 s / width (body-widths shifted in 0.5 s); r_k* use each k's delay
  - night / intersection / rain: keywords in the scene description (as in E1)
Difference from the audit: ego pose is taken at the key-frame LiDAR (not interpolated over all sweeps).
"""
import gc
import os
import time

import numpy as np
import pandas as pd

from common import (DELAY_MS, DET_NAME, EXPECTED, GT_FILE, differences, find_file, get_args, load_json, log,
                    relative_components, save_csv, stream_rows, yaw_of)


def main(a):
    t_start = time.time()
    # ---- GT boxes of the common subset ----
    path = find_file(a.data, GT_FILE)
    assert path, f'{GT_FILE}(.gz) not found in {a.data}'
    GT = load_json(path)
    tokens = sorted(GT)
    rows = [(t, i, g['name'], *g['translation'][:3], *g['size'][:3]) for t in tokens for i, g in enumerate(GT[t])]
    gt = pd.DataFrame(rows, columns=['sample_token', 'gi', 'cls', 'x', 'y', 'z', 'w', 'l', 'h'])
    gt['box_id'] = np.arange(len(gt))
    del GT
    gc.collect()
    log(f'samples: {len(tokens)} (expected {EXPECTED["n_samples"]}); GT boxes: {len(gt)} (expected {EXPECTED["n_boxes"]})')

    # ---- metadata, restricted to the scenes of these samples ----
    small = lambda name: load_json(os.path.join(a.meta, f'{name}.json'))
    scene_tab, sample_tab, log_tab = small('scene'), small('sample'), small('log')
    instance_tab, category_tab, attribute_tab = small('instance'), small('category'), small('attribute')
    sensor_tab, calib_tab = small('sensor'), small('calibrated_sensor')

    tok2sample = {s['token']: s for s in sample_tab}
    target_scene = {tok2sample[t]['scene_token'] for t in tokens}
    samples = [s for s in sample_tab if s['scene_token'] in target_scene]
    sample_set = {s['token'] for s in samples}
    scenes = [s for s in scene_tab if s['token'] in target_scene]
    log(f'scenes: {len(scenes)}; samples in these scenes: {len(samples)}')

    channel = {s['token']: s['channel'] for s in sensor_tab}
    lidar_cal = {c['token'] for c in calib_tab if channel[c['sensor_token']] == 'LIDAR_TOP'}
    sd_rows = stream_rows(os.path.join(a.meta, 'sample_data.json'),
                          keep=lambda r: r['is_key_frame'] and r['calibrated_sensor_token'] in lidar_cal and r['sample_token'] in sample_set)
    sample2pose = {r['sample_token']: r['ego_pose_token'] for r in sd_rows}
    need_pose = set(sample2pose.values())
    pose = {r['token']: r for r in stream_rows(os.path.join(a.meta, 'ego_pose.json'), keep=lambda r: r['token'] in need_pose)}
    ann_rows = stream_rows(os.path.join(a.meta, 'sample_annotation.json'), keep=lambda r: r['sample_token'] in sample_set)

    # ---- ego velocity and yaw rate per scene ----
    parts = []
    for sc in scenes:
        ss = sorted([s for s in samples if s['scene_token'] == sc['token']], key=lambda s: s['timestamp'])
        ts = np.array([s['timestamp'] for s in ss], dtype=np.int64)
        p = [pose[sample2pose[s['token']]] for s in ss]
        pos = np.array([q['translation'] for q in p], dtype=float)
        ev = differences(ts, pos[:, :2], 1.5)
        ew = differences(ts, np.unwrap([yaw_of(q['rotation']) for q in p]), 1.5)
        parts.append(pd.DataFrame({'sample_token': [s['token'] for s in ss], 'scene_token': sc['token'], 'timestamp': ts,
                                   'ego_x': pos[:, 0], 'ego_y': pos[:, 1], 'ego_vx': ev[:, 0], 'ego_vy': ev[:, 1],
                                   'ego_speed': np.linalg.norm(ev, axis=1), 'ego_yaw_rate': ew}))
    ego = pd.concat(parts, ignore_index=True)
    save_csv(ego, os.path.join(a.out, 'ego_samples.csv'), index=False)

    # ---- object velocity, relative motion, r ----
    cat_name = {c['token']: c['name'] for c in category_tab}
    inst_cat = {i['token']: cat_name[i['category_token']] for i in instance_tab}
    attr_name = {x['token']: x['name'] for x in attribute_tab}
    obj = pd.DataFrame({
        'annotation_token': [r['token'] for r in ann_rows],
        'sample_token': [r['sample_token'] for r in ann_rows],
        'instance_token': [r['instance_token'] for r in ann_rows],
        'category': [inst_cat[r['instance_token']] for r in ann_rows],
        'x_world': [r['translation'][0] for r in ann_rows],
        'y_world': [r['translation'][1] for r in ann_rows],
        'z_world': [r['translation'][2] for r in ann_rows],
        'width': [r['size'][0] for r in ann_rows],
        'visibility': [float(r['visibility_token']) if r['visibility_token'] in ('1', '2', '3', '4') else np.nan for r in ann_rows],
        'num_lidar_pts': [r['num_lidar_pts'] for r in ann_rows],
        'num_radar_pts': [r['num_radar_pts'] for r in ann_rows],
        'attrs': [';'.join(attr_name[x] for x in r['attribute_tokens']) for r in ann_rows]})
    del ann_rows
    gc.collect()
    obj = obj.merge(ego, on='sample_token', how='left', validate='many_to_one')
    obj = obj.sort_values(['instance_token', 'timestamp']).reset_index(drop=True)
    obj['object_vx'] = np.nan
    obj['object_vy'] = np.nan
    for _, g in obj.groupby('instance_token', sort=False):
        obj.loc[g.index, ['object_vx', 'object_vy']] = differences(g.timestamp.to_numpy(), g[['x_world', 'y_world']].to_numpy(), 1.5)
    obj['object_speed'] = np.hypot(obj.object_vx, obj.object_vy)
    pxy, exy = obj[['x_world', 'y_world']].to_numpy(), obj[['ego_x', 'ego_y']].to_numpy()
    obj['distance_xy'] = np.linalg.norm(pxy - exy, axis=1)
    rel, radial, lateral = relative_components(pxy, exy, obj[['object_vx', 'object_vy']].to_numpy(), obj[['ego_vx', 'ego_vy']].to_numpy())
    obj['relative_speed'] = np.linalg.norm(rel, axis=1)
    obj['relative_radial_velocity'] = radial
    obj['relative_lateral_velocity'] = np.abs(lateral)
    obj['abs_ego_yaw_rate'] = obj.ego_yaw_rate.abs()
    w_ok = np.isfinite(obj.width) & (obj.width > 0)
    obj['r500'] = np.where(w_ok, obj.relative_lateral_velocity * 0.5 / obj.width, np.nan)
    for k in range(1, 11):
        obj[f'r_k{k}'] = np.where(w_ok, obj.relative_lateral_velocity * DELAY_MS[f'k{k}'] / 1000 / obj.width, np.nan)
    loc = {x['token']: x['location'] for x in log_tab}
    sinfo = pd.DataFrame([{'scene_token': s['token'], 'scene_name': s['name'], 'location': loc[s['log_token']],
                           'night': int('night' in s['description'].lower()),
                           'intersection': int('intersection' in s['description'].lower()),
                           'rain': int('rain' in s['description'].lower())} for s in scenes])
    obj = obj.merge(sinfo, on='scene_token', how='left')
    log(f'annotations: {len(obj):,}; object speed missing: {obj.object_speed.isna().mean():.4f}')

    # ---- join features back to each GT box (sample token + position) ----
    key = lambda df, x, y, z: list(zip(df.sample_token, df[x].round(4), df[y].round(4), df[z].round(4)))
    obj['key'] = key(obj, 'x_world', 'y_world', 'z_world')
    gt['key'] = key(gt, 'x', 'y', 'z')
    obj_u = obj.drop_duplicates('key')
    feat = ['annotation_token', 'instance_token', 'scene_token', 'scene_name', 'category', 'distance_xy', 'object_speed',
            'ego_speed', 'relative_speed', 'relative_radial_velocity', 'relative_lateral_velocity', 'abs_ego_yaw_rate',
            'visibility', 'num_lidar_pts', 'num_radar_pts', 'width', 'attrs', 'night', 'intersection', 'rain', 'location',
            'r500'] + [f'r_k{k}' for k in range(1, 11)]
    gtf = gt.merge(obj_u[['key'] + feat], on='key', how='left', validate='many_to_one').drop(columns='key')
    gtf = gtf.sort_values('box_id').reset_index(drop=True)
    save_csv(gtf, os.path.join(a.out, 'gt_features.csv.gz'), index=False)

    # ---- reconciliation ----
    log('\nReconciliation')
    log(f'  GT boxes without a matching annotation: {int(gtf.annotation_token.isna().sum())} (expected 0)')
    log(f'  class consistent with official mapping: {float((gtf.category.map(DET_NAME) == gtf.cls).mean()):.6f} (expected 1)')
    per_cls = gtf.groupby('cls').size()
    for c, n in EXPECTED['boxes_per_class'].items():
        got = int(per_cls.get(c, 0))
        log(f'  {c:21s} {got:6d}  expected {n:6d}  {"OK" if got == n else "MISMATCH"}')
    tok_scene = pd.DataFrame({'sample_token': tokens}).merge(ego[['sample_token', 'scene_token']], on='sample_token').merge(sinfo, on='scene_token')
    log(f'  night samples: {int(tok_scene.night.sum())} (expected {EXPECTED["night_samples"]}); '
        f'intersection samples: {int(tok_scene.intersection.sum())} (expected {EXPECTED["intersection_samples"]})')
    log(f'saved {os.path.join(a.out, "gt_features.csv.gz")}  ({time.time() - t_start:.0f}s)')


if __name__ == '__main__':
    main(get_args('Step 1: per-GT-box features'))
