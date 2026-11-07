# Condition-variable table for the T1B conditional backfire analysis.
# One JSONL row per (val sample_token, sample_annotation) with observable
# conditions: distance, speed (instance-chain), visibility, num_lidar_pts,
# attributes, size — plus, per camera-history depth k in {2,5,8,10,12}:
# historical timestamp, ego shift between capture times, constant-velocity
# object displacement, and object-relative-to-ego displacement.
# Metadata-only (no sensor files needed). Output: /root/autodl-tmp/condition_table.jsonl
import json, os, math, time

META = '/root/autodl-tmp/nuscenes_meta/v1.0-trainval'
GT_COMMON = '/root/autodl-tmp/gt_common.json'
SUB1 = '/root/autodl-tmp/sub_table_t1b.json'
SUB2 = '/root/autodl-tmp/sub_table_t1b2.json'   # k10/k12 (if present)
OUT = '/root/autodl-tmp/condition_table.jsonl'
KS = (2, 5, 8, 10, 12)
CAM = 'CAM_FRONT'

t0 = time.time()
def log(m):
    print('[%ds] %s' % (int(time.time() - t0), m), flush=True)

gt_tokens = set(json.load(open(GT_COMMON)).keys())
log('gt_common tokens: %d' % len(gt_tokens))

log('loading sample_annotation.json ...')
anns_all = json.load(open(os.path.join(META, 'sample_annotation.json')))
log('annotations: %d' % len(anns_all))

samples = {s['token']: s for s in json.load(open(os.path.join(META, 'sample.json')))}
log('samples: %d' % len(samples))

# ego pose per token
ep = {e['token']: e for e in json.load(open(os.path.join(META, 'ego_pose.json')))}
sensor_ch = {s['token']: s['channel'] for s in json.load(open(os.path.join(META, 'sensor.json')))}
cs = {c['token']: sensor_ch[c['sensor_token']] for c in json.load(open(os.path.join(META, 'calibrated_sensor.json')))}
# lidar keyframe sample_data per sample
sd = json.load(open(os.path.join(META, 'sample_data.json')))
sample_lidar_sd = {}
scene_timeline = {}   # scene_token -> sorted list of (timestamp, ego translation xy)
for r in sd:
    st = r['sample_token']
    e = ep[r['ego_pose_token']]
    tl = scene_timeline.setdefault(samples[st]['scene_token'], [])
    tl.append((r['timestamp'], (e['translation'][0], e['translation'][1])))
    if cs[r['calibrated_sensor_token']] == 'LIDAR_TOP' and st in gt_tokens:
        sample_lidar_sd[st] = r
for tl in scene_timeline.values():
    tl.sort()
log('lidar keyframes in gt_common: %d, scenes: %d' % (len(sample_lidar_sd), len(scene_timeline)))

attr = {a['token']: a['name'] for a in json.load(open(os.path.join(META, 'attribute.json')))}
cat = {c['token']: c['name'] for c in json.load(open(os.path.join(META, 'category.json')))}
inst_cat = {i['token']: cat[i['category_token']] for i in json.load(open(os.path.join(META, 'instance.json')))}

# annotations of val samples, indexed by sample_token and by instance
anns_by_sample = {}
anns_by_inst = {}
for a in anns_all:
    if a['sample_token'] in gt_tokens:
        anns_by_sample.setdefault(a['sample_token'], []).append(a)
        anns_by_inst.setdefault(a['instance_token'], []).append(a)
del anns_all
log('val annotations kept: %d, instances: %d' % (sum(len(v) for v in anns_by_sample.values()), len(anns_by_inst)))

# per-instance adjacent-frame speed (m/s, global xy)
inst_speed = {}
for inst, lst in anns_by_inst.items():
    lst.sort(key=lambda a: samples[a['sample_token']]['timestamp'])
    for i, a in enumerate(lst):
        best = None
        for j in (i - 1, i + 1):
            if 0 <= j < len(lst):
                b = lst[j]
                dt = (samples[b['sample_token']]['timestamp'] - samples[a['sample_token']]['timestamp']) / 1e6
                if dt == 0:
                    continue
                dx = (b['translation'][0] - a['translation'][0]) / dt
                dy = (b['translation'][1] - a['translation'][1]) / dt
                if best is None or abs(dt) < abs(best[2]):
                    best = (dx, dy, dt)
        inst_speed[a['token']] = (best[0], best[1]) if best else (0.0, 0.0)
del anns_by_inst
log('speeds computed')

# historical timestamps per (token, k) from T1B sub tables
sub1 = json.load(open(SUB1))
sub2 = json.load(open(SUB2)) if os.path.exists(SUB2) else {}
def hist_rec(tok, k):
    e = sub1.get(tok) or sub2.get(tok)
    if not e:
        return None
    return (e.get(CAM) or {}).get('k%d' % k)

def nearest_ego(scene, ts):
    tl = scene_timeline[scene]
    lo, hi = 0, len(tl) - 1
    while lo < hi:
        mid = (lo + hi) // 2
        if tl[mid][0] < ts:
            lo = mid + 1
        else:
            hi = mid
    cands = [tl[lo]] + ([tl[lo - 1]] if lo > 0 else [])
    return min(cands, key=lambda x: abs(x[0] - ts))[1]

n_rows = 0
with open(OUT, 'w') as f:
    for tok in sorted(gt_tokens):
        if tok not in sample_lidar_sd:
            continue
        s = samples[tok]
        scene = s['scene_token']
        ts_cur = s['timestamp']
        e_cur = ep[sample_lidar_sd[tok]['ego_pose_token']]['translation']
        ecur_xy = (e_cur[0], e_cur[1])
        # cache per-k history geometry for this token
        kg = {}
        for k in KS:
            rec = hist_rec(tok, k)
            if rec is None or not rec.get('ts'):
                continue
            ts_h = rec['ts']
            dt = (ts_h - ts_cur) / 1e6  # seconds (negative)
            e_hist = nearest_ego(scene, ts_h)
            kg[k] = (ts_h, dt, e_hist, math.hypot(e_hist[0] - ecur_xy[0], e_hist[1] - ecur_xy[1]))
        for a in anns_by_sample.get(tok, []):
            obj = (a['translation'][0], a['translation'][1])
            dist = math.hypot(obj[0] - ecur_xy[0], obj[1] - ecur_xy[1])
            vx, vy = inst_speed.get(a['token'], (0.0, 0.0))
            row = {
                'sample': tok, 'ann': a['token'], 'category': inst_cat[a['instance_token']],
                'dist_m': round(dist, 2),
                'speed_mps': round(math.hypot(vx, vy), 3),
                'vx': round(vx, 3), 'vy': round(vy, 3),
                'visibility': int(a['visibility_token']),
                'num_lidar_pts': a['num_lidar_pts'],
                'attrs': [attr[t] for t in a['attribute_tokens']],
                'size': [round(x, 2) for x in a['size']],
            }
            for k in KS:
                if k not in kg:
                    continue
                ts_h, dt, e_hist, ego_shift = kg[k]
                obj_h = (obj[0] + vx * dt, obj[1] + vy * dt)
                rel = math.hypot((obj[0] - ecur_xy[0]) - (obj_h[0] - e_hist[0]),
                                 (obj[1] - ecur_xy[1]) - (obj_h[1] - e_hist[1]))
                row['k%d' % k] = {'dt_ms': round(dt * 1e3, 1), 'ego_shift_m': round(ego_shift, 2),
                                  'obj_shift_m': round(math.hypot(vx * dt, vy * dt), 2),
                                  'rel_disp_m': round(rel, 2)}
            f.write(json.dumps(row) + '\n')
            n_rows += 1
log('rows written: %d -> %s' % (n_rows, OUT))
print('CONDITION_TABLE_DONE')
