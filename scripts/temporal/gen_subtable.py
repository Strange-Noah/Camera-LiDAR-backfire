# 生成 temporal T1 替换表:逐样本 x 6 相机沿 sample_data.prev 走 k=1,2
# 输出: sub_table.json + hist_patterns.txt(tar 抽取用)
import pickle, json, os
from nuscenes import NuScenes

EVALROOT = '/root/autodl-tmp/evalroot'
PKL = '/root/mmdetection3d/data/nuscenes_trainval/nuscenes_infos_val.pkl'
OUT = '/root/autodl-tmp/sub_table.json'
PAT = '/root/autodl-tmp/hist_patterns.txt'
CAMS = ['CAM_FRONT', 'CAM_FRONT_RIGHT', 'CAM_FRONT_LEFT', 'CAM_BACK', 'CAM_BACK_LEFT', 'CAM_BACK_RIGHT']
KS = [1, 2]

info = pickle.load(open(PKL, 'rb'))
tokens = [d['token'] for d in info['data_list']]
print('val samples:', len(tokens))

nusc = NuScenes(version='v1.0-trainval', dataroot=EVALROOT, verbose=False)
sd = nusc.sample_data
samp = nusc.sample
by_token = {s['token']: s for s in samp}
sd_by_token = {r['token']: r for r in sd}

def follow_prev(cur_tok, k, scene_token):
    rec = sd_by_token[cur_tok]
    for _ in range(k):
        p = rec['prev']
        if not p:
            return None
        rec = sd_by_token[p]
        if rec['channel'] not in CAMS:
            return None
        # scene membership: prev record -> enclosing sample -> scene
        st = by_token.get(rec['sample_token'])
        if st is None or st['scene_token'] != scene_token:
            return None
    return rec

table = {}
patterns = set()
n_elig = {k: 0 for k in KS}
missing_cur = 0
for tok in tokens:
    s = by_token.get(tok)
    if s is None:
        missing_cur += 1
        continue
    scene = s['scene_token']
    entry = {}
    all_elig = {k: True for k in KS}
    for cam in CAMS:
        cur_sd = sd_by_token[s['data'][cam]]
        e = {
            'cur_token': cur_sd['token'],
            'cur_ts': cur_sd['timestamp'],
            'cur_file': cur_sd['filename'],
        }
        for k in KS:
            r = follow_prev(cur_sd['token'], k, scene)
            if r is None:
                e[f'k{k}'] = None
                all_elig[k] = False
            else:
                e[f'k{k}'] = {'token': r['token'], 'ts': r['timestamp'], 'file': r['filename']}
                patterns.add(r['filename'])
        entry[cam] = e
    for k in KS:
        if all_elig[k]:
            n_elig[k] += 1
    table[tok] = entry

json.dump(table, open(OUT, 'w'))
open(PAT, 'w').write('\n'.join(sorted(patterns)))
print('missing_cur:', missing_cur)
for k in KS:
    print(f'eligible_k{k}:', n_elig[k], '/', len(table))
print('unique hist files:', len(patterns))
# 延迟分布
import statistics
for k in KS:
    ds = []
    for tok, entry in table.items():
        for cam in CAMS:
            a, b = entry[cam]['k0_ts'] if False else (None, None)
        break
print('done')
