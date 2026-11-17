# pilot 验证:1) chunk clean 重跑 vs 全量 clean 基线 逐 token 对比(恒等性)
#             2) k1 pkl 替换正确性抽查(img_path 已换 + 字节级 md5 一致)
import json, pickle, hashlib, random, sys, os

PRED_T = '/root/autodl-tmp/preds_temporal'
CLEAN_FULL = '/root/autodl-tmp/preds/fusion_cumulative.json'
CHUNK = '/root/autodl-tmp/chunk'
HIST = '/root/autodl-tmp/camhist'

# 1) 恒等性
pilot = json.load(open(f'{PRED_T}/clean/blob01/pred_instances_3d/results_nusc.json'))
full = json.load(open(CLEAN_FULL))['results']
common = set(pilot['results']) & set(full)
print('IDENTITY tokens pilot:', len(pilot['results']), 'common:', len(common))
maxdiff = 0.0
ndiff_box = 0
for t in common:
    a, b = pilot['results'][t], full[t]
    if len(a) != len(b):
        ndiff_box += 1
        continue
    for ba, bb in zip(a, b):
        for f in range(3):
            d = abs(ba['translation'][f] - bb['translation'][f])
            if d > maxdiff:
                maxdiff = d
        for f in range(2):
            d = abs(ba['velocity'][f] - bb['velocity'][f])
            if d > maxdiff:
                maxdiff = d
print(f'IDENTITY max_abs_diff={maxdiff:.6f} box_count_mismatch={ndiff_box}')

# 2) k1 替换抽查
base = pickle.load(open('/root/autodl-tmp/pilot_pkls/nuscenes_infos_val.pkl', 'rb'))
k1 = pickle.load(open('/root/autodl-tmp/pilot_pkls/nuscenes_infos_val_k1.pkl', 'rb'))
base_by_tok = {d['token']: d for d in base['data_list']}
CAMS = list(k1['data_list'][0]['images'].keys())
random.seed(0)
sample = random.sample(k1['data_list'], min(5, len(k1['data_list'])))
n_changed, n_hash_ok = 0, 0
for d in sample:
    b = base_by_tok[d['token']]
    for cam in CAMS:
        o, n = b['images'][cam]['img_path'], d['images'][cam]['img_path']
        if o != n:
            n_changed += 1
        # chunk 里的文件 = camhist 里的文件(md5 校验)
        cf = os.path.join(CHUNK, 'samples', cam, n)
        hf = None
        # 从 sub_table 找来源路径
        st = json.load(open('/root/autodl-tmp/sub_table.json'))
        rec = st[d['token']][cam]['k1']
        hf = os.path.join(HIST, rec['file'])
        if os.path.exists(cf) and os.path.exists(hf):
            m = lambda p: hashlib.md5(open(p, 'rb').read()).hexdigest()
            if m(cf) == m(hf):
                n_hash_ok += 1
print(f'SUBST checked_samples={len(sample)} changed_paths={n_changed}/{len(sample)*len(CAMS)} hash_ok={n_hash_ok}')
# 时间方向
st = json.load(open('/root/autodl-tmp/sub_table.json'))
bad = 0
for d in k1['data_list'][:200]:
    for cam in CAMS:
        e = st[d['token']][cam]
        if e['k1'] and not (e['k1']['ts'] < e['cur_ts']):
            bad += 1
print('TIME_DIR violations:', bad)
