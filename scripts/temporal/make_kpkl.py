# 由 chunk base pkl + sub_table 生成 k=1/2 的 T1 错位 pkl,并把历史图硬链接进 chunk
# 用法: make_kpkl.py <CHUNK> <HIST> <SUBTABLE_JSON> <K>
# 注意: sample_data.filename 自带目录前缀(如 sweeps/CAM_BACK/xx.jpg),按原名从 HIST 取,
#       硬链接到 CHUNK/samples/<CAM>/<basename>(加载器按 data_prefix 在此查找),img_path 记 basename。
import pickle, json, os, sys

CHUNK, HIST, SUB, K = sys.argv[1], sys.argv[2], sys.argv[3], int(sys.argv[4])
base = pickle.load(open(f'{CHUNK}/nuscenes_infos_val.pkl', 'rb'))
table = json.load(open(SUB))
CAMS = ['CAM_FRONT', 'CAM_FRONT_RIGHT', 'CAM_FRONT_LEFT', 'CAM_BACK', 'CAM_BACK_LEFT', 'CAM_BACK_RIGHT']

kept, dropped_no_prev, dropped_no_file = [], 0, 0
for d in base['data_list']:
    tok = d['token']
    e = table.get(tok)
    if e is None:
        dropped_no_prev += 1
        continue
    newd = dict(d)
    imgs = {}
    ok, miss_prev = True, False
    for cam in CAMS:
        im = dict(d['images'][cam])
        rec = e[cam].get(f'k{K}')
        if rec is None:
            ok, miss_prev = False, True
            break
        src = os.path.join(HIST, rec['file'])
        if not os.path.exists(src):
            ok = False
            break
        base_name = os.path.basename(rec['file'])
        dst = os.path.join(CHUNK, 'samples', cam, base_name)
        if not os.path.exists(dst):
            os.link(src, dst)
        im['img_path'] = base_name
        imgs[cam] = im
    if not ok:
        if miss_prev:
            dropped_no_prev += 1
        else:
            dropped_no_file += 1
        continue
    newd['images'] = imgs
    kept.append(newd)

for i, d in enumerate(kept):
    d['sample_idx'] = i
out = dict(metainfo=base['metainfo'], data_list=kept)
pickle.dump(out, open(f'{CHUNK}/nuscenes_infos_val_k{K}.pkl', 'wb'))
print(f'KPkl K={K} KEPT {len(kept)} DROP_NOPREV {dropped_no_prev} DROP_NOFILE {dropped_no_file}')
