# 修复 pass3 chunk:按 pkl 记录路径逐一核对,缺失的从 gather/gather2 按文件名找回(硬链接)
import pickle, os, shutil, sys

CHUNK = '/root/autodl-tmp/chunk'
G = '/root/autodl-tmp/gather'
G2 = '/root/autodl-tmp/gather2'
PKL = f'{CHUNK}/nuscenes_infos_val.pkl'

info = pickle.load(open(PKL, 'rb'))

def norm(p):
    # 去掉 ./data/nuscenes_trainval/ 前缀,得到 chunk 内相对路径
    for pre in ('./data/nuscenes_trainval/', 'data/nuscenes_trainval/'):
        if p.startswith(pre):
            return p[len(pre):]
    return p

pools = []
for base in (G, G2):
    for sub in ('samples/LIDAR_TOP', 'sweeps/LIDAR_TOP'):
        d = os.path.join(base, sub)
        if os.path.isdir(d):
            pools.append(d)
    sd = os.path.join(base, 'samples')
    if os.path.isdir(sd):
        for cam in os.listdir(sd):
            cd = os.path.join(sd, cam)
            if os.path.isdir(cd):
                pools.append(cd)

missing = []
fixed = 0
for d in info['data_list']:
    refs = [norm(d['lidar_points']['lidar_path'])]
    for sw in d.get('lidar_sweeps', []):
        refs.append(norm(sw['lidar_points']['lidar_path']))
    for cam, im in d['images'].items():
        refs.append(norm(im['img_path']))
    for rel in refs:
        dst = os.path.join(CHUNK, rel)
        if os.path.exists(dst):
            continue
        name = os.path.basename(rel)
        src = next((os.path.join(pd, name) for pd in pools if os.path.exists(os.path.join(pd, name))), None)
        if src is None:
            missing.append(rel)
        else:
            os.makedirs(os.path.dirname(dst), exist_ok=True)
            try:
                os.link(src, dst)
            except OSError:
                shutil.copy2(src, dst)
            fixed += 1

print('FIXED', fixed)
print('STILL_MISSING', len(missing))
for m in missing[:20]:
    print('  MISS', m)
sys.exit(1 if missing else 0)
