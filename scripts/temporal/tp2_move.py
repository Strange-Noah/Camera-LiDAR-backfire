# tp2 移动阶段:把 gather_t 中"已入 chunk pkl 的样本"的文件 mv 进 chunk(不复制,控磁盘)
import pickle, json, os, sys

GT = '/root/autodl-tmp/gather_t'
CHUNK = '/root/autodl-tmp/chunk'

info = pickle.load(open(f'{CHUNK}/nuscenes_infos_val.pkl', 'rb'))
moved = 0
for d in info['data_list']:
    # 当前图像 + lidar 样本
    for cam, im in d['images'].items():
        src = os.path.join(GT, 'samples', cam, im['img_path'])
        dst = os.path.join(CHUNK, 'samples', cam, im['img_path'])
        if os.path.exists(src) and not os.path.exists(dst):
            os.makedirs(os.path.dirname(dst), exist_ok=True)
            os.rename(src, dst)
            moved += 1
    lp = d['lidar_points']['lidar_path']
    src = os.path.join(GT, 'samples', 'LIDAR_TOP', lp)
    dst = os.path.join(CHUNK, 'samples', 'LIDAR_TOP', lp)
    if os.path.exists(src) and not os.path.exists(dst):
        os.makedirs(os.path.dirname(dst), exist_ok=True)
        os.rename(src, dst)
        moved += 1
    # sweeps(双位置)
    for sw in d.get('lidar_sweeps', []):
        n = os.path.basename(sw['lidar_points']['lidar_path'])
        dst = os.path.join(CHUNK, 'sweeps', 'LIDAR_TOP', n)
        if os.path.exists(dst):
            continue
        for src in (os.path.join(GT, 'sweeps', 'LIDAR_TOP', n),
                    os.path.join(GT, 'samples', 'LIDAR_TOP', n)):
            if os.path.exists(src):
                os.makedirs(os.path.dirname(dst), exist_ok=True)
                os.rename(src, dst)
                moved += 1
                break
print('TP2_MOVED', moved)
