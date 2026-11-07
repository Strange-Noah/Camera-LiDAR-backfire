# -*- coding: utf-8 -*-
"""Normalize chunk pkls (val/k1/k2) to the proven main-loop path scheme:
   - main lidar / images: './data/nuscenes_trainval/samples/...'
   - lidar sweeps:        './data/nuscenes_trainval/sweeps/...'
   - replaced (historical) camera images: absolute camhist path
   Misfiled sweeps (pkl says samples/... but file sits in sweeps/) are hardlinked
   back to the recorded location. Prints hardlink count and any MISSING paths.
"""
import pickle, os, shutil

CHUNK = '/root/autodl-tmp/chunk'
CAMHIST = '/root/autodl-tmp/camhist'
PREFIX = 'data/nuscenes_trainval/'

def ensure_layout():
    d = os.path.join(CHUNK, 'data', 'nuscenes_trainval')
    os.makedirs(d, exist_ok=True)
    for name in ('samples', 'sweeps', 'maps', 'v1.0-trainval'):
        link = os.path.join(d, name)
        if os.path.islink(link) or os.path.exists(link):
            continue
        os.symlink(os.path.join('..', '..', name), link)

def link_missing_lidar(dst_rel, stats):
    dst = os.path.join(CHUNK, dst_rel)
    if os.path.exists(dst):
        return True
    b = os.path.basename(dst_rel)
    alt = os.path.join(CHUNK, 'sweeps', 'LIDAR_TOP', b)
    if os.path.exists(alt):
        os.link(alt, dst)
        stats['hardlink'] += 1
        return True
    stats['missing'].append(dst_rel)
    return False

def fix_sample(s, stats):
    # main lidar
    lp = s['lidar_points']['lidar_path']
    rel = '%ssamples/LIDAR_TOP/%s' % (PREFIX, os.path.basename(lp))
    if link_missing_lidar(rel, stats):
        s['lidar_points']['lidar_path'] = './' + rel
    # camera images (current or replaced-historical)
    for cam, item in (s.get('images') or {}).items():
        p = item.get('img_path')
        if not p:
            continue
        b = os.path.basename(p)
        rel = '%ssamples/%s/%s' % (PREFIX, cam, b)
        if os.path.exists(os.path.join(CHUNK, rel)):
            item['img_path'] = './' + rel
            continue
        alt = os.path.join(CAMHIST, 'sweeps', cam, b)
        if os.path.exists(alt):
            item['img_path'] = alt  # absolute
            stats['camhist'] += 1
        else:
            stats['missing'].append('%s: %s' % (cam, b))
    # lidar sweeps
    for key in ('lidar_sweeps', 'sweeps'):
        for sw in (s.get(key) or []):
            lp = (sw.get('lidar_points') or {}).get('lidar_path')
            if not lp:
                continue
            b = os.path.basename(lp)
            if 'samples/LIDAR_TOP' in lp:
                rel = '%ssamples/LIDAR_TOP/%s' % (PREFIX, b)
            else:
                rel = '%ssweeps/LIDAR_TOP/%s' % (PREFIX, b)
            if link_missing_lidar(rel, stats):
                sw['lidar_points']['lidar_path'] = './' + rel
    # camera sweeps (if present under images)
    for cam, item in (s.get('images') or {}).items():
        for sw in (item.get('sweeps') or []):
            p = sw.get('img_path')
            if not p:
                continue
            b = os.path.basename(p)
            rel = '%ssweeps/%s/%s' % (PREFIX, cam, b)
            if os.path.exists(os.path.join(CHUNK, rel)):
                sw['img_path'] = './' + rel
            else:
                stats['missing'].append('camsweep %s: %s' % (cam, b))

def main():
    ensure_layout()
    for name in ('nuscenes_infos_val.pkl', 'nuscenes_infos_val_k1.pkl', 'nuscenes_infos_val_k2.pkl'):
        path = os.path.join(CHUNK, name)
        d = pickle.load(open(path, 'rb'))
        stats = {'hardlink': 0, 'camhist': 0, 'missing': []}
        for s in d['data_list']:
            fix_sample(s, stats)
        bak = path + '.bak'
        if not os.path.exists(bak):
            shutil.copy2(path, bak)
        with open(path, 'wb') as f:
            pickle.dump(d, f)
        print('%s: samples=%d hardlink=%d camhist_img=%d MISSING=%d'
              % (name, len(d['data_list']), stats['hardlink'], stats['camhist'], len(stats['missing'])))
        for m in stats['missing'][:20]:
            print('  MISSING:', m)

if __name__ == '__main__':
    main()
