# E6 (JOINT camera+LiDAR temporal misalignment) cycling driver.
# Grid: camera history depth kc in {2,5,8} (T1 stale-camera semantics: historical
# image, calibration/ego pose kept CURRENT) x lidar stale depth kl in {1,3,6}
# (E2 naive stale-LiDAR semantics: main cloud := k-th past sweep, un-warped,
# sweeps := sweeps[kl:]).
# Camera history files are resolved via sub_table10.json (T1 prev-chain map);
# missing history images are extracted from the nuScenes blobs on demand, then
# hard-linked into chunk/samples/<CAM>/ (loader convention, same as T1).
# Lidar files are dual-placed (samples/ + sweeps/) exactly as in cycle_e2/e5.
# By default this script WAITS until cycle_e5.py has finished (single GPU + shared
# chunk cleanup make concurrent runs unsafe). Set E6_NOWAIT=1 to override (pilot).
# Set E6_CELLS='2:1,5:3,8:6' to restrict the grid (pilot). Set E6_CLEANUP=0 to
# skip cleanup (pilot alongside E5).
# 2026-11-03 patch: dual_place/cam_place tolerate OSError (ENOSPC) so a full disk
# logs LINK_FAIL instead of killing the driver (missing files are caught by the
# completeness check and re-scanned next round).
import pickle, json, os, subprocess, sys, time
import numpy as np

CHUNK = '/root/autodl-tmp/chunk'
PRED = '/root/autodl-tmp/preds_temporal'
MM = '/root/mmdetection3d'
CFG = 'projects/BEVFusion/configs/bevfusion_lidar-cam_voxel0075_second_secfpn_8xb4-cyclic-20e_nus-3d.py'
CKPT = 'checkpoints/bevfusion_lidar-cam_voxel0075_second_secfpn_8xb4-cyclic-20e_nus-3d-5239b1af.pth'
BLOBS = [f'/root/autodl-pub/nuScenes/Fulldatasetv1.0/Trainval/v1.0-trainval{i:02d}_blobs.tgz' for i in range(1, 11)]
CAMS = ['CAM_FRONT', 'CAM_FRONT_RIGHT', 'CAM_FRONT_LEFT', 'CAM_BACK', 'CAM_BACK_LEFT', 'CAM_BACK_RIGHT']
KCS = (2, 5, 8)
KLS = (1, 3, 6)
if os.environ.get('E6_CELLS'):
    CELLS = [tuple(map(int, c.split(':'))) for c in os.environ['E6_CELLS'].split(',')]
else:
    CELLS = [(kc, kl) for kc in KCS for kl in KLS]
N_TARGET = int(os.environ.get('E6_N_TARGET', '200'))
BASE_PKL = f'{CHUNK}/nuscenes_infos_val.pkl'
SUB_PATH = '/root/autodl-tmp/sub_table10.json'
BLOBMAP_PATH = '/root/autodl-tmp/cycle_blobmap_e6.json'
ATT_PATH = '/root/autodl-tmp/cycle_attempts_e6.json'
SKIP_PATH = '/root/autodl-tmp/cycle_skip_e6.json'
LINKED_PATH = '/root/autodl-tmp/cycle_linked_e6.json'
LOG = open('/root/autodl-tmp/cycle_e6.log', 'a')

def log(m):
    LOG.write(f'[{time.strftime("%H:%M:%S")}] {m}\n')
    LOG.flush()

# --- wait for E5 to finish (unless NOWAIT) ---
if os.environ.get('E6_NOWAIT') != '1':
    log('WAIT_E5 start')
    while True:
        try:
            tail = open('/root/autodl-tmp/cycle_e5.log').read()[-300:]
        except FileNotFoundError:
            tail = ''
        alive = subprocess.run(['pgrep', '-f', 'cycle_e5.py'], capture_output=True, text=True).stdout.strip()
        if 'ALL_DONE' in tail and not alive:
            break
        log(f'WAIT_E5 e5_alive={bool(alive)}')
        time.sleep(300)
    log('WAIT_E5 done, E6 starting')

BASE = pickle.load(open(BASE_PKL, 'rb'))
DATA = BASE['data_list']
SUB = json.load(open(SUB_PATH))
TOK = [d['token'] for d in DATA]
BYTOK = {d['token']: d for d in DATA}
SWEEP_N = {d['token']: len(d.get('lidar_sweeps', [])) for d in DATA}

def done_set(cell):
    p = f'{PRED}/_done_e6_c{cell[0]}_l{cell[1]}.json'
    return set(json.load(open(p))) if os.path.exists(p) else set()

DONE = {c: done_set(c) for c in CELLS}
BLOBMAP = json.load(open(BLOBMAP_PATH)) if os.path.exists(BLOBMAP_PATH) else {}
TOK2BLOB = json.load(open('/root/autodl-tmp/tok2blob.json')) if os.path.exists('/root/autodl-tmp/tok2blob.json') else {}
ATT = json.load(open(ATT_PATH)) if os.path.exists(ATT_PATH) else {}
SKIP = set(json.load(open(SKIP_PATH))) if os.path.exists(SKIP_PATH) else set()
LINKED = set(json.load(open(LINKED_PATH))) if os.path.exists(LINKED_PATH) else set()

def save_state():
    json.dump(BLOBMAP, open(BLOBMAP_PATH, 'w'))
    json.dump(ATT, open(ATT_PATH, 'w'))
    json.dump(sorted(SKIP), open(SKIP_PATH, 'w'))
    json.dump(sorted(LINKED), open(LINKED_PATH, 'w'))

def cam_ok(tok, kc):
    e = SUB.get(tok)
    if e is None:
        return False
    return all((e[c].get(f'k{kc}') or {}).get('file') for c in CAMS)

def k_eligible(tok):
    return [c for c in CELLS if cam_ok(tok, c[0]) and SWEEP_N.get(tok, 0) >= c[1]]

def k_pending(tok):
    return [c for c in k_eligible(tok) if tok not in DONE[c]]

def scene_of(tok):
    return BYTOK[tok]['lidar_points']['lidar_path'].split('__')[0]

def all_tokens():
    base = sorted(set(TOK) - SKIP)
    return sorted(base, key=lambda t: (scene_of(t), BYTOK[t]['timestamp'], t))

def fully_done(tok):
    return not k_pending(tok)

def sweep_basenames(tok):
    d = BYTOK[tok]
    return [os.path.basename(sw['lidar_points']['lidar_path']) for sw in d.get('lidar_sweeps', [])]

def lidar_needed(tok, kl):
    return sweep_basenames(tok)[kl - 1:]

def cam_rel(tok, kc):
    # dataset-relative history image paths (e.g. sweeps/CAM_FRONT/xx.jpg)
    return [SUB[tok][c][f'k{kc}']['file'] for c in CAMS]

def dual_place(names):
    for n in names:
        p1 = os.path.join(CHUNK, 'samples/LIDAR_TOP', n)
        p2 = os.path.join(CHUNK, 'sweeps/LIDAR_TOP', n)
        try:
            if os.path.exists(p1) and not os.path.exists(p2):
                os.link(p1, p2)
            elif os.path.exists(p2) and not os.path.exists(p1):
                os.link(p2, p1)
        except OSError as e:
            log(f'LINK_FAIL {n}: {e}')

def cam_place(tok, kc):
    for rel in cam_rel(tok, kc):
        cam = rel.split('/')[1]
        bn = os.path.basename(rel)
        src = os.path.join(CHUNK, rel)
        dst = os.path.join(CHUNK, 'samples', cam, bn)
        if os.path.exists(src) and not os.path.exists(dst):
            try:
                os.link(src, dst)
                LINKED.add(dst)
            except OSError as e:
                log(f'LINK_FAIL {dst}: {e}')

def lidar_files(tok, kl):
    return [(sub, n) for n in lidar_needed(tok, kl) for sub in ('samples/LIDAR_TOP', 'sweeps/LIDAR_TOP')]

def complete(tok, cell):
    kc, kl = cell
    if not all(os.path.exists(os.path.join(CHUNK, sub, n)) for sub, n in lidar_files(tok, kl)):
        return False
    return all(os.path.exists(os.path.join(CHUNK, 'samples', rel.split('/')[1], os.path.basename(rel)))
               for rel in cam_rel(tok, kc))

def e6_sample(tok, cell):
    """joint-misaligned copy: historical images @kc + naive stale main lidar @kl"""
    kc, kl = cell
    d = BYTOK[tok]
    sw = d.get('lidar_sweeps', [])
    nd = dict(d)
    nd['lidar_points'] = dict(d['lidar_points'])
    nd['lidar_points']['lidar_path'] = os.path.basename(sw[kl - 1]['lidar_points']['lidar_path'])
    nd['lidar_sweeps'] = [
        dict(s, lidar_points=dict(s['lidar_points'],
             lidar_path='./data/nuscenes_trainval/sweeps/LIDAR_TOP/' + os.path.basename(s['lidar_points']['lidar_path'])))
        for s in sw[kl:]
    ]
    imgs = {}
    for cam in CAMS:
        im = dict(d['images'][cam])
        im['img_path'] = os.path.basename(SUB[tok][cam][f'k{kc}']['file'])
        imgs[cam] = im
    nd['images'] = imgs
    return nd

def write_batch(cell, tokens, tag):
    sub = [e6_sample(t, cell) for t in tokens]
    for i, d in enumerate(sub):
        d['sample_idx'] = i
    out = dict(metainfo=BASE['metainfo'], data_list=sub)
    p = f'{CHUNK}/nuscenes_infos_val_e6_c{cell[0]}_l{cell[1]}_{tag}.pkl'
    pickle.dump(out, open(p, 'wb'))
    return p

def infer_cell(cell, pkl, tag, tokens):
    kc, kl = cell
    outdir = f'{PRED}/e6_c{kc}_l{kl}/{tag}'
    os.makedirs(outdir, exist_ok=True)
    cmd = ['tools/test.py', CFG, CKPT, '--cfg-options',
           f'test_dataloader.dataset.data_root={CHUNK}/',
           f'test_dataloader.dataset.ann_file={pkl}',
           f'test_evaluator.ann_file={pkl}',
           'test_evaluator.format_only=True',
           f'test_evaluator.jsonfile_prefix={outdir}']
    env = dict(os.environ)
    env['PATH'] = '/root/miniconda3/envs/bevfusion/bin:' + env['PATH']
    env.pop('CUDA_VISIBLE_DEVICES', None)
    r = subprocess.run(['/root/miniconda3/envs/bevfusion/bin/python'] + cmd, cwd=MM, env=env,
                       stdout=open(f'/root/autodl-tmp/cycle_infer_e6_c{kc}_l{kl}_{tag}.out', 'w'),
                       stderr=subprocess.STDOUT)
    if r.returncode != 0:
        log(f'INFER_FAIL e6_c{kc}_l{kl} {tag} rc={r.returncode}')
        return False
    r2 = subprocess.run(['/root/miniconda3/envs/bevfusion/bin/python', '/root/accum_e6.py', PRED, tag, str(kc), str(kl)],
                        capture_output=True, text=True)
    log(f'ACCUM e6_c{kc}_l{kl} {tag}: {r2.stdout.strip()}')
    return r2.returncode == 0

def cleanup():
    if os.environ.get('E6_CLEANUP') == '0':
        return
    pend = set()
    pend_cam = set()
    for t in all_tokens():
        if not fully_done(t):
            for cell in k_pending(t):
                kc, kl = cell
                pend.update(os.path.join(sub, n) for sub, n in lidar_files(t, kl))
                for rel in cam_rel(t, kc):
                    pend_cam.add(os.path.join(CHUNK, 'samples', rel.split('/')[1], os.path.basename(rel)))
    removed = 0
    for sub in ('samples/LIDAR_TOP', 'sweeps/LIDAR_TOP'):
        d = os.path.join(CHUNK, sub)
        for n in os.listdir(d):
            rel = os.path.join(sub, n)
            if rel not in pend:
                os.remove(os.path.join(d, n))
                removed += 1
    # camera history links: only delete files E6 itself created, never dataset files
    for cam in CAMS:
        d = os.path.join(CHUNK, 'samples', cam)
        for n in os.listdir(d):
            p = os.path.join(d, n)
            if p in LINKED and p not in pend_cam:
                os.remove(p)
                LINKED.discard(p)
                removed += 1
    log(f'CLEANUP removed {removed}, pending-lidar {len(pend)}, pending-cam {len(pend_cam)}, linked-left {len(LINKED)}')

round_no = 0
while True:
    undone = [t for t in all_tokens() if not fully_done(t)]
    if not undone:
        log('ALL_DONE')
        break
    maxcell = max(CELLS, key=lambda c: sum(1 for t in undone if c in k_pending(t)))
    pool = [t for t in undone if maxcell in k_pending(t)] or undone
    targets = pool[:N_TARGET]
    log(f'ROUND {round_no} pending={len(undone)} pool={len(pool)} targets={len(targets)} maxcell=c{maxcell[0]}_l{maxcell[1]}')
    for t in targets:
        for cell in k_pending(t):
            dual_place(lidar_needed(t, cell[1]))
            cam_place(t, cell[0])
    ready = set(t for t in targets if all(complete(t, cell) for cell in k_pending(t)))
    for t in ready:
        if t not in BLOBMAP:
            BLOBMAP[t] = 0
        ATT.pop(t, None)
    save_state()
    nextb = {t: TOK2BLOB.get(t, min(BLOBMAP.get(t, 0), 9)) for t in targets if t not in ready}
    FALLBACK = set()
    scans = 0
    while True:
        cand = [t for t in nextb if t not in ready and nextb[t] < 10]
        if not cand:
            break
        if len(ready) >= max(1, int(0.98 * len(targets))):
            log(f'ROUND {round_no} early-stop scanning, ready {len(ready)}/{len(targets)}')
            break
        bs = sorted({min(nextb[t], 9) for t in cand if t not in FALLBACK})
        prog = [t for t in cand if t in FALLBACK]
        if prog:
            m = min(nextb[t] for t in prog)
            bs = sorted(set(bs) | set(range(m, min(m + 3, 10))))
        m = bs[0]
        miss = set()
        for t in cand:
            for cell in k_pending(t):
                kc, kl = cell
                for sub, n in lidar_files(t, kl):
                    rel = os.path.join(sub, n)
                    if not os.path.exists(os.path.join(CHUNK, rel)):
                        miss.add(rel)
                for rel2 in cam_rel(t, kc):
                    if not os.path.exists(os.path.join(CHUNK, rel2)):
                        miss.add(rel2)
        procs = []
        for b in bs:
            if miss:
                pat = f'/root/autodl-tmp/cycle_patterns_e6_b{b}.txt'
                open(pat, 'w').write('\n'.join(sorted(miss)))
                log(f'ROUND {round_no} scan blob{b+1} wave for {len(cand)} tokens ({len(miss)} files)')
                procs.append(subprocess.Popen(
                    ['tar', '-xzf', BLOBS[b], '-C', CHUNK, '-T', pat],
                    stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL))
                scans += 1
        for pr in procs:
            pr.wait()
        for t in cand:
            for cell in k_pending(t):
                dual_place(lidar_needed(t, cell[1]))
                cam_place(t, cell[0])
        for t in cand:
            if all(complete(t, cell) for cell in k_pending(t)):
                ready.add(t)
                BLOBMAP[t] = TOK2BLOB.get(t, m)
                ATT.pop(t, None)
            else:
                if t not in FALLBACK:
                    FALLBACK.add(t)
                    nextb[t] = 0
                else:
                    nextb[t] = max(bs) + 1
        save_state()
        log(f'ROUND {round_no} ready {len(ready)}/{len(targets)} (scans={scans})')
    notready = [t for t in targets if t not in ready]
    if notready:
        log(f'ROUND {round_no} NOT_READY {len(notready)}')
        for t in notready:
            ATT[t] = ATT.get(t, 0) + 1
            if ATT[t] >= 2:
                SKIP.add(t)
                log(f'ROUND {round_no} SKIP_TOKEN {t} (unready {ATT[t]} rounds)')
        save_state()
    for cell in CELLS:
        toks = [t for t in ready if cell in k_pending(t)]
        if not toks:
            continue
        kc, kl = cell
        log(f'ROUND {round_no} INFER_START e6_c{kc}_l{kl} n={len(toks)}')
        tag = f'cyc{round_no}'
        p = write_batch(cell, toks, tag)
        if not infer_cell(cell, p, tag, toks):
            log(f'ROUND {round_no} e6_c{kc}_l{kl} FAILED, stays pending')
        for c2 in CELLS:
            DONE[c2] = done_set(c2)
    cleanup()
    log(f'ROUND {round_no} END done ' + ' '.join(f'c{c[0]}_l{c[1]}={len(DONE[c])}' for c in CELLS))
    round_no += 1
log('EXIT')
