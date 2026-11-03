# E5 (motion-COMPENSATED stale LiDAR + current camera) cycling driver.
# Same eligibility / blob-cycling / batching as cycle_e2 (E2 naive). The only
# difference: the stale main sweep's points are warped into the current lidar
# frame (comp_lib.compensate_sample) and saved as samples/LIDAR_TOP/COMP__<name>,
# which the pkl references as main cloud. Answers: how much of E2's instant
# collapse is naive-pose artifact vs genuine information loss?
import pickle, json, os, subprocess, sys, time
import numpy as np

CHUNK = '/root/autodl-tmp/chunk'
PRED = '/root/autodl-tmp/preds_temporal'
MM = '/root/mmdetection3d'
CFG = 'projects/BEVFusion/configs/bevfusion_lidar-cam_voxel0075_second_secfpn_8xb4-cyclic-20e_nus-3d.py'
CKPT = 'checkpoints/bevfusion_lidar-cam_voxel0075_second_secfpn_8xb4-cyclic-20e_nus-3d-5239b1af.pth'
BLOBS = [f'/root/autodl-pub/nuScenes/Fulldatasetv1.0/Trainval/v1.0-trainval{i:02d}_blobs.tgz' for i in range(1, 11)]
KS = list(range(1, 10))          # E5 k = 1..9
N_TARGET = 200
BASE_PKL = f'{CHUNK}/nuscenes_infos_val.pkl'
BLOBMAP_PATH = '/root/autodl-tmp/cycle_blobmap.json'
ATT_PATH = '/root/autodl-tmp/cycle_attempts_e5.json'
SKIP_PATH = '/root/autodl-tmp/cycle_skip_e5.json'
LOG = open('/root/autodl-tmp/cycle_e5.log', 'a')

sys.path.insert(0, '/root')
from comp_lib import comp_name, compensate_sample  # noqa: E402

def log(m):
    LOG.write(f'[{time.strftime("%H:%M:%S")}] {m}\n')
    LOG.flush()

BASE = pickle.load(open(BASE_PKL, 'rb'))
DATA = BASE['data_list']
IMG_BAD = set()
for d in DATA:
    for cam, ent in d['images'].items():
        if not os.path.exists(os.path.join(CHUNK, 'samples', cam, os.path.basename(ent['img_path']))):
            IMG_BAD.add(d['token'])
            break
log(f'IMG_BAD tokens (camera image absent from dataset): {len(IMG_BAD)}')
TOK = [d['token'] for d in DATA if d['token'] not in IMG_BAD]
BYTOK = {d['token']: d for d in DATA}
SWEEP_N = {d['token']: len(d.get('lidar_sweeps', [])) for d in DATA}

def done_set(k):
    p = f'{PRED}/_done_e5_k{k}.json'
    return set(json.load(open(p))) if os.path.exists(p) else set()

DONE = {k: done_set(k) for k in KS}
BLOBMAP = json.load(open(BLOBMAP_PATH)) if os.path.exists(BLOBMAP_PATH) else {}
TOK2BLOB = json.load(open('/root/autodl-tmp/tok2blob.json')) if os.path.exists('/root/autodl-tmp/tok2blob.json') else {}
ATT = json.load(open(ATT_PATH)) if os.path.exists(ATT_PATH) else {}
SKIP = set(json.load(open(SKIP_PATH))) if os.path.exists(SKIP_PATH) else set()

def save_state():
    json.dump(BLOBMAP, open(BLOBMAP_PATH, 'w'))
    json.dump(ATT, open(ATT_PATH, 'w'))
    json.dump(sorted(SKIP), open(SKIP_PATH, 'w'))

def k_eligible(tok):
    return [k for k in KS if SWEEP_N.get(tok, 0) >= k]

def k_pending(tok):
    return [k for k in k_eligible(tok) if tok not in DONE[k]]

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

def needed_basenames(tok, k):
    return sweep_basenames(tok)[k - 1:]

def dual_place(names):
    for n in names:
        p1 = os.path.join(CHUNK, 'samples/LIDAR_TOP', n)
        p2 = os.path.join(CHUNK, 'sweeps/LIDAR_TOP', n)
        if os.path.exists(p1) and not os.path.exists(p2):
            os.link(p1, p2)
        elif os.path.exists(p2) and not os.path.exists(p1):
            os.link(p2, p1)

def lidar_files(tok, k):
    return [(sub, n) for n in needed_basenames(tok, k)
            for sub in ('samples/LIDAR_TOP', 'sweeps/LIDAR_TOP')]

def complete(tok, k):
    return all(os.path.exists(os.path.join(CHUNK, sub, n)) for sub, n in lidar_files(tok, k))

def e5_sample(tok, k):
    """transformed copy: main := COMPENSATED k-th past sweep, sweeps := sweeps[k:]"""
    d = BYTOK[tok]
    sw = d.get('lidar_sweeps', [])
    nd = dict(d)
    nd['lidar_points'] = dict(d['lidar_points'])
    nd['lidar_points']['lidar_path'] = comp_name(os.path.basename(sw[k - 1]['lidar_points']['lidar_path']))
    nd['lidar_sweeps'] = [
        dict(s, lidar_points=dict(s['lidar_points'],
             lidar_path='./data/nuscenes_trainval/sweeps/LIDAR_TOP/' + os.path.basename(s['lidar_points']['lidar_path'])))
        for s in sw[k:]
    ]
    return nd

def write_batch(k, tokens, tag):
    sub = [e5_sample(t, k) for t in tokens]
    for i, d in enumerate(sub):
        d['sample_idx'] = i
    out = dict(metainfo=BASE['metainfo'], data_list=sub)
    p = f'{CHUNK}/nuscenes_infos_val_e5_k{k}_{tag}.pkl'
    pickle.dump(out, open(p, 'wb'))
    return p

def infer_k(k, pkl, tag, tokens):
    outdir = f'{PRED}/e5_k{k}/{tag}'
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
                       stdout=open(f'/root/autodl-tmp/cycle_infer_e5_k{k}_{tag}.out', 'w'),
                       stderr=subprocess.STDOUT)
    if r.returncode != 0:
        log(f'INFER_FAIL e5_k{k} {tag} rc={r.returncode}')
        return False
    r2 = subprocess.run(['/root/miniconda3/envs/bevfusion/bin/python', '/root/accum_e5.py', PRED, tag, str(k)],
                        capture_output=True, text=True)
    log(f'ACCUM e5_k{k} {tag}: {r2.stdout.strip()}')
    if r2.returncode == 0:
        for t in tokens:
            cn = comp_name(sweep_basenames(t)[k - 1])
            p = os.path.join(CHUNK, 'samples/LIDAR_TOP', cn)
            if os.path.exists(p):
                os.remove(p)
    return r2.returncode == 0

def cleanup():
    pend = set()
    pend_comp = set()
    for t in all_tokens():
        if not fully_done(t):
            for k in k_pending(t):
                pend.update(os.path.join(sub, n) for sub, n in lidar_files(t, k))
                pend_comp.add(comp_name(sweep_basenames(t)[k - 1]))
    removed = 0
    for sub in ('samples/LIDAR_TOP', 'sweeps/LIDAR_TOP'):
        d = os.path.join(CHUNK, sub)
        for n in os.listdir(d):
            rel = os.path.join(sub, n)
            if n.startswith('COMP__'):
                if n not in pend_comp:
                    os.remove(os.path.join(d, n))
                    removed += 1
            elif rel not in pend:
                os.remove(os.path.join(d, n))
                removed += 1
    log(f'CLEANUP removed {removed}, pending-files {len(pend)}, pending-comp {len(pend_comp)}')

round_no = 0
while True:
    undone = [t for t in all_tokens() if not fully_done(t)]
    if not undone:
        log('ALL_DONE')
        break
    pool = [t for t in undone if 9 in k_pending(t)] or undone
    targets = pool[:N_TARGET]
    log(f'ROUND {round_no} pending={len(undone)} pool9={len(pool)} targets={len(targets)}')
    for t in targets:
        for k in k_pending(t):
            dual_place(needed_basenames(t, k))
    ready = set(t for t in targets if all(complete(t, k) for k in k_pending(t)))
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
            for k in k_pending(t):
                for sub, n in lidar_files(t, k):
                    rel = os.path.join(sub, n)
                    if not os.path.exists(os.path.join(CHUNK, rel)):
                        miss.add(rel)
        procs = []
        for b in bs:
            if miss:
                pat = f'/root/autodl-tmp/cycle_patterns_e5_b{b}.txt'
                open(pat, 'w').write('\n'.join(sorted(miss)))
                log(f'ROUND {round_no} scan blob{b+1} wave for {len(cand)} tokens ({len(miss)} files)')
                procs.append(subprocess.Popen(
                    ['tar', '-xzf', BLOBS[b], '-C', CHUNK, '-T', pat],
                    stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL))
                scans += 1
        for pr in procs:
            pr.wait()
        for t in cand:
            for k in k_pending(t):
                dual_place(needed_basenames(t, k))
        for t in cand:
            if all(complete(t, k) for k in k_pending(t)):
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
    for k in KS:
        toks = [t for t in ready if k in k_pending(t)]
        if not toks:
            continue
        shifts = []
        for t in toks:
            try:
                _, ms = compensate_sample(BYTOK[t], k)
                shifts.append(ms)
            except Exception as e:
                log(f'COMP_FAIL e5_k{k} {t}: {e}')
        toks = [t for t in toks if os.path.exists(os.path.join(CHUNK, 'samples/LIDAR_TOP', comp_name(sweep_basenames(t)[k - 1])))]
        if not toks:
            continue
        if shifts:
            log(f'ROUND {round_no} COMP e5_k{k} n={len(toks)} mean_shift={np.mean(shifts):.2f}m max={np.max(shifts):.2f}m')
        tag = f'cyc{round_no}'
        p = write_batch(k, toks, tag)
        log(f'ROUND {round_no} INFER_START e5_k{k} n={len(toks)}')
        if not infer_k(k, p, tag, toks):
            log(f'ROUND {round_no} e5_k{k} FAILED, stays pending')
        for kk in KS:
            DONE[kk] = done_set(kk)
    cleanup()
    log(f'ROUND {round_no} END done ' + ' '.join(f'e5_k{k}={len(DONE[k])}' for k in KS))
    round_no += 1
log('EXIT')
