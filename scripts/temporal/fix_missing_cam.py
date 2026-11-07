# One-off repair: restore current-frame camera images deleted by T1 cleanup.
# Sources: camhist backup first, then targeted tar extraction from nuScenes blobs.
import pickle, json, os, subprocess, time, collections

CH = '/root/autodl-tmp/chunk'
BLOBS = [f'/root/autodl-pub/nuScenes/Fulldatasetv1.0/Trainval/v1.0-trainval{i:02d}_blobs.tgz' for i in range(1, 11)]
BLOBMAP = json.load(open('/root/autodl-tmp/cycle_blobmap.json'))
LOG = open('/root/autodl-tmp/fix_missing_cam.log', 'a')
def log(m):
    LOG.write(f'[{time.strftime("%H:%M:%S")}] {m}\n'); LOG.flush()

d = pickle.load(open(f'{CH}/nuscenes_infos_val.pkl', 'rb'))

# frame basename -> sample token (for blob lookup)
frame2tok = {}
for s in d['data_list']:
    for cam, ent in s['images'].items():
        frame2tok[os.path.basename(ent['img_path'])] = s['token']

missing = []  # (cam, basename)
for s in d['data_list']:
    for cam, ent in s['images'].items():
        p = os.path.join(CH, 'samples', cam, os.path.basename(ent['img_path']))
        if not os.path.exists(p):
            src = os.path.join('/root/autodl-tmp/camhist/samples', cam, os.path.basename(ent['img_path']))
            if os.path.exists(src):
                os.link(src, p)
            else:
                missing.append((cam, os.path.basename(ent['img_path'])))
log(f'start: missing after camhist top-up = {len(missing)}')

# group missing frames by candidate blob (via BLOBMAP of any sample sharing the frame)
def cand_blobs(cam, name):
    tok = frame2tok.get(name)
    if tok is not None and str(tok) in BLOBMAP:
        return [BLOBMAP[str(tok)]]
    if tok is not None and tok in BLOBMAP:
        return [BLOBMAP[tok]]
    return list(range(10))  # unknown -> scan all

by_blob = collections.defaultdict(set)
unknown = []
for cam, name in missing:
    bs = cand_blobs(cam, name)
    if len(bs) == 10:
        unknown.append((cam, name))
    for b in bs:
        by_blob[b].add(f'samples/{cam}/{name}')
log(f'blob grouping: ' + ' '.join(f'b{b+1}={len(v)}' for b, v in sorted(by_blob.items())) + f' unknown={len(unknown)}')

remaining = set(missing)
for b in sorted(by_blob):
    pats = by_blob[b]
    pat_file = f'/root/autodl-tmp/fixcam_b{b}.txt'
    open(pat_file, 'w').write('\n'.join(sorted(pats)))
    log(f'scan blob{b+1} for {len(pats)} files')
    r = subprocess.run(['tar', '-xzf', BLOBS[b], '-C', CH, '-T', pat_file],
                       stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    found = sum(1 for cam, name in remaining if os.path.exists(os.path.join(CH, 'samples', cam, name)))
    log(f'blob{b+1} done rc={r.returncode}, cumulative found {found}/{len(missing)}')

# anything still missing gets a full sweep of all blobs not yet scanned... report first
still = [(cam, name) for cam, name in missing if not os.path.exists(os.path.join(CH, 'samples', cam, name))]
log(f'after targeted scans still missing: {len(still)}')
for cam, name in still[:20]:
    log(f'  MISS {cam} {name}')

# final global verification over all samples
nmiss = 0
for s in d['data_list']:
    for cam, ent in s['images'].items():
        if not os.path.exists(os.path.join(CH, 'samples', cam, os.path.basename(ent['img_path']))):
            nmiss += 1
log(f'FINAL missing current-frame images across all 6019 samples: {nmiss}')
