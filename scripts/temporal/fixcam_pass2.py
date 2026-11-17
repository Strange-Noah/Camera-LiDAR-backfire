# Pass 2 repair: find remaining missing current-frame camera images.
# Launches parallel tar scans over all 10 blobs; early-exits once all restored.
import pickle, os, subprocess, time

CH = '/root/autodl-tmp/chunk'
BLOBS = [f'/root/autodl-pub/nuScenes/Fulldatasetv1.0/Trainval/v1.0-trainval{i:02d}_blobs.tgz' for i in range(1, 11)]
LOG = open('/root/autodl-tmp/fixcam_pass2.log', 'a')
def log(m):
    LOG.write(f'[{time.strftime("%H:%M:%S")}] {m}\n'); LOG.flush()

d = pickle.load(open(f'{CH}/nuscenes_infos_val.pkl', 'rb'))

def missing():
    out = []
    for s in d['data_list']:
        for cam, ent in s['images'].items():
            p = os.path.join(CH, 'samples', cam, os.path.basename(ent['img_path']))
            if not os.path.exists(p):
                out.append(f'samples/{cam}/{os.path.basename(ent["img_path"])}')
    return out

mis = missing()
log(f'pass2 start missing={len(mis)}')
if not mis:
    log('PASS2_NOTHING_TO_DO')
    raise SystemExit(0)

pat = '/root/autodl-tmp/pass2_all.txt'
procs = []
for b in range(10):
    open(pat, 'w').write('\n'.join(mis))
    procs.append((b, subprocess.Popen(
        ['tar', '-xzf', BLOBS[b], '-C', CH, '-T', pat],
        stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)))

deadline = time.time() + 4 * 3600
while time.time() < deadline:
    time.sleep(60)
    mis = missing()
    log(f'check: still missing={len(mis)}')
    if not mis:
        for b, p in procs:
            if p.poll() is None:
                p.kill()
        log('ALL_RESTORED')
        break
else:
    log('TIMEOUT')

mis = missing()
log(f'pass2 end missing={len(mis)}')
for m in mis[:30]:
    log(f'  MISS {m}')
