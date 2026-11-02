# Build exact token -> blob-index map (0..9) from blob member listings.
import pickle, json, os

lists = {}
for b in range(1, 11):
    p = f'/root/autodl-tmp/bloblists/b{b}.txt'
    lists[b - 1] = set(open(p).read().splitlines())

d = pickle.load(open('/root/autodl-tmp/chunk/nuscenes_infos_val.pkl', 'rb'))
out = {}
missing = []
for s in d['data_list']:
    base = 'samples/LIDAR_TOP/' + os.path.basename(s['lidar_points']['lidar_path'])
    hit = [b for b, ls in lists.items() if base in ls]
    if hit:
        out[s['token']] = hit[0]
    else:
        missing.append(base)
json.dump(out, open('/root/autodl-tmp/tok2blob.json', 'w'))
print('mapped', len(out), 'of', len(d['data_list']), '| unmapped', len(missing))
for m in missing[:10]:
    print('  MISS', m)
