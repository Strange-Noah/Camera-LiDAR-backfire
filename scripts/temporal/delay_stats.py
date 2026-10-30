# 统计 temporal 替换的实际延迟分布(基于 sub_table.json)
import json, statistics

t = json.load(open('/root/autodl-tmp/sub_table.json'))
for k in (1, 2):
    ds = []
    n_elig = 0
    for tok, entry in t.items():
        ok = True
        for cam, e in entry.items():
            r = e.get(f'k{k}')
            if r is None:
                ok = False
                break
        if not ok:
            continue
        n_elig += 1
        # 用 CAM_FRONT 代表(六路同一车,时间基本同步)
        e = entry['CAM_FRONT']
        ds.append((e['cur_ts'] - e[f'k{k}']['ts']) / 1e6)
    ds_ms = [d * 1000 for d in ds]
    ds_ms.sort()
    n = len(ds_ms)
    print(f'k={k}: eligible={n_elig} delay_ms mean={statistics.mean(ds_ms):.1f} '
          f'median={statistics.median(ds_ms):.1f} p5={ds_ms[int(n*0.05)]:.1f} '
          f'p95={ds_ms[int(n*0.95)]:.1f} max={ds_ms[-1]:.1f}')
