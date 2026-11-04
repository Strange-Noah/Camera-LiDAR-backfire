"""Offline E4-compatible primitives; no inference or official scoring.

Matches Temporal conditional step2_match.py float64 semantics, including stable
prediction score ties and first GT in original order for equal distances.
Callers must establish dataset eligibility and global coordinates BEFORE calling.
"""
import numpy as np


def match_sample(gt, predictions):
    """Return one hit per ordered GT; keys are (sample_token, annotation_token)."""
    keys = [(g['sample_token'], g['annotation_token']) for g in gt]
    if len(keys) != len(set(keys)):
        raise ValueError('duplicate GT annotation')
    tokens = {k[0] for k in keys}
    if len(tokens) > 1:
        raise ValueError('multiple samples')
    hit = np.zeros(len(gt), dtype=bool)
    for g in gt:
        if not np.isfinite(g['translation']).all():
            raise ValueError('nonfinite GT')
    for p in predictions:
        if not np.isfinite(p['translation']).all() or not np.isfinite(p['detection_score']):
            raise ValueError('nonfinite prediction')
        if tokens and p['sample_token'] not in tokens:
            raise ValueError('wrong sample')
    for cls in dict.fromkeys(g['name'] for g in gt):
        ids = np.array([i for i,g in enumerate(gt) if g['name'] == cls])
        xy = np.array([gt[i]['translation'][:2] for i in ids], dtype=float)
        preds = [p for p in predictions if p['detection_name'] == cls and p['detection_score'] >= .1]
        preds.sort(key=lambda p: -p['detection_score'])
        for p in preds:
            d = np.hypot(p['translation'][0]-xy[:,0], p['translation'][1]-xy[:,1])
            d[hit[ids]] = np.inf
            j = int(np.argmin(d))
            if d[j] < 2.:
                hit[ids[j]] = True
    return hit


def validate_population(expected, conditions):
    if len(expected) != len(set(expected)):
        raise ValueError('duplicate sample')
    if len(conditions) != 27:
        raise ValueError('expected 27 conditions')
    for tokens in conditions:
        if len(tokens) != len(set(tokens)) or set(tokens) != set(expected):
            raise ValueError('population mismatch; intersection forbidden')


def summarize(scene_n, scene_fn, bootstrap=500, seed=20261104):
    """Counts columns: LiDAR, clean, sigma1 seeds0..4, ..., sigma5 seeds0..4.

    Scene-only paired percentile CI conditional on these five fixed calibration
    seeds; sample SD is separately reported and is not a population CI.
    """
    n = np.asarray(scene_n, dtype=float)
    fn = np.asarray(scene_fn, dtype=float)
    if fn.shape != (len(n),27) or not np.isfinite(fn).all() or not np.isfinite(n).all():
        raise ValueError('invalid count shape/values')
    if (n < 0).any() or (fn < 0).any() or (fn > n[:,None]).any() or (fn % 1).any() or (n % 1).any():
        raise ValueError('invalid counts')
    if n.sum() == 0:
        return {'N':0, 'status':'EMPTY'}
    rate = 100*fn.sum(0)/n.sum()
    B = rate[0]-rate[1]
    D = (rate[2:]-rate[1]).reshape(5,5)
    G = (rate[2:]-rate[0]).reshape(5,5)
    rng = np.random.default_rng(seed)
    w = rng.multinomial(len(n), np.full(len(n),1/len(n)), size=bootstrap)
    den = w@n
    valid = den > 0
    br = 100*(w[valid]@fn)/den[valid,None]
    bg = (br[:,2:]-br[:,0,None]).reshape(-1,5,5).mean(2)
    mean = G.mean(1)
    brackets = [[i,i+1] for i in range(1,5) if mean[i-1] <= 0 < mean[i]]
    return dict(N=int(n.sum()), scenes=int((n>0).sum()), B=B,
                rates_percent=rate.tolist(), D=D.tolist(), G=G.tolist(),
                D_mean=D.mean(1).tolist(), D_sd=D.std(1,ddof=1).tolist(),
                G_mean=mean.tolist(), G_sd=G.std(1,ddof=1).tolist(),
                backfire_seeds=(G>0).sum(1).tolist(),
                scene_only_G_CI=np.percentile(bg,[2.5,97.5],axis=0).T.tolist(),
                valid_bootstrap=int(valid.sum()), baseline_no_benefit=bool(B<=0),
                first_positive_sigma=next((i+1 for i,v in enumerate(mean) if v>0),None),
                upward_brackets=brackets, status='DESCRIPTIVE')
