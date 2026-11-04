"""Unmodified Temporal conditional match function for parity testing.
Source: scripts/temporal/conditional_threshold/step2_match.py,
nuo-ft-conditional-threshold branch; Git blob 3f5990eaba8d7d8042291e1e5ee44a6b7293dfbb.
"""
def match_setting(results, gt_idx, tokens, ego_xy, n_boxes):
    hit = {t: np.zeros(n_boxes, bool) for t in SCORE_THRESHOLDS}
    present = {t: np.zeros(n_boxes, bool) for t in SCORE_THRESHOLDS}
    counts = {}                                           # (thr, cls, dist_label) -> [n_pred, n_fp]
    empty = (np.array([], dtype=int), np.zeros((0, 2)))
    for tok in tokens:
        ex, ey = ego_xy.get(tok, (np.nan, np.nan))
        by_cls = {}
        for p in results.get(tok, []):
            if p['detection_score'] >= SCORE_THRESHOLDS[0]:
                by_cls.setdefault(p['detection_name'], []).append(p)
        for cls, preds in by_cls.items():
            preds.sort(key=lambda p: -p['detection_score'])
            score = np.array([p['detection_score'] for p in preds])
            pxy = np.array([p['translation'][:2] for p in preds], dtype=float)
            dist_bin = np.digitize(np.hypot(pxy[:, 0] - ex, pxy[:, 1] - ey), DIST_EDGES[1:-1])
            ids, gxy = gt_idx.get((tok, cls), empty)
            for thr in SCORE_THRESHOLDS:
                sel = np.flatnonzero(score >= thr)
                if len(sel) and len(ids):
                    present[thr][ids] = True
                matched = np.zeros(len(ids), bool)
                is_fp = np.ones(len(sel), bool)
                if len(ids):
                    for n, i in enumerate(sel):
                        d = np.hypot(pxy[i, 0] - gxy[:, 0], pxy[i, 1] - gxy[:, 1])
                        d[matched] = np.inf
                        j = int(np.argmin(d))
                        if d[j] < MATCH_DIST_M:
                            matched[j] = True
                            is_fp[n] = False
                    hit[thr][ids[matched]] = True
                for b, lab in enumerate(DIST_LABELS):
                    m = dist_bin[sel] == b
                    c = counts.setdefault((thr, cls, lab), [0, 0])
                    c[0] += int(m.sum())
                    c[1] += int((is_fp & m).sum())
    fp = pd.DataFrame([dict(threshold=k[0], cls=k[1], pred_distance=k[2], n_pred=v[0], n_fp=v[1]) for k, v in counts.items()])
    return hit, present, fp
