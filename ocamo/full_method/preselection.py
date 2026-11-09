"""Binary LogitBoost with weighted least-squares regression stumps.

No sklearn/new package required. Stumps are classification-tree base learners;
unpublished MATLAB ensemble settings cannot be claimed numerically identical.
"""
import numpy as np


def _stump(x, weight, response):
    best = None
    total_w = weight.sum()
    total_z = np.dot(weight, response)
    # Maximize explained weighted response energy, equivalent to minimum SSE.
    for axis in range(x.shape[1]):
        order = np.argsort(x[:, axis], kind='stable')
        xx, ww, zz = x[order, axis], weight[order], response[order]
        cw, cz = np.cumsum(ww), np.cumsum(ww*zz)
        candidates = np.flatnonzero(np.diff(xx) > 0)
        if not len(candidates):
            continue
        left_w, right_w = cw[candidates], total_w-cw[candidates]
        valid = (left_w > 1e-12) & (right_w > 1e-12)
        candidates = candidates[valid]
        if not len(candidates):
            continue
        lw, rw = cw[candidates], total_w-cw[candidates]
        lz, rz = cz[candidates], total_z-cz[candidates]
        scores = lz**2/lw+rz**2/rw
        i = int(np.argmax(scores)); index = candidates[i]
        if best is None or scores[i] > best[0]:
            best = (scores[i], {'axis': axis, 'threshold': float((xx[index]+xx[index+1])/2),
                               'left': float(lz[i]/lw[i]), 'right': float(rz[i]/rw[i])})
    if best is None:
        return {'axis': 0, 'threshold': 0., 'left': float(total_z/total_w),
                'right': float(total_z/total_w)}
    return best[1]


def predict_tree(x, tree):
    return np.where(x[:, tree['axis']] <= tree['threshold'], tree['left'], tree['right'])


def fit(x, y, rounds=100):
    x, y = np.asarray(x, float), np.asarray(y, float)
    if x.ndim != 2 or x.shape[1] != 42 or len(x) != len(y):
        raise ValueError('Expected n by 42 labeled training observations')
    if not np.isfinite(x).all() or set(y.tolist()) != {0., 1.}:
        raise ValueError('Need finite features and both training classes; no fake classifier')
    score = np.zeros(len(x)); trees = []
    for _ in range(rounds):
        p = 1/(1+np.exp(-np.clip(2*score, -40, 40)))
        w = np.maximum(p*(1-p), 1e-12)
        z = (y-p)/w
        tree = _stump(x, w, z)
        score += .5*predict_tree(x, tree)
        trees.append(tree)
    return {'algorithm': 'binary LogitBoost', 'base_learner': 'regression stump',
            'rounds': rounds, 'trees': trees, 'threshold': 0.}


def scores(model, x):
    x = np.asarray(x, float)
    return .5*sum((predict_tree(x, tree) for tree in model['trees']), np.zeros(len(x)))


def metrics(y, prediction):
    y, prediction = np.asarray(y, int), np.asarray(prediction, bool)
    tp = int(((y == 1) & prediction).sum()); tn = int(((y == 0) & ~prediction).sum())
    fp = int(((y == 0) & prediction).sum()); fn = int(((y == 1) & ~prediction).sum())
    return {'tp': tp, 'tn': tn, 'fp': fp, 'fn': fn,
            'accuracy': float((tp+tn)/len(y)) if len(y) else None,
            'suitable_precision': tp/(tp+fp) if tp+fp else None,
            'suitable_recall': tp/(tp+fn) if tp+fn else None}


def train_scene_disjoint(training, validation, evaluation_scenes, rounds=100, data_loss=.4):
    train_scenes = sorted({r['scene'] for r in training})
    val_scenes = sorted({r['scene'] for r in validation})
    if (set(train_scenes) & set(val_scenes) or
            (set(train_scenes) | set(val_scenes)) & set(evaluation_scenes)):
        raise ValueError('Scene leakage between train/validation/evaluation')
    if len(train_scenes) < 5 or not val_scenes:
        raise ValueError('Need at least five training scenes and independent validation')
    folds = []
    for fold in range(5):
        held = set(train_scenes[fold::5])
        tr = [r for r in training if r['scene'] not in held]
        te = [r for r in training if r['scene'] in held]
        model = fit([r['features'] for r in tr], [r['label'] for r in tr], rounds)
        prediction = scores(model, [r['features'] for r in te]) >= 0
        folds.append({'fold': fold, 'held_scenes': sorted(held),
                      'metrics': metrics([r['label'] for r in te], prediction)})
    model = fit([r['features'] for r in training], [r['label'] for r in training], rounds)
    val_scores = scores(model, [r['features'] for r in validation])
    model['threshold'] = float(np.quantile(val_scores, data_loss, method='linear'))
    model['training_scenes'] = train_scenes
    model['validation_scenes'] = val_scenes
    model['evaluation_scenes'] = list(evaluation_scenes)
    model['requested_validation_data_loss'] = data_loss
    model['actual_validation_data_loss'] = float(np.mean(val_scores < model['threshold']))
    model['validation_metrics'] = metrics([r['label'] for r in validation],
                                          val_scores >= model['threshold'])
    model['scene_cross_validation'] = folds
    return model
