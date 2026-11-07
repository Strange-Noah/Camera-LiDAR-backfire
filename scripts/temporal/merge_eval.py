# 合并 18 个日志的预测结果,用官方 NuScenesEval 一次性评测(等价于全量 val)
import json, glob, os, sys

PRED = '/root/autodl-tmp/preds'
EVALROOT = '/root/autodl-tmp/evalroot'

def merge(model):
    out = f'{PRED}/{model}_cumulative.json'
    r = json.load(open(out))
    print(model.upper(), 'SAMPLES', len(r['results']), '->', out)
    return out

def evaluate(model, result_path):
    from nuscenes import NuScenes
    from nuscenes.eval.detection.evaluate import NuScenesEval
    from nuscenes.eval.detection.configs.config_factory import detection_configs
    nusc = NuScenes(version='v1.0-trainval', dataroot=EVALROOT, verbose=False)
    nusc_eval = NuScenesEval(nusc, config=detection_configs('detection_cvpr_2019'),
                             result_path=result_path, eval_set='val',
                             output_dir=os.path.dirname(result_path), verbose=False)
    nusc_eval.main(render_curves=False)
    s = json.load(open(os.path.join(os.path.dirname(result_path), 'metrics_summary.json')))
    print(f"[{model}] mAP={s['mean_ap']:.4f} NDS={s['nd_score']:.4f} "
          f"mATE={s['tp_errors']['trans_err']:.4f} mASE={s['tp_errors']['scale_err']:.4f} "
          f"mAOE={s['tp_errors']['orient_err']:.4f} mAVE={s['tp_errors']['vel_err']:.4f} "
          f"mAAE={s['tp_errors']['attr_err']:.4f}")

if __name__ == '__main__':
    os.makedirs(EVALROOT, exist_ok=True)
    os.symlink('/root/autodl-tmp/nuscenes_meta/v1.0-trainval', f'{EVALROOT}/v1.0-trainval') if not os.path.exists(f'{EVALROOT}/v1.0-trainval') else None
    os.symlink('/root/autodl-tmp/nuscenes_meta/maps', f'{EVALROOT}/maps') if not os.path.exists(f'{EVALROOT}/maps') else None
    models = sys.argv[1:] or ['lidar', 'fusion']
    for m in models:
        p = merge(m)
        evaluate(m, p)
