# Temporal Misalignment (T1) pipeline

在磁盘受限的云端实例(AutoDL 50GB 数据盘)上对 nuScenes val(6,019 样本)完成
BEVFusion baseline 全量评测 + T1 时间错位实验的全部脚本。

原始路径前缀 `remote_` 已去掉;路径假设实例布局:
- 代码:`/root/mmdetection3d`(mmdet3d v1.4.0 + BEVFusion)
- 数据 blob:`/root/autodl-pub/nuScenes/Fulldatasetv1.0/Trainval/v1.0-trainval{01..10}_blobs.tgz`
- 工作盘:`/root/autodl-tmp/`(chunk / camhist / preds / preds_temporal / sub_table.json)

## 执行顺序

1. **环境搭建**:`setup_env_1.sh` → `setup_env_5.sh`
   (conda env bevfusion;torch 2.1.1+cu121;mmcv 2.1.0 / mmdet 3.2.0;mmdet3d v1.4.0 editable;
   spconv-cu121;编译 BEVFusion 算子;下载官方 LiDAR-only / fusion 权重)
2. **Baseline 全量 val(分块流水)**:`extract_val.sh` → `chunk_pipeline.sh` →(补漏)`pass2.sh` → `pass3.sh` → `fix_pass3.py` → `pass3_rerun.sh` → `accumulate2.py` + `merge_eval.py`
   逐 blob 解压→按"文件实际存在"筛 pkl 子集→LiDAR-only + fusion 推理→累计预测→删数据控磁盘。
3. **T1 替换表**:`gen_subtable.py`(沿 sample_data.prev 走 k=1,2,输出 sub_table.json + hist_patterns.txt)
   → `hist_extract.sh`(从 10 个 blob 抽全部 70,428 张历史相机图到 camhist,常驻)
   → `chain.sh`(等 hist 完成 → pilot blob01 含 clean 恒等重跑 → `pilot_check.py` → 全量 10 blob)
4. **T1 主循环**:`temporal_pipeline.sh`(每 blob:组 chunk → `make_kpkl.py` 生成 k1/k2 错位 pkl → 仅 fusion 推理 → 累计;恒等噪声底由 pilot 验证 <0.0002)
5. **T1 补漏**:`temporal_pass2.sh`(跨 blob 样本按精确文件名收集)→ `tp2_move.py` → `make_kpkl.py` → 推理
   → `fix_tp2_sweeps.py`(501 个官方打包错位 sweep 硬链接归位 + pkl 路径校验 MISSING=0)
6. **评测**:`delay_stats.py`(延迟分布)→ `subset_eval.py`(clean/k1/k2/LiDAR 四份预测在共同 5,869 子集上配对算官方指标)

## 结果(2026-10-27,RTX 3090)

配对子集 N=5869:fusion clean 0.6855 → k1(≈77ms)0.6810 → k2(≈168ms)0.6699;LiDAR-only 0.6479。
轻度错位内无 backfire;详见 `docs/temporal_misalignment/` 与 `results/temporal_misalignment/metrics_summary.json`。
