# T1 vs T1B Delay Labels — Measured Offsets (Addendum)

**Date:** 2026-11-10
**Status:** COMPLETE (CPU only, substitution tables on remote)
**Answers:** the open delay-label question from @wangYilia's 2026-11-07 review of `t1b_fn_analysis_en.md`.

## Question recap

The T1B FN report labels k10 ≈ 834 ms and k12 ≈ 1010 ms (measured), while the data handoff quoted "T1 k10 ≈ 958 ms". Which is right, and what are the measured T1 k9/k10 offsets?

## Method

Measured directly from the substitution tables actually consumed by the inference drivers — `sub_table10.json` (T1 k-series) and `sub_table_t1b2.json` (T1B k10/k12 extension). Each table stores, per token and camera, the exact replacement file and its timestamp. Delay = `cur_ts − replacement_ts` on CAM_FRONT (same convention as the T1B report), over tokens where all six cameras have a k-entry.

## Results

| k | T1 (sub_table10) eligible | T1 mean delay | T1 median | T1B reported |
|---|---|---|---|---|
| k8 | 5,719 | 671.9 ms | 650 ms | 672 ms |
| k9 | 5,719 | **757.6 ms** | 750 ms | — |
| k10 | 5,718 | **834.1 ms** | 850 ms | 834 ms |
| k12 | (not in table) | — | — | 1010 ms |

## Key finding: T1 and T1B select identical frames

For all 5,719 tokens eligible at k10, the replacement file in `sub_table10.json` (T1) is **byte-identical** to the one in `sub_table_t1b2.json` (T1B) — 0 differences. This is by design: the T1B "ego-consistent" variant changes only the **calibration handling** (stale frame delivered with its own ego pose instead of the current ego pose); the stale frame itself is the same.

**Conclusion: the "~958 ms" figure in the handoff was wrong (a nominal-label leftover); the measured T1 k10 offset is 834 ms mean / 850 ms median — identical to T1B k10, as it must be since the frames are the same.** T1 k9 = 758 ms mean / 750 ms median.

## Recommended label convention (both series)

Use measured CAM_FRONT means: k2 = 167 ms, k5 = 420 ms, k8 = 672 ms, k9 = 758 ms, k10 = 834 ms, k12 = 1010 ms. Where a T1-specific label is needed (e.g. plots of the T1 raw-swap series), use the same numbers — frame selection is identical; only calibration treatment differs between T1 and T1B.

## Artifacts

- Script: `calc_t1_k910.py` (remote `/root/`), output: k8/k9/k10 rows above
- Input tables: `/root/autodl-tmp/sub_table10.json`, `/root/autodl-tmp/sub_table_t1b2.json`
