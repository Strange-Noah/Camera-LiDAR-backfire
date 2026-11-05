# Fixed r_rot × LiDAR support

## Question and selection rationale

After Phase A was completed and published, test whether GT LiDAR support still distinguishes degradation within comparable fixed geometric sensitivity groups. Phase A showed large differences in both initial benefit and degradation for r_rot and point count; their joint behavior could distinguish geometric sensitivity from camera dependence. This is different from rerunning the already accepted Distance × r_rot or Distance × LiDAR analyses.

## Observation

At σ=5°, within 0.5≤r_rot<1, mean D is **14.180 / 6.490 / 1.186pp** for 0–2 / 3–10 / >10 LiDAR points (N=3,739 / 5,935 / 11,163). Within r_rot≥1, D is **20.531 / 8.490 / 2.676pp** (N=7,526 / 7,456 / 3,401). Corresponding G is **5.221 / 3.811 / 0.658pp** and **10.114 / 5.794 / 1.647pp**.

Thus point support retains a strong conditional association inside the two populated higher-r groups. Sparse groups also have substantially larger initial benefit: e.g. B at r≥1 is 10.417 / 2.696 / 1.029pp. Geometric sensitivity alone is insufficient to summarize how much the camera helps and how much can be lost.

The 0.1–0.5, 0–2-point cell has only 174 objects and already lacks clean benefit. It is flagged and receives no robust interpretation. The low-r group contains only dense observations, so no point-count contrast is possible there. Missing geometry remains explicit, including a one-object sparse cell.

[Heatmaps](figures/benefit_degradation.png) · [Full seed table](table.csv) · [Summary and scene intervals](summary.json)

## Limits and next question

The broad r bins do not equalize continuous r, distance, appearance or visibility. These are descriptive conditional associations, not isolated causal effects or deployment-ready thresholds. Reused Distance controls support a complementary interpretation but do not eliminate every composition difference.

A second pair is justified by a different question: Phase A selected-camera directions differ in D, while both geometry and point support have now been examined. **Distance × selected camera** can assess whether directional differences largely reflect the distance mixture or remain within distance strata. Off-axis angle and scene motion proxies show weaker screening effects and will not trigger additional pairs.
