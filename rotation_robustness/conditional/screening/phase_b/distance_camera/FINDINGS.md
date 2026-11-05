# Distance × selected camera direction

## Question and sequential selection

After Phase A and the fixed r_rot × LiDAR support analysis were saved and published, test whether selected-camera differences persist within fixed distance strata. Phase A showed directional differences in degradation; this pair asks a different question from geometric sensitivity and point support, and controls the obvious distance mixture without new feature extraction.

## Findings

At σ=5°, within ≥40 m, front-selected objects have D=23.409pp and G=13.696pp (N=2,687), compared with rear-selected objects D=12.727pp and G=5.918pp (N=2,761). Directional association therefore remains within distance strata; the marginal pattern is not entirely explained by distance mixture.

There is no universal directional ranking. Within 30–40 m, back-right has larger D than front (16.000 versus 11.275pp), but smaller G (4.085 versus 7.450pp), because its initial benefit is much larger (11.915 versus 3.826pp). Separating benefit from degradation changes the interpretation.

Near groups have very small or nonpositive clean benefit. For example, front-right at 0–10 m has B=−0.365pp, so positive G there is not a newly induced loss of clean advantage. Empty cells remain explicit; the two-object missing-camera cell at 10–20 m is flagged. All 53,043 objects, including 379 missing-camera observations, remain in the table. All 30 distance-by-known-camera cells satisfy the predefined count/scene support rule.

[Heatmaps](figures/benefit_degradation.png) · [All conditions and seed values](table.csv) · [Full summary and paired scene intervals](summary.json)

Figure abbreviations: F=front, FR=front-right, FL=front-left, B=back, BL=back-left, BR=back-right. Values are percentage-point differences in object miss rates, not official mAP/NDS.

## Interpretation and limits

The camera label is assigned by largest clean projected width; it is not the contribution of one isolated camera. All six cameras are fused, and frozen realized per-camera rotations differ. LiDAR support, continuous distance, projection geometry, appearance and scene composition remain confounded. These results support a conditional association, not a causal camera ranking or a deployment rule. Seed SD and scene-only intervals are retained in the summary rather than treating five rotations as independent GT samples.

## Stop decision

This completes the second distinct new pair. Together with the accepted Distance × r_rot and Distance × LiDAR analyses, the results cover geometry, point support and camera direction. Width is closely tied to existing geometry, while off-axis angle and scene motion proxies showed weaker screening differences. More pairs would add redundancy or require missing metadata. No further pair search or robustness development is started.
