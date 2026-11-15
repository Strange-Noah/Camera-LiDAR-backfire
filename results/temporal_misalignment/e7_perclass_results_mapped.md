# E7 per-class AP (2m center distance, paired subset n = 5,869) — mapped by prediction file
# Source: remote /root/autodl-tmp/subeval_pc/*_pc.json (official nuScenes label_aps, 2026-11-09)

| setting | mAP | car | truck | bus | trailer | cv | ped | bicycle | moto | barrier | cone |
|---|---|---|---|---|---|---|---|---|---|---|---|
| fusion clean | 0.6855 | 92.74 | 72.30 | 88.64 | 61.95 | 38.76 | 88.59 | 64.32 | 79.53 | 75.66 | 79.62 |
| D3 (density) | 0.6765 | 92.44 | 71.67 | 88.27 | 60.92 | 37.87 | 88.17 | 62.16 | 76.76 | 75.27 | 79.60 |
| D6 (density) | 0.6564 | 91.80 | 70.10 | 86.52 | 58.67 | 36.28 | 86.73 | 56.83 | 72.85 | 74.59 | 78.62 |
| S3 (staleness) | 0.6626 | 91.12 | 71.13 | 87.98 | 61.13 | 38.84 | 85.74 | 62.15 | 76.19 | 74.58 | 77.92 |
| S6 (staleness) | 0.6517 | 89.90 | 70.27 | 87.72 | 60.36 | 38.43 | 83.60 | 60.93 | 74.68 | 74.40 | 77.77 |
| L3 (replica) | 0.6106 | 87.79 | 70.21 | 85.89 | 59.82 | 38.01 | 85.20 | 58.32 | 69.37 | 74.42 | 76.23 |
| L6 (replica) | 0.5254 | 65.91 | 55.18 | 66.11 | 52.78 | 34.14 | 80.58 | 47.36 | 51.76 | 73.08 | 72.57 |

## Interaction term per class (L − D − S + clean, pp)

| class | kl=3 | kl=6 |
|---|---|---|
| car | −3.0 | **−23.1** |
| bus | −1.7 | **−19.5** |
| truck | −0.3 | **−12.9** |
| motorcycle | −4.1 | **−16.2** |
| bicycle | −1.7 | −6.1 |
| trailer | −0.3 | −4.3 |
| pedestrian | −0.1 | −1.2 |

Reading: density alone mostly hurts bicycle/motorcycle (small objects needing point density);
staleness alone spreads the damage across pedestrian/car/motorcycle; the super-additive
interaction at kl=6 concentrates on fast rigid vehicles (car, bus, truck, motorcycle) while
pedestrian is almost unaffected (−1.2 pp) — consistent with the motion-anchoring mechanism
(pedestrian motion between frames is small and predictable; vehicles move far in 300 ms).
