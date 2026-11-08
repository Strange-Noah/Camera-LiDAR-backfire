# Rotation-only correction record and geometry adaptation

Column-vector transforms use `T_A_B` to map B coordinates to A. With the actual acquisition-specific ego poses:

`T_C_L = inverse(T_W_Ecamera @ T_Ecamera_C) @ (T_W_Elidar @ T_Elidar_L)`.

The camera and LiDAR timestamps need not coincide. This compensates ego motion between captures, not independently moving objects. The clean camera→LiDAR matrix is `inverse(T_C_L)=[R,t]`. Inject one persistent camera-local Q per scene/camera/seed as `[R@Q,t]`, matching the historical right-multiplication spatial convention while keeping the same Q through this sequence. Historical accepted runs are not modified or reinterpreted.

The estimator retains its original three parameter axes in LiDAR coordinates. To keep translation fixed while evaluating rotations about the camera origin, its feature input is `p_L-t`. With corrupted rotation Rc, the adapted projection is

`p_C(theta) = Rc.T @ Exp(theta) @ (p_L-t)`.

Kernel correlation still uses the exact original unnormalized negative sum of `exp(-d²/(2h²))` over k nearest edges, h=9px,k=10. Only projection/intrinsics/bounds and centered input geometry differ. No feature weights, new regularizer, feature-count normalization, Hessian protection or optimizer changes are introduced. The callback binding is isolated in the adapter; the official source files remain untouched. Actual1600×900 bounds and per-camera K replace CARLA constants. Canny thresholds stay50/100; the edge crop scales from row360 at1080px to the lower two-thirds of the actual image.

nuScenes features retain explicit point indices and ring labels. Raw fifth-channel ring IDs are checked, each ring is stably sorted by azimuth, official get_corners is applied inside each ring, and the first/last five positions are excluded to avoid seam/convolution-boundary artifacts. This is an adaptation, not a claim of identical CARLA scan ordering. Coordinates/indices remain available before summation for a later separately authorized weighting study.

The estimated camera→LiDAR rotation is `Rest=Exp(theta).T @ Rc`, translation t unchanged. Export the right correction `A=Rc.T @ Rest` in a homogeneous4×4 matrix with zero translation. Then `T_corrected=T_corrupted @ A4`. The exact oracle is `A=Q.T`; its projection and calibration geometry are restored to floating-point tolerance. Estimated records were exported from actual online states, not substituted oracle estimates.

Each remote corrections.jsonl record includes scene/sample IDs, camera, camera/LiDAR timestamps, frame index/history length, seed, theta, corrupted C2L, right correction, corrected C2L, feature/edge counts, residual rotation, acquisition time, upstream commit and convention. The run summary/config gives perturbation scale and estimator settings. Full records remain remote; no raw sensor images/features are published. Convergence graphs contain no sensor imagery. Clean-projection diagnostic overlays were generated remotely and retained there; numeric transform chains are checked, but these sensor-image overlays were not downloaded or independently visually reviewed on the Mac.

Calibration state is recreated per scene and per seed. Ten burn-in updates are retained and checked; no pre-update convergence claim is made. Because the estimator parameter axes are LiDAR-based while the injected Q is camera-local, its ideal numerical theta can vary slightly with acquisition-specific relative poses. Export always uses the current transform. Dynamic objects, asynchronous sensors, edge choices and a short keyframe sequence remain potential limitations; this run does not identify the cause of mixed convergence.

For optional BEVFusion evaluation, join by scene/sample/camera and insert corrected camera2lidar into a separate input-calibration view, recompute its inverse/projection matrices, and preserve LiDAR tensors, GT, intrinsic/model transforms and checkpoint identities. Do not overwrite accepted configurations or predictions. This integration has not been executed in BEVFusion; no mAP/NDS recovery is claimed and GPU inference remains unauthorized.
