"""CPU-only dataset/calibration figures using the official devkit."""
import json
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from nuscenes.nuscenes import NuScenes
root=Path('/workspace/datasets/nuscenes-trainval-smoke')
manifest=json.loads((root/'subset-manifest.json').read_text())
nusc=NuScenes(version='v1.0-trainval', dataroot=str(root), verbose=False)
sample=nusc.get('sample',manifest['sample_token'])
out=Path('/home/bfbaseline/trainval-figures');out.mkdir()
nusc.render_pointcloud_in_image(sample['token'],pointsensor_channel='LIDAR_TOP',
    camera_channel='CAM_FRONT',out_path=str(out/'camera-lidar-projection.png'))
plt.close('all')
nusc.render_sample_data(sample['data']['LIDAR_TOP'],with_anns=True,underlay_map=False,
    axes_limit=40,out_path=str(out/'lidar-ground-truth.png'),verbose=False)
plt.close('all')
print('TRAINVAL_CPU_FIGURES_SAVED')
