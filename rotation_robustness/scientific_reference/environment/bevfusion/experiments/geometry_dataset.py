"""Inject into a defensive copy after upstream get_data_info, before pipeline."""
import copy
import numpy as np
from .rotation_protocol import CHANNELS, calibrated_matrices


def inject_geometry(clean, channels, cameras):
    if len(channels) != 6 or set(channels) != set(CHANNELS):
        raise ValueError('Unverified Camera channel mapping')
    result = copy.deepcopy(clean)
    trace = []
    for i, channel in enumerate(channels):
        before = np.asarray(clean['camera2lidar'][i])
        t, inv, proj = calibrated_matrices(before, clean['camera_intrinsics'][i], cameras[channel])
        # Identity keeps the ORIGINAL clean float32 inverse/projection byte-for-byte.
        # Recomputing an inverse of rounded float32 R can differ slightly from upstream.
        if cameras[channel]['sigma_R_degrees_per_axis'] != 0:
            for key, value in [('camera2lidar',t),('lidar2camera',inv),('lidar2image',proj)]:
                result[key][i] = value.astype(np.asarray(clean[key][i]).dtype)
        trace.append(dict(channel=channel, clean_camera2lidar=before.tolist(),
            camera2lidar=np.asarray(result['camera2lidar'][i]).tolist(),
            lidar2camera=np.asarray(result['lidar2camera'][i]).tolist(),
            lidar2image=np.asarray(result['lidar2image'][i]).tolist(),
            camera_intrinsics=np.asarray(clean['camera_intrinsics'][i]).tolist(), parameters=cameras[channel]))
    return result, trace


def wrap_dataset(dataset, cameras, trace_callback=None, clean_repeat=False):
    original = dataset.__class__
    class RotatedDataset(original):
        def get_data_info(self, index):
            info = self.data_infos[index]
            channels = list(info['cams'])  # Same iteration as pinned upstream.
            clean = super().get_data_info(index)
            if len(channels)!=6 or set(channels)!=set(CHANNELS):
                raise ValueError('Unexpected infos Camera keys')
            for i, c in enumerate(channels):
                if str(clean['image_paths'][i]) != str(info['cams'][c]['data_path']):
                    raise ValueError('Camera order/path mismatch')
            changed, trace = inject_geometry(clean, channels, cameras)
            if trace_callback:
                trace_callback(info['token'], trace)
            return clean if clean_repeat else changed
    dataset.__class__ = RotatedDataset
    return dataset
