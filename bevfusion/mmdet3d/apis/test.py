from tqdm import tqdm
import torch


def _move_to_device(value, device):
    if isinstance(value, torch.Tensor):
        return value.to(device, non_blocking=True)
    if isinstance(value, dict):
        return {key: _move_to_device(item, device) for key, item in value.items()}
    if isinstance(value, list):
        return [_move_to_device(item, device) for item in value]
    if isinstance(value, tuple):
        return tuple(_move_to_device(item, device) for item in value)
    return value


def single_gpu_test(model, data_loader):
    model.eval()
    results = []
    dataset = data_loader.dataset
    device = next(model.parameters()).device
    device_input_keys = {
        "img",
        "points",
        "radar",
        "camera2ego",
        "lidar2ego",
        "lidar2camera",
        "lidar2image",
        "camera_intrinsics",
        "camera2lidar",
        "img_aug_matrix",
        "lidar_aug_matrix",
        "depths",
    }
    for data in tqdm(data_loader, total=len(data_loader), desc="Inference"):
        # MMCV's former MMDataParallel scattered these tensors to the selected
        # GPU. The compatibility runner uses a plain model, so do it here.
        for key in device_input_keys.intersection(data):
            data[key] = _move_to_device(data[key], device)
        with torch.no_grad():
            result = model(return_loss=False, rescale=True, **data)
        results.extend(result)

    return results
