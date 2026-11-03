import os
import pickle
from os import path as osp
from pathlib import Path

from .mmcv_compat import mmcv, track_iter_progress
import numpy as np
from pycocotools import mask as maskUtils
from pycocotools.coco import COCO
from mmengine.utils import mkdir_or_exist



def _poly2mask(mask_ann, img_h, img_w):
    if isinstance(mask_ann, list):
        # polygon -- a single object might consist of multiple parts
        # we merge all parts into one mask rle code
        rles = maskUtils.frPyObjects(mask_ann, img_h, img_w)
        rle = maskUtils.merge(rles)
    elif isinstance(mask_ann["counts"], list):
        # uncompressed RLE
        rle = maskUtils.frPyObjects(mask_ann, img_h, img_w)
    else:
        # rle
        rle = mask_ann
    mask = maskUtils.decode(rle)
    return mask


def _parse_coco_ann_info(ann_info):
    gt_bboxes = []
    gt_labels = []
    gt_bboxes_ignore = []
    gt_masks_ann = []

    for i, ann in enumerate(ann_info):
        if ann.get("ignore", False):
            continue
        x1, y1, w, h = ann["bbox"]
        if ann["area"] <= 0:
            continue
        bbox = [x1, y1, x1 + w, y1 + h]
        if ann.get("iscrowd", False):
            gt_bboxes_ignore.append(bbox)
        else:
            gt_bboxes.append(bbox)
            gt_masks_ann.append(ann["segmentation"])

    if gt_bboxes:
        gt_bboxes = np.array(gt_bboxes, dtype=np.float32)
        gt_labels = np.array(gt_labels, dtype=np.int64)
    else:
        gt_bboxes = np.zeros((0, 4), dtype=np.float32)
        gt_labels = np.array([], dtype=np.int64)

    if gt_bboxes_ignore:
        gt_bboxes_ignore = np.array(gt_bboxes_ignore, dtype=np.float32)
    else:
        gt_bboxes_ignore = np.zeros((0, 4), dtype=np.float32)

    ann = dict(bboxes=gt_bboxes, bboxes_ignore=gt_bboxes_ignore, masks=gt_masks_ann)

    return ann


def crop_image_patch_v2(pos_proposals, pos_assigned_gt_inds, gt_masks):
    import torch
    from torch.nn.modules.utils import _pair

    device = pos_proposals.device
    num_pos = pos_proposals.size(0)
    fake_inds = torch.arange(num_pos, device=device).to(dtype=pos_proposals.dtype)[
        :, None
    ]
    rois = torch.cat([fake_inds, pos_proposals], dim=1)  # Nx5
    mask_size = _pair(28)
    rois = rois.to(device=device)
    gt_masks_th = (
        torch.from_numpy(gt_masks)
        .to(device)
        .index_select(0, pos_assigned_gt_inds)
        .to(dtype=rois.dtype)
    )
    # Use RoIAlign could apparently accelerate the training (~0.1s/iter)
    from mmcv.ops import roi_align

    targets = roi_align(gt_masks_th, rois, mask_size[::-1], 1.0, 0, True).squeeze(1)
    return targets


def crop_image_patch(pos_proposals, gt_masks, pos_assigned_gt_inds, org_img):
    num_pos = pos_proposals.shape[0]
    masks = []
    img_patches = []
    for i in range(num_pos):
        gt_mask = gt_masks[pos_assigned_gt_inds[i]]
        bbox = pos_proposals[i, :].astype(np.int32)
        x1, y1, x2, y2 = bbox
        w = np.maximum(x2 - x1 + 1, 1)
        h = np.maximum(y2 - y1 + 1, 1)

        mask_patch = gt_mask[y1 : y1 + h, x1 : x1 + w]
        masked_img = gt_mask[..., None] * org_img
        img_patch = masked_img[y1 : y1 + h, x1 : x1 + w]

        img_patches.append(img_patch)
        masks.append(mask_patch)
    return img_patches, masks


def _resolve_data_file(data_root, filename):
    path = Path(filename)
    if path.is_file():
        return path
    if not path.is_absolute():
        rooted = Path(data_root) / path
        if rooted.is_file():
            return rooted
    raise FileNotFoundError(f"Could not find point cloud file: {filename}")


def _load_nuscenes_points(data_root, info, load_dim=5, sweeps_num=10):
    """Load a nuScenes keyframe and its earlier sweeps in keyframe coordinates."""
    key_path = _resolve_data_file(data_root, info["lidar_path"])
    points = np.fromfile(key_path, dtype=np.float32)
    if points.size % load_dim:
        raise ValueError(f"Unexpected point-cloud size in {key_path}: {points.size}")
    points = points.reshape(-1, load_dim).copy()
    points[:, 4] = 0
    frames = [points]
    key_time = info["timestamp"] / 1e6
    if not info.get("sweeps"):
        keep = ~((np.abs(points[:, 0]) < 1.0) & (np.abs(points[:, 1]) < 1.0))
        frames.extend(points[keep].copy() for _ in range(sweeps_num))

    sweeps = info.get("sweeps", [])[:sweeps_num]
    for sweep in sweeps:
        sweep_path = _resolve_data_file(data_root, sweep["data_path"])
        sweep_points = np.fromfile(sweep_path, dtype=np.float32)
        if sweep_points.size % load_dim:
            print(
                f"WARNING: skipping malformed sweep {sweep_path}: "
                f"{sweep_points.size} float values is not divisible by {load_dim}."
            )
            continue
        sweep_points = sweep_points.reshape(-1, load_dim).copy()
        keep = ~((np.abs(sweep_points[:, 0]) < 1.0) & (np.abs(sweep_points[:, 1]) < 1.0))
        sweep_points = sweep_points[keep]
        rotation = np.asarray(sweep["sensor2lidar_rotation"], dtype=np.float32)
        translation = np.asarray(sweep["sensor2lidar_translation"], dtype=np.float32)
        sweep_points[:, :3] = sweep_points[:, :3] @ rotation.T + translation
        sweep_points[:, 4] = key_time - sweep["timestamp"] / 1e6
        frames.append(sweep_points)
    return np.concatenate(frames, axis=0)


def _points_in_lidar_boxes(points, boxes):
    """Return an [N points, M boxes] mask for bottom-centered LiDAR boxes."""
    result = np.zeros((len(points), len(boxes)), dtype=np.bool_)
    xyz = points[:, :3]
    for index, box in enumerate(boxes):
        delta = xyz - box[:3]
        cosine, sine = np.cos(box[6]), np.sin(box[6])
        local_x = delta[:, 0] * cosine - delta[:, 1] * sine
        local_y = delta[:, 0] * sine + delta[:, 1] * cosine
        result[:, index] = (
            (np.abs(local_x) <= box[3] / 2)
            & (np.abs(local_y) <= box[4] / 2)
            & (delta[:, 2] >= 0)
            & (delta[:, 2] <= box[5])
        )
    return result


def _create_nuscenes_gt_database(
    data_path, info_prefix, info_path, used_classes, database_save_path,
    db_info_save_path, relative_path=True, sweeps_num=10,
):
    with open(info_path, "rb") as stream:
        infos = pickle.load(stream)["infos"]
    os.makedirs(database_save_path, exist_ok=True)
    all_db_infos = {}
    group_counter = 0

    for info in track_iter_progress(infos):
        if "gt_boxes" not in info:
            continue
        boxes = np.asarray(info["gt_boxes"], dtype=np.float32).reshape(-1, 7)
        names = np.asarray(info["gt_names"])
        valid = np.asarray(info.get("valid_flag", np.ones(len(boxes), dtype=bool)), dtype=bool)
        boxes, names = boxes[valid], names[valid]
        if not len(boxes):
            continue

        points = _load_nuscenes_points(data_path, info, sweeps_num=sweeps_num)
        point_indices = _points_in_lidar_boxes(points, boxes)
        group_ids = np.arange(len(boxes), dtype=np.int64)
        sample_id = info.get("token", info.get("sample_idx", "unknown"))

        for index, name in enumerate(names):
            object_points = points[point_indices[:, index]].copy()
            object_points[:, :3] -= boxes[index, :3]
            filename = f"{sample_id}_{name}_{index}.bin"
            absolute_path = osp.join(database_save_path, filename)
            object_points.tofile(absolute_path)
            if used_classes is not None and name not in used_classes:
                continue

            relative_file = osp.join(f"{info_prefix}_gt_database", filename)
            entry = {
                "name": str(name),
                "path": relative_file if relative_path else absolute_path,
                "image_idx": sample_id,
                "gt_idx": int(index),
                "box3d_lidar": boxes[index],
                "num_points_in_gt": int(len(object_points)),
                "difficulty": 0,
                "group_id": group_counter + int(group_ids[index]),
            }
            all_db_infos.setdefault(str(name), []).append(entry)
        group_counter += len(boxes)

    for name, entries in all_db_infos.items():
        print(f"load {len(entries)} {name} database infos")
    with open(db_info_save_path, "wb") as stream:
        pickle.dump(all_db_infos, stream)

def create_groundtruth_database(
    dataset_class_name,
    data_path,
    info_prefix,
    info_path=None,
    mask_anno_path=None,
    used_classes=None,
    database_save_path=None,
    db_info_save_path=None,
    relative_path=True,
    add_rgb=False,
    lidar_only=False,
    bev_only=False,
    coors_range=None,
    with_mask=False,
    load_augmented=None,
):
    """Given the raw data, generate the ground truth database.

    Args:
        dataset_class_name （str): Name of the input dataset.
        data_path (str): Path of the data.
        info_prefix (str): Prefix of the info file.
        info_path (str): Path of the info file.
            Default: None.
        mask_anno_path (str): Path of the mask_anno.
            Default: None.
        used_classes (list[str]): Classes have been used.
            Default: None.
        database_save_path (str): Path to save database.
            Default: None.
        db_info_save_path (str): Path to save db_info.
            Default: None.
        relative_path (bool): Whether to use relative path.
            Default: True.
        with_mask (bool): Whether to use mask.
            Default: False.
    """
    print(f"Create GT Database of {dataset_class_name}")
    if dataset_class_name == "NuScenesDataset" and not with_mask and not load_augmented:
        if database_save_path is None:
            database_save_path = osp.join(data_path, f"{info_prefix}_gt_database")
        if db_info_save_path is None:
            db_info_save_path = osp.join(data_path, f"{info_prefix}_dbinfos_train.pkl")
        return _create_nuscenes_gt_database(
            data_path, info_prefix, info_path, used_classes,
            database_save_path, db_info_save_path, relative_path=relative_path, sweeps_num=10,
        )

    # Dataset-builder path is only needed for non-nuScenes conversions.
    from mmdet3d.core.bbox import box_np_ops
    from mmdet3d.datasets import build_dataset
    from mmdet.structures.bbox import bbox_overlaps
    dataset_cfg = dict(
        type=dataset_class_name, dataset_root=data_path, ann_file=info_path
    )
    if dataset_class_name == "KittiDataset":
        dataset_cfg.update(
            test_mode=False,
            split="training",
            modality=dict(
                use_lidar=True,
                use_depth=False,
                use_lidar_intensity=True,
                use_camera=with_mask,
            ),
            pipeline=[
                dict(
                    type="LoadPointsFromFile",
                    coord_type="LIDAR",
                    load_dim=4,
                    use_dim=4,
                ),
                dict(
                    type="LoadAnnotations3D",
                    with_bbox_3d=True,
                    with_label_3d=True,
                ),
            ],
        )

    elif dataset_class_name == "NuScenesDataset":
        if not load_augmented:
            dataset_cfg.update(
                use_valid_flag=True,
                pipeline=[
                    dict(
                        type="LoadPointsFromFile",
                        coord_type="LIDAR",
                        load_dim=5,
                        use_dim=5,
                    ),
                    dict(
                        type="LoadPointsFromMultiSweeps",
                        sweeps_num=10,
                        use_dim=[0, 1, 2, 3, 4],
                        pad_empty_sweeps=True,
                        remove_close=True,
                    ),
                    dict(
                        type="LoadAnnotations3D", with_bbox_3d=True, with_label_3d=True
                    ),
                ],
            )
        else:
            dataset_cfg.update(
                use_valid_flag=True,
                pipeline=[
                    dict(
                        type="LoadPointsFromFile",
                        coord_type="LIDAR",
                        load_dim=16,
                        use_dim=list(range(16)),
                        load_augmented=load_augmented,
                    ),
                    dict(
                        type="LoadPointsFromMultiSweeps",
                        sweeps_num=10,
                        load_dim=16,
                        use_dim=list(range(16)),
                        pad_empty_sweeps=True,
                        remove_close=True,
                        load_augmented=load_augmented,
                    ),
                    dict(
                        type="LoadAnnotations3D", with_bbox_3d=True, with_label_3d=True
                    ),
                ],
            )

    elif dataset_class_name == "WaymoDataset":
        dataset_cfg.update(
            test_mode=False,
            split="training",
            modality=dict(
                use_lidar=True,
                use_depth=False,
                use_lidar_intensity=True,
                use_camera=False,
            ),
            pipeline=[
                dict(
                    type="LoadPointsFromFile",
                    coord_type="LIDAR",
                    load_dim=6,
                    use_dim=5,
                ),
                dict(
                    type="LoadAnnotations3D",
                    with_bbox_3d=True,
                    with_label_3d=True,
                ),
            ],
        )

    dataset = build_dataset(dataset_cfg)

    if database_save_path is None:
        database_save_path = osp.join(data_path, f"{info_prefix}_gt_database")
    if db_info_save_path is None:
        db_info_save_path = osp.join(data_path, f"{info_prefix}_dbinfos_train.pkl")
    mkdir_or_exist(database_save_path)
    all_db_infos = dict()
    if with_mask:
        coco = COCO(osp.join(data_path, mask_anno_path))
        imgIds = coco.getImgIds()
        file2id = dict()
        for i in imgIds:
            info = coco.loadImgs([i])[0]
            file2id.update({info["file_name"]: i})

    group_counter = 0
    for j in track_iter_progress(list(range(len(dataset)))):
        input_dict = dataset.get_data_info(j)
        dataset.pre_pipeline(input_dict)
        example = dataset.pipeline(input_dict)
        annos = example["ann_info"]
        image_idx = example["sample_idx"]
        points = example["points"].tensor.numpy()
        gt_boxes_3d = annos["gt_bboxes_3d"].tensor.numpy()
        names = annos["gt_names"]
        group_dict = dict()
        if "group_ids" in annos:
            group_ids = annos["group_ids"]
        else:
            group_ids = np.arange(gt_boxes_3d.shape[0], dtype=np.int64)
        difficulty = np.zeros(gt_boxes_3d.shape[0], dtype=np.int32)
        if "difficulty" in annos:
            difficulty = annos["difficulty"]

        num_obj = gt_boxes_3d.shape[0]
        point_indices = box_np_ops.points_in_rbbox(points, gt_boxes_3d)

        if with_mask:
            # prepare masks
            gt_boxes = annos["gt_bboxes"]
            img_path = osp.split(example["img_info"]["filename"])[-1]
            if img_path not in file2id.keys():
                print(f"skip image {img_path} for empty mask")
                continue
            img_id = file2id[img_path]
            kins_annIds = coco.getAnnIds(imgIds=img_id)
            kins_raw_info = coco.loadAnns(kins_annIds)
            kins_ann_info = _parse_coco_ann_info(kins_raw_info)
            h, w = annos["img_shape"][:2]
            gt_masks = [_poly2mask(mask, h, w) for mask in kins_ann_info["masks"]]
            # get mask inds based on iou mapping
            bbox_iou = bbox_overlaps(kins_ann_info["bboxes"], gt_boxes)
            mask_inds = bbox_iou.argmax(axis=0)
            valid_inds = bbox_iou.max(axis=0) > 0.5

            # mask the image
            # use more precise crop when it is ready
            # object_img_patches = np.ascontiguousarray(
            #     np.stack(object_img_patches, axis=0).transpose(0, 3, 1, 2))
            # crop image patches using roi_align
            # object_img_patches = crop_image_patch_v2(
            #     torch.Tensor(gt_boxes),
            #     torch.Tensor(mask_inds).long(), object_img_patches)
            object_img_patches, object_masks = crop_image_patch(
                gt_boxes, gt_masks, mask_inds, annos["img"]
            )

        for i in range(num_obj):
            filename = f"{image_idx}_{names[i]}_{i}.bin"
            abs_filepath = osp.join(database_save_path, filename)
            rel_filepath = osp.join(f"{info_prefix}_gt_database", filename)

            # save point clouds and image patches for each object
            gt_points = points[point_indices[:, i]]
            gt_points[:, :3] -= gt_boxes_3d[i, :3]

            if with_mask:
                if object_masks[i].sum() == 0 or not valid_inds[i]:
                    # Skip object for empty or invalid mask
                    continue
                img_patch_path = abs_filepath + ".png"
                mask_patch_path = abs_filepath + ".mask.png"
                mmcv.imwrite(object_img_patches[i], img_patch_path)
                mmcv.imwrite(object_masks[i], mask_patch_path)

            with open(abs_filepath, "w") as f:
                gt_points.tofile(f)

            if (used_classes is None) or names[i] in used_classes:
                db_info = {
                    "name": names[i],
                    "path": rel_filepath,
                    "image_idx": image_idx,
                    "gt_idx": i,
                    "box3d_lidar": gt_boxes_3d[i],
                    "num_points_in_gt": gt_points.shape[0],
                    "difficulty": difficulty[i],
                }
                local_group_id = group_ids[i]
                # if local_group_id >= 0:
                if local_group_id not in group_dict:
                    group_dict[local_group_id] = group_counter
                    group_counter += 1
                db_info["group_id"] = group_dict[local_group_id]
                if "score" in annos:
                    db_info["score"] = annos["score"][i]
                if with_mask:
                    db_info.update({"box2d_camera": gt_boxes[i]})
                if names[i] in all_db_infos:
                    all_db_infos[names[i]].append(db_info)
                else:
                    all_db_infos[names[i]] = [db_info]

    for k, v in all_db_infos.items():
        print(f"load {len(v)} {k} database infos")

    with open(db_info_save_path, "wb") as f:
        pickle.dump(all_db_infos, f)

