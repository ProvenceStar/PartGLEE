import json
import logging
import os
from detectron2.utils.file_io import PathManager
from detectron2.data.datasets.coco import load_sem_seg

logger = logging.getLogger(__name__)

def load_pascalvoc_sem_seg(gt_root, image_root, gt_ext="png", image_ext="jpg", dataset_name_in_dict='pascalvoc'):
    """
    Load semantic segmentation Pascal-Part-116. All files under "gt_root" with "gt_ext" extension are
    treated as ground truth annotations and all files under "image_root" with "image_ext" extension
    as input images. Ground truth and input images are matched using file paths relative to
    "gt_root" and "image_root" respectively without taking into account file extensions.
    This works for COCO as well as some other datasets.

    Args:
        gt_root (str): full path to ground truth semantic segmentation files. Semantic segmentation
            annotations are stored as images with integer values in pixels that represent
            corresponding semantic labels.
        image_root (str): the directory where the input images are.
        gt_ext (str): file extension for ground truth annotations.
        image_ext (str): file extension for input images.

    Returns:
        list[dict]:
            a list of dicts in detectron2 standard format without instance-level
            annotation.

    Notes:
        1. This function does not read the image and ground truth files.
           The results do not have the "image" and "sem_seg" fields.
    """

    # We match input images with ground truth based on their relative filepaths (without file
    # extensions) starting from 'image_root' and 'gt_root' respectively.
    def file2id(folder_path, file_path):
        # extract relative path starting from `folder_path`
        image_id = os.path.normpath(os.path.relpath(file_path, start=folder_path))
        # remove file extension
        image_id = os.path.splitext(image_id)[0]
        return image_id

    input_files = sorted(
        (os.path.join(image_root, f) for f in PathManager.ls(image_root) if f.endswith(image_ext)),
        key=lambda file_path: file2id(image_root, file_path),
    )
    gt_files = sorted(
        (os.path.join(gt_root, f) for f in PathManager.ls(gt_root) if f.endswith(gt_ext)),
        key=lambda file_path: file2id(gt_root, file_path),
    )

    assert len(gt_files) > 0, "No annotations found in {}.".format(gt_root)

    # Use the intersection, so that val2017_100 annotations can run smoothly with val2017 images
    if len(input_files) != len(gt_files):
        logger.warn(
            "Directory {} and {} has {} and {} files, respectively.".format(
                image_root, gt_root, len(input_files), len(gt_files)
            )
        )
        input_basenames = [os.path.basename(f)[: -len(image_ext)] for f in input_files]
        gt_basenames = [os.path.basename(f)[: -len(gt_ext)] for f in gt_files]
        intersect = list(set(input_basenames) & set(gt_basenames))
        # sort, otherwise each worker may obtain a list[dict] in different order
        intersect = sorted(intersect)
        logger.warn("Will use their intersection of {} files.".format(len(intersect)))
        input_files = [os.path.join(image_root, f + image_ext) for f in intersect]
        gt_files = [os.path.join(gt_root, f + gt_ext) for f in intersect]

    logger.info(
        "Loaded {} images with semantic segmentation from {}".format(len(input_files), image_root)
    )

    dataset_dicts = []
    for (img_path, gt_path) in zip(input_files, gt_files):
        record = {}
        record["file_name"] = img_path
        record["sem_seg_file_name"] = gt_path
        record["dataset_name"] = dataset_name_in_dict
        dataset_dicts.append(record)

    return dataset_dicts

def load_obj_part_sem_seg(gt_root, image_root, gt_ext="png", image_ext="jpg", data_list=None):
    data_dicts = load_pascalvoc_sem_seg(gt_root, image_root, gt_ext, image_ext)
    if data_list is not None:
        img_list = json.load(open(data_list,'r'))
        img_list = [item["file_name"] for item in img_list]
    new_data_dicts = []
    for i,data in enumerate(data_dicts):
        if data_list is not None:
            if data["file_name"] not in img_list:
                continue
        data_dicts[i]["obj_sem_seg_file_name"] = data["sem_seg_file_name"].replace('part','obj')
        new_data_dicts.append(data_dicts[i])
    return new_data_dicts

def load_pascal_voc_obj_part_sem_seg(gt_root, image_root, gt_ext="png", image_ext="jpg", data_list=None):
    data_dicts = load_pascalvoc_sem_seg(gt_root, image_root, gt_ext, image_ext)
    if data_list is not None:
        img_list = json.load(open(data_list,'r'))
        img_list = [item["file_name"] for item in img_list]
    new_data_dicts = []
    for i,data in enumerate(data_dicts):
        if data_list is not None:
            if data["file_name"] not in img_list:
                continue
        data_dicts[i]["obj_sem_seg_file_name"] = data["sem_seg_file_name"].replace('part','obj')
        new_data_dicts.append(data_dicts[i])
    return new_data_dicts

def load_binary_mask(gt_root, image_root, gt_ext="png", image_ext="jpg", label_count="_part_label_count.json", base_classes=None):
    """
    Flatten the results of `load_sem_seg` to annotations for binary mask.

    `label_count_file` contains a dictionary like:
    ```
    {
        "xxx.png":[0,3,5],
        "xxxx.png":[3,4,7],
    }
    ```
    """
    label_count_file = gt_root + label_count
    with open(label_count_file) as f:
        label_count_dict = json.load(f)

    data_dicts = load_pascalvoc_sem_seg(gt_root, image_root, gt_ext, image_ext)
    flattened_data_dicts = []
    for data in data_dicts:
        data['obj_sem_seg_file_name'] = data["sem_seg_file_name"].replace('_part','_obj')
        category_per_image = label_count_dict[
            os.path.basename(data["sem_seg_file_name"])
        ]
        if base_classes is not None:
            category_per_image = [i for i in category_per_image if i in base_classes]
        flattened_data = [
            dict(**{"category_id": cat}, **data) for cat in category_per_image
        ]
        flattened_data_dicts.extend(flattened_data)
    logger.info(
        "Loaded {} images with flattened semantic segmentation from {}".format(
            len(flattened_data_dicts), image_root
        )
    )
    return flattened_data_dicts
