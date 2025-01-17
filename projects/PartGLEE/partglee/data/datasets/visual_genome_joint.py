# Copyright (c) Facebook, Inc. and its affiliates.
import contextlib
import datetime
import io
import json
import logging
import numpy as np
import os
import shutil
import pycocotools.mask as mask_util
from fvcore.common.timer import Timer
from iopath.common.file_io import file_lock
from PIL import Image

from detectron2.structures import Boxes, BoxMode, PolygonMasks, RotatedBoxes
from detectron2.utils.file_io import PathManager

from detectron2.data import DatasetCatalog, MetadataCatalog
from lvis import LVIS

"""
This file contains functions to parse COCO-format annotations into dicts in "Detectron2 format".
"""
logger = logging.getLogger(__name__)

__all__ = ["register_vg_joint_instances"]


_PREDEFINED_SPLITS_VG_JOINT = {
    # mixed
    "vg_train_joint": ("visual_genome/images", "visual_genome/annotations/train_from_objects.json"),
    "vg_captiontrain_joint": ("visual_genome/images", "visual_genome/annotations/train.json"),
}

with open("datasets/visual_genome/vg_object_category.txt", 'r') as file:
    vg_object_categories = file.readlines()
    VG_OBJECT_CATEGORIES = [line.strip() for line in vg_object_categories]

with open("datasets/visual_genome/vg_part_category.txt", 'r') as file:
    vg_part_categories = file.readlines()
    VG_PART_CATEGORIES = [line.strip() for line in vg_part_categories]

def _get_vg_meta():
    categories = [{'supercategory': 'None', 'id': 1, 'name': 'None'}]
    vg_categories = sorted(categories, key=lambda x: x["id"])
    thing_classes = [k["name"] for k in vg_categories]
    meta = {"thing_classes": thing_classes}
    return meta

def register_all_vg_joint(root):
    for key, (image_root, json_file) in _PREDEFINED_SPLITS_VG_JOINT.items():
        # Assume pre-defined datasets live in `./datasets`.
        register_vg_joint_instances(
            key,
            _get_vg_meta(),
            os.path.join(root, json_file) if "://" not in json_file else json_file,
            os.path.join(root, image_root),
            dataset_name_in_dict="vg_joint"
        )

def register_vg_joint_instances(name, metadata, json_file, image_root, dataset_name_in_dict=None):
    """
    """
    DatasetCatalog.register(name, lambda: load_vg_json(
        json_file, image_root, dataset_name = dataset_name_in_dict))
    MetadataCatalog.get(name).set(
        json_file=json_file, image_root=image_root,
        evaluator_type="vg", **metadata
    )

def load_vg_json(json_file, image_root, dataset_name=None, prompt=None):

    json_file = PathManager.get_local_path(json_file)

    timer = Timer()
    lvis_api = LVIS(json_file)
    if timer.seconds() > 1:
        logger.info("Loading {} takes {:.2f} seconds.".format(
            json_file, timer.seconds()))
    img_ids = sorted(lvis_api.imgs.keys())
    imgs = lvis_api.load_imgs(img_ids)
    anns = [lvis_api.img_ann_map[img_id] for img_id in img_ids]

    ann_ids = [ann["id"] for anns_per_image in anns for ann in anns_per_image]
    assert len(set(ann_ids)) == len(ann_ids), \
        "Annotation ids in '{}' are not unique".format(json_file)

    imgs_anns = list(zip(imgs, anns))
    logger.info("Loaded {} images in the LVIS v1 format from {}".format(
        len(imgs_anns), json_file))

    dataset_dicts = []

    for (img_dict, anno_dict_list) in imgs_anns:
        record = {}
        if "file_name" in img_dict:
            file_name = img_dict["file_name"]
            record["file_name"] = os.path.join(image_root, file_name)

        record["height"] = int(img_dict["height"])
        record["width"] = int(img_dict["width"])
        image_id = record["image_id"] = img_dict["id"]

        objs = []
        for anno in anno_dict_list:
            assert anno["image_id"] == image_id
            if anno.get('iscrowd', 0) > 0:
                continue
            obj = {"bbox": anno["bbox"], "bbox_mode": BoxMode.XYWH_ABS}
            obj["category_id"] = 0
            if "caption_with_token" in anno.keys():
                obj["object_description"] = anno["caption_with_token"]
            elif "object_name" in anno.keys():
                obj["object_description"] = anno["object_name"]
            else:
                obj["object_description"] = anno["caption"]
            objs.append(obj)
        record["annotations"] = objs
        if len(record["annotations"]) == 0:
            continue
        record["task"] = dataset_name
        record["dataset_name"] = dataset_name
        dataset_dicts.append(record)
    return dataset_dicts