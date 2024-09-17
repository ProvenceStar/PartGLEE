# Copyright (c) Facebook, Inc. and its affiliates.
import logging
import os
from detectron2.data import DatasetCatalog, MetadataCatalog
# from detectron2.data.datasets.register_coco import register_coco_instances
from detectron2.data.datasets.coco import load_coco_json

PASCAL_VOC_CATEGORIES = [
  {'id': 1, 'name': 'aeroplane'},
  {'id': 2, 'name': 'bicycle'},
  {'id': 3, 'name': 'bird'},
  {'id': 4, 'name': 'boat'},
  {'id': 5, 'name': 'bottle'},
  {'id': 6, 'name': 'bus'},
  {'id': 7, 'name': 'car'},
  {'id': 8, 'name': 'cat'},
  {'id': 9, 'name': 'chair'},
  {'id': 10, 'name': 'cow'},
  {'id': 11, 'name': 'diningtable'},
  {'id': 12, 'name': 'dog'},
  {'id': 13, 'name': 'horse'},
  {'id': 14, 'name': 'motorbike'},
  {'id': 15, 'name': 'person'},
  {'id': 16, 'name': 'pottedplant'},
  {'id': 17, 'name': 'sheep'},
  {'id': 18, 'name': 'sofa'},
  {'id': 19, 'name': 'train'},
  {'id': 20, 'name': 'tvmonitor'},
]

def _get_partimagenet_metadata():
    id_to_name = {x['id']: x['name'] for x in PASCAL_VOC_CATEGORIES}
    thing_dataset_id_to_contiguous_id = {
        x: i for i, x in enumerate(sorted(id_to_name))}
    thing_classes = [id_to_name[k] for k in sorted(id_to_name)]
    return {
        "thing_dataset_id_to_contiguous_id": thing_dataset_id_to_contiguous_id,
        "thing_classes": thing_classes}


def register_pascal_voc_instances(name, metadata, json_file, image_root):
    DatasetCatalog.register(name, lambda: load_coco_json(
        json_file, image_root, name, dataset_name_in_dict='voc'))
    MetadataCatalog.get(name).set(
        json_file=json_file, image_root=image_root,
        evaluator_type="coco", **metadata
    )

_PASCAL_VOC = {
    # "pascal_voc_train": ("pascal_part/VOCdevkit/VOC2010/JPEGImages", "pascal_part/voc2010_train.json"),
    # "pascal_voc_val": ("pascal_part/VOCdevkit/VOC2010/JPEGImages", "pascal_part/voc2010_val.json"),
    "pascal_voc_train_segm": ("pascal_part/VOCdevkit/VOC2010/JPEGImages", "pascal_part/voc2010_train_segm.json"), # Original Pascal-VOC-2010, which contains more images than Pascal-Part
    "pascal_voc_val_segm": ("pascal_part/VOCdevkit/VOC2010/JPEGImages", "pascal_part/voc2010_val_segm.json"),
    # "pascal_voc_train_object": ("pascal_part/VOCdevkit/VOC2010/JPEGImages", "pascal_part/voc2010_train_object.json"),
    # "pascal_voc_val_object": ("pascal_part/VOCdevkit/VOC2010/JPEGImages", "pascal_part/voc2010_val_object.json"),
    # "pascal_part_object_train": ("pascal_part/VOCdevkit/VOC2010/JPEGImages", "pascal_part/train_object_only.json"),
    # "pascal_part_object_val": ("pascal_part/VOCdevkit/VOC2010/JPEGImages", "pascal_part/val_object_only.json"),
}

def register_all_pascal_voc(root):
    for key, (image_root, json_file) in _PASCAL_VOC.items():
        register_pascal_voc_instances(
            key,
            _get_partimagenet_metadata(),
            os.path.join(root, json_file) if "://" not in json_file else json_file,
            os.path.join(root, image_root),
        )

# _root = os.getenv("DETECTRON2_DATASETS", "datasets")
# register_all_pascal_voc(_root)