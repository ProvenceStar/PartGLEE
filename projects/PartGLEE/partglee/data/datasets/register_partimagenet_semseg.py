# Copyright (c) Facebook, Inc. and its affiliates.
import json
import logging
import os

from detectron2.data import DatasetCatalog, MetadataCatalog
from detectron2.utils.file_io import PathManager

try:
    import cv2  # noqa
except ImportError:
    # OpenCV is an optional dependency at the moment
    pass


logger = logging.getLogger(__name__)


PARTIMAGENET_PART_CATEGORIES = ['Quadruped-Head', 'Quadruped-Torso', 'Quadruped-Foot', 'Quadruped-Tail', 'Biped-Head', 'Biped-Torso', 'Biped-Head', 'Biped-Foot', 'Biped-Tail',
                                'Fish-Head', 'Fish-Torso', 'Fish-Fin', 'Fish-Tail', 'Bird-Head', 'Bird-Torso', 'Bird-Wing', 'Bird-Foot', 'Bird-Tail', 'Snake-Head', 'Snake-Torso',
                                'Reptile-Head', 'Reptile-Torso', 'Reptile-Foot', 'Reptile-Tail', 'Car-Body', 'Car-Tire', 'Car-Side_Mirror', 'Bicycle-Head', 'Bicycle-Body',
                                'Bicycle-Seat', 'Bicycle-Tire', 'Boat-Body', 'Boat-Sail', 'Aeroplane-Head', 'Aeroplane-Body', 'Aeroplane-Wing', 'Aeroplane-Enging', 'Aeroplane-Tail',
                                'Bottle-Body', 'Bottle-Mouth', 'Background']
PARTIMAGENET_OBJECT_CATEGORIES = [str(i) for i in range(159)]

_PREDEFINED_SPLITS_PARTIMAGENET_SEMANTIC = {
    "partimagenet_semseg_train": (
        "PartImageNetSemSeg/images/train",
        "PartImageNetSemSeg/annotations/train",
        "PartImageNetSemSeg/annotations/train_whole",
    ),
    "partimagenet_semseg_val": (
        "PartImageNetSemSeg/images/val/",
        "PartImageNetSemSeg/annotations/val",
        "PartImageNetSemSeg/annotations/val_whole",
    ),
    "partimagenet_semseg_test": (
        "PartImageNetSemSeg/images/test",
        "PartImageNetSemSeg/annotations/test",
        "PartImageNetSemSeg/annotations/test_whole",
    )
}


def get_metadata():
    meta = {}
    # The following metadata maps contiguous id from [0, #thing categories +
    # #stuff categories) to their names and colors. We have to replica of the
    # same name and color under "thing_*" and "stuff_*" because the current
    # visualization function in D2 handles thing and class classes differently
    # due to some heuristic used in Panoptic FPN. We keep the same naming to
    # enable reusing existing visualization functions.
    part_classes = PARTIMAGENET_PART_CATEGORIES
    object_classes = PARTIMAGENET_OBJECT_CATEGORIES

    meta["part_classes"] = part_classes
    meta["object_classes"] = object_classes

    return meta


def _get_partimagenet_files(image_dir, gt_part_dir, gt_object_dir):
    files = []
    # scan through the directory
    # need to generate a json for each image to make it compatible

    for basename in PathManager.ls(image_dir):
        image_file = os.path.join(image_dir, basename)
        suffix = ".JPEG"
        assert basename.endswith(suffix), basename
        basename = basename[: -len(suffix)]

        label_part_file = os.path.join(gt_part_dir, basename + ".png")
        label_object_file = os.path.join(gt_object_dir, basename + ".png")
        json_file = os.path.join(gt_part_dir, basename + ".json")

        files.append((image_file, label_part_file, label_object_file, json_file))

    assert len(files), "No images found in {}".format(image_dir)
    for f in files[0]:
        assert PathManager.isfile(f), f
    return files


def load_semantic_gt(image_dir, gt_part_dir, gt_object_dir):
    """
    Args:
        image_dir (str): path to the raw dataset.
        gt_dir (str): path to the raw annotations.

    Returns:
        list[dict]: a list of dict, each has "file_name" and
            "sem_seg_file_name".
    """
    ret = []
    # gt_dir is small and contain many small files. make sense to fetch to local first
    gt_part_dir = PathManager.get_local_path(gt_part_dir)
    gt_object_dir = PathManager.get_local_path(gt_object_dir)
    for image_file, label_part_file, label_object_file, json_file in _get_partimagenet_files(image_dir, gt_part_dir, gt_object_dir):
        with PathManager.open(json_file, "r") as f:
            jsonobj = json.load(f)
        ret.append(
            {
                "file_name": image_file,
                "sem_seg_part_file_name": label_part_file,
                "sem_seg_object_file_name": label_object_file,
                "height": jsonobj["imgHeight"],
                "width": jsonobj["imgWidth"],
                "dataset_name": "partimagenet_semseg",
            }
        )
    assert len(ret), f"No images found in {image_dir}!"
    return ret


def register_partimagenet_annos_sem_seg(name, metadata, image_root, semantic_part_root, semantic_object_root):
    semantic_name = name

    MetadataCatalog.get(semantic_name).set(
        part_classes=metadata["part_classes"],
        object_classes=metadata["object_classes"],
    )

    DatasetCatalog.register(
        semantic_name,
        lambda: load_semantic_gt(image_root, semantic_part_root, semantic_object_root),
    )
    MetadataCatalog.get(semantic_name).set(
        image_root=image_root,
        semantic_part_root=semantic_part_root,
        semantic_object_root=semantic_object_root,
        evaluator_type="hierarchical_seg",
        ignore_part_label=255,
        ignore_object_label=255,
        **metadata,
    )


def register_all_partimagenet_annos_sem_seg(root):
    for (
        prefix,
        (semantic_root, semantic_part_gt, semantic_object_gt),
    ) in _PREDEFINED_SPLITS_PARTIMAGENET_SEMANTIC.items():
        register_partimagenet_annos_sem_seg(
            prefix,
            get_metadata(),
            os.path.join(root, semantic_root),
            os.path.join(root, semantic_part_gt),
            os.path.join(root, semantic_object_gt),
        )


# _root = os.getenv("DETECTRON2_DATASETS", "datasets")
# register_all_partimagenet_annos_sem_seg(_root)