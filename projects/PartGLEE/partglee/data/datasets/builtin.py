
import os

from .refcoco import (
    register_refcoco,
    _get_refcoco_meta,
)
from .sa1b import (_get_sa1b_meta,register_sa1b)
from .uvo_image import (_get_uvo_image_meta, register_UVO_image)
from .uvo_video import (_get_uvo_dense_video_meta, register_UVO_dense_video)
from .burst_video import (_get_burst_video_meta, register_burst_video, _get_burst_image_meta)
from .tao import _get_tao_image_meta
# from .flicker import register_flicker, _get_flicker_meta
from detectron2.data.datasets.register_coco import register_coco_instances
from .open_image import _get_builtin_metadata_openimage
from .objects365_v2 import _get_builtin_metadata
from .objects365 import _get_builtin_metadata_obj365v1
from .vis import (
    register_ytvis_instances,
    _get_ytvis_2019_instances_meta,
    _get_ytvis_2021_instances_meta,
    _get_ovis_instances_meta,
    _get_ytvis19_image_meta,
    _get_ytvis21_image_meta,
    _get_ovis_image_meta,
    _get_lvvis_instances_meta
    )
from .odinw import _get_odinw_image_meta
from .rvos import (
    register_rytvis_instances,
    )
from .bdd100k import (
    _get_bdd_obj_det_meta,
    _get_bdd_inst_seg_meta,
    _get_bdd_obj_track_meta
)
from .VisualGenome import register_vg_instances, _get_vg_meta
from .paco import register_all_paco
from .pascal_joint import register_all_pascal_joint
# from .pascal_part import register_all_pascal_part, register_all_pascal_part_open_vocabulary
from .partimagenet import register_all_partimagenet
from .register_coco_panoptic_annos_semseg import register_all_coco_panoptic_annos_sem_seg
from .register_ade20k_joint import register_all_ade20k_joint
from .register_pascalvoc_joint import register_all_pascalvoc_joint
from .register_partimagenet_joint import register_all_partimagenet_joint
from .register_pascal_part_open_vocabulary import register_all_pascal_part_open_vocabulary
# from .visual_genome_joint import register_all_vg_joint
from .register_ade_part_234 import register_ade20k_part_234
from .register_ade20k_base import register_all_ade20k_base
from .register_pascal_part_116 import register_pascal_part_116
from .register_pascalvoc_base import register_all_pascalvoc_base
from .pascal_voc_2010 import register_all_pascal_voc
from .register_seginw_instance import register_all_seginw
from .register_partimagenet_semseg import register_all_partimagenet_annos_sem_seg

# ==== Predefined splits for REFCOCO datasets ===========
_PREDEFINED_SPLITS_REFCOCO = {
    # refcoco
    "refcoco-unc-train": ("/opt/tiger/NAS/jiangyi.enjoy/dataset/coco/train2014", "annotations/refcoco-unc/instances_train.json"),
    "refcoco-unc-val": ("/opt/tiger/NAS/jiangyi.enjoy/dataset/coco/train2014", "annotations/refcoco-unc/instances_val.json"),
    "refcoco-unc-testA": ("/opt/tiger/NAS/jiangyi.enjoy/dataset/coco/train2014", "annotations/refcoco-unc/instances_testA.json"),
    "refcoco-unc-testB": ("/opt/tiger/NAS/jiangyi.enjoy/dataset/coco/train2014", "annotations/refcoco-unc/instances_testB.json"),
    # refcocog
    "refcocog-umd-train": ("/opt/tiger/NAS/jiangyi.enjoy/dataset/coco/train2014", "annotations/refcocog-umd/instances_train.json"),
    "refcocog-umd-val": ("/opt/tiger/NAS/jiangyi.enjoy/dataset/coco/train2014", "annotations/refcocog-umd/instances_val.json"),
    "refcocog-umd-test": ("/opt/tiger/NAS/jiangyi.enjoy/dataset/coco/train2014", "annotations/refcocog-umd/instances_test.json"),
    # "refcocog-google-val": ("/opt/tiger/NAS/jiangyi.enjoy/dataset/coco/train2014", "annotations/refcocog-google/instances_val.json"),
    # refcoco+
    "refcocoplus-unc-train": ("/opt/tiger/NAS/jiangyi.enjoy/dataset/coco/train2014", "annotations/refcocoplus-unc/instances_train.json"),
    "refcocoplus-unc-val": ("/opt/tiger/NAS/jiangyi.enjoy/dataset/coco/train2014", "annotations/refcocoplus-unc/instances_val.json"),
    "refcocoplus-unc-testA": ("/opt/tiger/NAS/jiangyi.enjoy/dataset/coco/train2014", "annotations/refcocoplus-unc/instances_testA.json"),
    "refcocoplus-unc-testB": ("/opt/tiger/NAS/jiangyi.enjoy/dataset/coco/train2014", "annotations/refcocoplus-unc/instances_testB.json"),
    # mixed
    "refcoco-mixed": ("/opt/tiger/NAS/jiangyi.enjoy/dataset/coco/train2014", "annotations/refcoco-mixed/instances_train.json"),
    "refcoco-mixed-filter": ("/opt/tiger/NAS/jiangyi.enjoy/dataset/coco/train2014", "annotations/refcoco-mixed/instances_train_filter.json"),
    # ref_VOS_image_level
    "refytb-imagelevel": ("ref-youtube-vos/train/JPEGImages", "/opt/tiger/NAS/junfengwu/datasets/custom_annotations/RVOS_refcocofmt.json"),
}


def register_all_refcoco(root):
    for key, (image_root, json_file) in _PREDEFINED_SPLITS_REFCOCO.items():
        # Assume pre-defined datasets live in `./datasets`.
        register_refcoco(
            key,
            _get_refcoco_meta(),
            os.path.join(root, json_file) if "://" not in json_file else json_file,
            os.path.join(root, image_root),
        )
 


# ==== Predefined splits for VisualGenome datasets ===========
_PREDEFINED_SPLITS_VG = {
    # mixed
    "vg_train": ("visual_genome/images", "visual_genome/annotations/train_from_objects.json"),
    "vg_captiontrain": ("visual_genome/images", "visual_genome/annotations/train.json"),
}


def register_all_vg(root):
    for key, (image_root, json_file) in _PREDEFINED_SPLITS_VG.items():
        # Assume pre-defined datasets live in `./datasets`.
        register_vg_instances(
            key,
            _get_vg_meta(),
            os.path.join(root, json_file) if "://" not in json_file else json_file,
            os.path.join(root, image_root),
            dataset_name_in_dict="vg"
        )

###  GRIT 20M

# ==== Predefined splits for VisualGenome datasets ===========
_PREDEFINED_SPLITS_GRIT20M = {
    "grit_30w": ("/opt/tiger/NAS/jiangyi.enjoy/dataset/grit-20m/images/", "/opt/tiger/NAS/junfengwu/datasets/GRIT20M/grit_30w.json"),
    "grit_5m": ("/opt/tiger/NAS/jiangyi.enjoy/dataset/grit-20m/images/", "/opt/tiger/NAS/junfengwu/datasets/GRIT20M/grit_5m.json"),
}


def register_all_grit(root):
    for key, (image_root, json_file) in _PREDEFINED_SPLITS_GRIT20M.items():
        # Assume pre-defined datasets live in `./datasets`.
        register_vg_instances(
            key,
            _get_vg_meta(),
            os.path.join(root, json_file) if "://" not in json_file else json_file,
            os.path.join(root, image_root),
            dataset_name_in_dict="grit"
        )


_PREDEFINED_SPLITS_SA1B = {
    # SA-1B
    # "sa1b_32_train": ("/opt/tiger/NAS/junfengwu/datasets/SA1B_64tar/images/", "/opt/tiger/NAS/junfengwu/datasets/SA1B_64tar/sa1b_32_train.json"),
    # "sa1b_64_train": ("/opt/tiger/NAS/junfengwu/datasets/SA1B_64tar/images/", "/opt/tiger/NAS/junfengwu/datasets/SA1B_64tar/sa1b_64_train.json"),
    "sa1b_joint": ("datasets/sa-1b/sa_000000/", "datasets/sa-1b/sa_000000_joint.json"),
    "sa1b_2w": ("/opt/tiger/NAS/junfengwu/datasets/SA1B_scaleup/images/", "/opt/tiger/NAS/junfengwu/datasets/SA1B_scaleup/sa1b_2w.json"),
    "sa1b_1m": ("/opt/tiger/NAS/junfengwu/datasets/SA1B_scaleup/images/", "/opt/tiger/NAS/junfengwu/datasets/SA1B_scaleup/sa1b_1m.json"),
    "sa1b_2m": ("/opt/tiger/NAS/junfengwu/datasets/SA1B_scaleup/images/", "/opt/tiger/NAS/junfengwu/datasets/SA1B_scaleup/sa1b_2m.json"),
}

def register_all_sa1b(root):
    for key, (image_root, json_file) in _PREDEFINED_SPLITS_SA1B.items():
        # Assume pre-defined datasets live in `./datasets`.
        if 'joint' in key:
            register_sa1b(
                key,
                _get_sa1b_meta(),
                json_file,
                image_root,
                has_mask = False,
                use_part=True,
            )
        else:
            register_sa1b(
                key,
                _get_sa1b_meta(),
                json_file,
                image_root,
                has_mask = False,
            )

_PREDEFINED_SPLITS_burst_image = {
    # BURST-image
    "image_bur": ("TAO/frames/val/", "TAO/burst_annotations/TAO_val_lvisformat.json"),
}

 
def register_all_BURST_image(root):
    for key, (image_root, json_file) in _PREDEFINED_SPLITS_burst_image.items():
        # Assume pre-defined datasets live in `./datasets`.
        register_coco_instances(
            key,
            _get_burst_image_meta(),
            os.path.join(root, json_file) if "://" not in json_file else json_file,
            os.path.join(root, image_root),
            dataset_name_in_dict="image_bur",
            evaluator_type = 'lvis'
        )




_PREDEFINED_SPLITS_TAO_image = {
    # TAO-image
    "image_tao": ("TAO/frames/", "TAO/annotations-1.2/TAO_val_withlabel_lvisformat.json"),
}

 
def register_all_TAO_image(root):
    for key, (image_root, json_file) in _PREDEFINED_SPLITS_TAO_image.items():
        # Assume pre-defined datasets live in `./datasets`.
        register_coco_instances(
            key,
            _get_tao_image_meta(),
            os.path.join(root, json_file) if "://" not in json_file else json_file,
            os.path.join(root, image_root),
            dataset_name_in_dict="image_tao",
            evaluator_type = 'lvis'
        )

_PREDEFINED_SPLITS_VIS_image = {
    # ytvis-image
    "image_yt19": ("ytvis_2019/train/JPEGImages", "ytvis_2019/annotations/ytvis19_cocofmt.json"),
}

 
def register_all_YTVIS19_image(root):
    for key, (image_root, json_file) in _PREDEFINED_SPLITS_VIS_image.items():
        # Assume pre-defined datasets live in `./datasets`.
        register_coco_instances(
            key,
            _get_ytvis19_image_meta(),
            os.path.join(root, json_file) if "://" not in json_file else json_file,
            os.path.join(root, image_root),
            dataset_name_in_dict="image_yt19"
        )

_PREDEFINED_SPLITS_VIS21_image = {
    # ytvis-image
    "image_yt21": ("ytvis_2021/train/JPEGImages", "ytvis_2021/annotations/ytvis21_cocofmt.json"),
}
 
def register_all_YTVIS21_image(root):
    for key, (image_root, json_file) in _PREDEFINED_SPLITS_VIS21_image.items():
        # Assume pre-defined datasets live in `./datasets`.
        register_coco_instances(
            key,
            _get_ytvis21_image_meta(),
            os.path.join(root, json_file) if "://" not in json_file else json_file,
            os.path.join(root, image_root),
            dataset_name_in_dict="image_yt21"
        )

_PREDEFINED_SPLITS_OVIS_image = {
    # ytvis-image
    "image_o": ("ovis/train", "ovis/ovis_cocofmt.json"),
}
 
def register_all_OVIS_image(root):
    for key, (image_root, json_file) in _PREDEFINED_SPLITS_OVIS_image.items():
        # Assume pre-defined datasets live in `./datasets`.
        register_coco_instances(
            key,
            _get_ovis_image_meta(),
            os.path.join(root, json_file) if "://" not in json_file else json_file,
            os.path.join(root, image_root),
            dataset_name_in_dict="image_o"
        )





_PREDEFINED_SPLITS_UVO_image = {
    # UVO-image
    "UVO_frame_train": ("/opt/tiger/NAS/junfengwu/datasets/UVO/uvo_videos_frames", "/opt/tiger/NAS/junfengwu/datasets/custom_annotations/UVO/annotations/FrameSet/UVO_frame_train_onecate.json"),
    "UVO_frame_val": ("/opt/tiger/NAS/junfengwu/datasets/UVO/uvo_videos_frames", "/opt/tiger/NAS/junfengwu/datasets/custom_annotations/UVO/annotations/FrameSet/UVO_frame_val_onecate.json"),
    "UVO_frame_val_noncoco": ("/opt/tiger/NAS/junfengwu/datasets/UVO/uvo_videos_frames", "/opt/tiger/NAS/junfengwu/datasets/custom_annotations/UVO/annotations/FrameSet/UVO_frame_val_noncoco_onecate.json"),
}


def register_all_UVO_image(root):
    for key, (image_root, json_file) in _PREDEFINED_SPLITS_UVO_image.items():
        # Assume pre-defined datasets live in `./datasets`.
        register_UVO_image(
            key,
            _get_uvo_image_meta(),
            os.path.join(root, json_file) if "://" not in json_file else json_file,
            os.path.join(root, image_root),
            dataset_name_in_dict = 'UVO_image'
        )
 

_PREDEFINED_SPLITS_UVO_dense_video = {
    # UVO-dense-video_with category
    "UVO_dense_video_train": ("/opt/tiger/NAS/junfengwu/datasets/UVO/uvo_videos_dense_frames_jpg", "/opt/tiger/NAS/junfengwu/datasets/UVO/annotations/VideoDenseSet/UVO_video_train_dense_objectlabel.json"),
    "UVO_dense_video_val": ("/opt/tiger/NAS/junfengwu/datasets/UVO/uvo_videos_dense_frames_jpg", "/opt/tiger/NAS/junfengwu/datasets/UVO/annotations/VideoDenseSet/UVO_video_val_dense_objectlabel.json"),
}


def register_all_UVO_dense_video(root):
    for key, (image_root, json_file) in _PREDEFINED_SPLITS_UVO_dense_video.items():
        # Assume pre-defined datasets live in `./datasets`.
        register_UVO_dense_video(
            key,
            _get_uvo_dense_video_meta(),
            os.path.join(root, json_file) if "://" not in json_file else json_file,
            os.path.join(root, image_root),
            dataset_name_in_dict = 'uvo_video'
        )



_PREDEFINED_SPLITS_BURST_video = {
    # tao-video_without category  BURST benchmark
    "BURST_video_train": ("TAO/frames/train/", "TAO/burst_annotations/TAO_train_withlabel_ytvisformat.json"),
    "BURST_video_val": ("TAO/frames/val/", "TAO/burst_annotations/TAO_val_withlabel_ytvisformat.json"),
}


def register_all_BURST_video(root):
    for key, (image_root, json_file) in _PREDEFINED_SPLITS_BURST_video.items():
        # Assume pre-defined datasets live in `./datasets`.
        register_burst_video(
            key,
            _get_burst_video_meta(),
            os.path.join(root, json_file) if "://" not in json_file else json_file,
            os.path.join(root, image_root),
            dataset_name_in_dict = 'burst'
        )




_PREDEFINED_SPLITS_TAO_video = {
    # tao-video_without category  BURST benchmark
    # "BURST_video_train": ("TAO/frames/train/", "TAO/burst_annotations/TAO_train_withlabel_ytvisformat.json"),
    "TAO_video_val": ("TAO/frames/", "TAO/TAO_annotations/validation_ytvisfmt.json"),
}


def register_all_TAO_video(root):
    for key, (image_root, json_file) in _PREDEFINED_SPLITS_TAO_video.items():
        # Assume pre-defined datasets live in `./datasets`.
        register_ytvis_instances(
            key,
            _get_tao_image_meta(),
            os.path.join(root, json_file) if "://" not in json_file else json_file,
            os.path.join(root, image_root),
            dataset_name_in_dict = 'tao_video'
        )



_PREDEFINED_SPLITS_OPEN_IMAGE = {
    "openimage_train": ("/opt/tiger/NAS/jiangyi.enjoy/dataset/openimages/detection/", "/opt/tiger/NAS/junfengwu/datasets/open_image/openimages_v6_train_bbox_splitdir.json"),
    "openimage_val": ("/opt/tiger/NAS/jiangyi.enjoy/dataset/openimages/detection/", "/opt/tiger/NAS/junfengwu/datasets/open_image/openimages_v6_val_bbox_splitdir.json"),
    "openimage_joint_train": ("openimages/openimages_sub/", "openimages/openimages_subset.json"),
}


def register_all_openimage(root):
    for key, (image_root, json_file) in _PREDEFINED_SPLITS_OPEN_IMAGE.items():
        if 'joint' in key:
            register_coco_instances(
                key,
                _get_builtin_metadata_openimage(),
                os.path.join(root, json_file) if "://" not in json_file else json_file,
                os.path.join(root, image_root),
                dataset_name_in_dict="openimage_joint"
            )
        else:
            register_coco_instances(
                key,
                _get_builtin_metadata_openimage(),
                os.path.join(root, json_file) if "://" not in json_file else json_file,
                os.path.join(root, image_root),
                dataset_name_in_dict="openimage"
            )




_PREDEFINED_SPLITS_OBJECTS365V2 = {
     "objects365_v2_train": ("/opt/tiger/NAS/junfengwu/datasets/Objects365V2/images/", "/opt/tiger/NAS/junfengwu/datasets/Objects365V2/annotations/zhiyuan_objv2_train_new.json"),
     "objects365_v2_masktrain": ("/opt/tiger/NAS/junfengwu/datasets/Objects365V2/images/", "/opt/tiger/NAS/junfengwu/datasets/Objects365V2/annotations/objects365_v2_train_with_mask.json"),
    "objects365_v2_val": ("/opt/tiger/NAS/junfengwu/datasets/Objects365V2/images/", "/opt/tiger/NAS/junfengwu/datasets/Objects365V2/annotations/zhiyuan_objv2_val_new.json"),
}

# _PREDEFINED_SPLITS_OBJECTS365V2 = {
#     "objects365_v2_train": ("/opt/tiger/Objects365V2/images/", "/opt/tiger/Objects365V2/annotations/zhiyuan_objv2_train_new.json"),
#     "objects365_v2_masktrain": ("/opt/tiger/Objects365V2/images/", "/opt/tiger/Objects365V2/annotations/objects365_v2_train_with_mask.json"),
#     "objects365_v2_val": ("/opt/tiger/Objects365V2/images/", "/opt/tiger/Objects365V2/annotations/zhiyuan_objv2_val_new.json"),
# }

def register_all_obj365v2(root):
    for key, (image_root, json_file) in _PREDEFINED_SPLITS_OBJECTS365V2.items():
        register_coco_instances(
            key,
            _get_builtin_metadata(),
            os.path.join(root, json_file) if "://" not in json_file else json_file,
            os.path.join(root, image_root),
            dataset_name_in_dict="obj365v2"
        )


_PREDEFINED_SPLITS_OBJECTS365V1 = {
    "objects365_v1_train": ("Objects365v1/train", "Objects365v1/objects365_train.json"),
    "objects365_v1_masktrain": ("Objects365v1/train", "Objects365v1/objects365_v1_train_with_mask.json"),
    "objects365_v1_val": ("Objects365v1/val/val", "Objects365v1/objects365_val.json"),
    "objects365_v1_val_mini": ("Objects365v1/val/val", "Objects365v1/objects365_val_mini.json"),
}

def register_all_obj365v1(root):
    for key, (image_root, json_file) in _PREDEFINED_SPLITS_OBJECTS365V1.items():
        register_coco_instances(
            key,
            _get_builtin_metadata_obj365v1(),
            os.path.join(root, json_file) if "://" not in json_file else json_file,
            os.path.join(root, image_root),
            dataset_name_in_dict="obj365v1"
        )


######## video instance segmentationi


# ==== Predefined splits for YTVIS 2019 ===========
_PREDEFINED_SPLITS_YTVIS_2019 = {
    "ytvis_2019_train": ("ytvis_2019/train/JPEGImages",
                         "ytvis_2019/annotations/instances_train_sub.json"),
    "ytvis_2019_val": ("ytvis_2019/val/JPEGImages",
                       "ytvis_2019/annotations/instances_val_sub.json"),
                    #  "ytvis_2019/annotations/instances_val_sub_GT.json"),
    "ytvis_2019_test": ("ytvis_2019/test/JPEGImages",
                        "ytvis_2019/test.json"),
    "ytvis_2019_dev": ("ytvis_2019/train/JPEGImages",
                       "ytvis_2019/instances_train_sub.json"),
}


# ==== Predefined splits for YTVIS 2021 ===========
_PREDEFINED_SPLITS_YTVIS_2021 = {
    "ytvis_2021_train": ("ytvis_2021/train/JPEGImages",
                         "ytvis_2021/annotations/instances_train_sub.json"),
    "ytvis_2021_val": ("ytvis_2021/val/JPEGImages",
                       "ytvis_2021/annotations/instances_val_sub.json"),
    "ytvis_2021_test": ("ytvis_2021/test/JPEGImages",
                        "ytvis_2021/test.json"),
    "ytvis_2021_dev": ("ytvis_2021/train/JPEGImages",
                       "ytvis_2021/instances_train_sub.json"),
    "ytvis_2022_val_full": ("ytvis_2022/val/JPEGImages",
                        "ytvis_2022/instances.json"),
    "ytvis_2022_val_sub": ("ytvis_2022/val/JPEGImages",
                       "ytvis_2022/instances_sub.json"),
}


_PREDEFINED_SPLITS_OVIS = {
    "ovis_train": ("ovis/train",
                         "ovis/annotations_train.json"),
    "ovis_val": ("ovis/valid",
                       "ovis/annotations_valid.json"),
    "ovis_train_sub": ("ovis/train",
                         "ovis/ovis_sub_train.json"),
    "ovis_val_sub": ("ovis/train",
                       "ovis/ovis_sub_val.json"),
}



def register_all_ytvis_2019(root):
    for key, (image_root, json_file) in _PREDEFINED_SPLITS_YTVIS_2019.items():
        # Assume pre-defined datasets live in `./datasets`.
        register_ytvis_instances(
            key,
            _get_ytvis_2019_instances_meta(),
            os.path.join(root, json_file) if "://" not in json_file else json_file,
            os.path.join(root, image_root),
            dataset_name_in_dict="ytvis19"
        )


def register_all_ytvis_2021(root):
    for key, (image_root, json_file) in _PREDEFINED_SPLITS_YTVIS_2021.items():
        # Assume pre-defined datasets live in `./datasets`.
        register_ytvis_instances(
            key,
            _get_ytvis_2021_instances_meta(),
            os.path.join(root, json_file) if "://" not in json_file else json_file,
            os.path.join(root, image_root),
            dataset_name_in_dict="ytvis21"
        )


def register_all_ovis(root):
    for key, (image_root, json_file) in _PREDEFINED_SPLITS_OVIS.items():
        # Assume pre-defined datasets live in `./datasets`.
        register_ytvis_instances(
            key,
            _get_ovis_instances_meta(),
            os.path.join(root, json_file) if "://" not in json_file else json_file,
            os.path.join(root, image_root),
            dataset_name_in_dict="ovis"
        )

_PREDEFINED_SPLITS_LVVIS = {
    "lvvis_train": ("lvvis/train/JPEGImages",
                         "lvvis/train_instances.json"),
    "lvvis_val": ("lvvis/val/JPEGImages",
                       "lvvis/val_instances.json"),
}

def register_all_lvvis(root):
    for key, (image_root, json_file) in _PREDEFINED_SPLITS_LVVIS.items():
        # Assume pre-defined datasets live in `./datasets`.
        register_ytvis_instances(
            key,
            _get_lvvis_instances_meta(),
            os.path.join(root, json_file) if "://" not in json_file else json_file,
            os.path.join(root, image_root),
            dataset_name_in_dict="lvvis"
        )





_PREDEFINED_SPLITS_REFYTBVOS = {
    "rvos-refcoco-mixed": ("coco/train2014", "annotations/refcoco-mixed/instances_train_video.json"),
    "rvos-refytb-train": ("ref-youtube-vos/train/JPEGImages", "ref-youtube-vos/train.json"),
    "rvos-refytb-val": ("ref-youtube-vos/valid/JPEGImages", "ref-youtube-vos/valid.json"),
}

def register_all_refytbvos_videos(root):
    for key, (image_root, json_file) in _PREDEFINED_SPLITS_REFYTBVOS.items():
        # Assume pre-defined datasets live in `./datasets`.
        register_rytvis_instances(
            key,
            _get_refcoco_meta(),
            os.path.join(root, json_file) if "://" not in json_file else json_file,
            os.path.join(root, image_root),
            has_expression=True
        )

# ==== Predefined splits for BDD100K object detection ===========
_PREDEFINED_SPLITS_BDD_OBJ_DET = {
    "bdd_det_train": ("/opt/tiger/NAS/junfengwu/datasets/bdd100k/images/100k/train", "/opt/tiger/NAS/junfengwu/datasets/bdd100k/labels/det_20/det_train_cocofmt.json"),
    "bdd_det_val": ("/opt/tiger/NAS/junfengwu/datasets/bdd100k/images/100k/val", "/opt/tiger/NAS/junfengwu/datasets/bdd100k/labels/det_20/det_val_cocofmt.json"),
}

# ==== Predefined splits for BDD100K instance segmentation ===========
_PREDEFINED_SPLITS_BDD_INST_SEG = {
    "bdd_inst_train": ("/opt/tiger/NAS/junfengwu/datasets/bdd100k/images/10k/train", "/opt/tiger/NAS/junfengwu/datasets/bdd100k/labels/ins_seg/polygons/ins_seg_train_cocoformat.json"),
    "bdd_inst_val": ("/opt/tiger/NAS/junfengwu/datasets/bdd100k/images/10k/val", "/opt/tiger/NAS/junfengwu/datasets/bdd100k/labels/ins_seg/polygons/ins_seg_val_cocoformat.json"),
}

# ==== Predefined splits for BDD100K box tracking ===========
_PREDEFINED_SPLITS_BDD_BOX_TRACK = {
    "bdd_box_track_train": ("/opt/tiger/NAS/junfengwu/datasets/bdd100k/images/track/train", "/opt/tiger/NAS/junfengwu/datasets/bdd100k/labels/box_track_20/box_track_train_cocofmt_uni.json"),
    "bdd_box_track_val": ("/opt/tiger/NAS/junfengwu/datasets/bdd100k/images/track/val", "/opt/tiger/NAS/junfengwu/datasets/bdd100k/labels/box_track_20/box_track_val_cocofmt_uni.json"),
}

# ==== Predefined splits for BDD100K seg tracking ===========
_PREDEFINED_SPLITS_BDD_SEG_TRACK = {
    "bdd_seg_track_train": ("/opt/tiger/NAS/junfengwu/datasets/bdd100k/images/seg_track_20/train", "/opt/tiger/NAS/junfengwu/datasets/bdd100k/labels/seg_track_20/seg_track_train_cocoformat_uni.json"),
    "bdd_seg_track_val": ("/opt/tiger/NAS/junfengwu/datasets/bdd100k/images/seg_track_20/val", "/opt/tiger/NAS/junfengwu/datasets/bdd100k/labels/seg_track_20/seg_track_val_cocoformat_uni.json"),
}


def register_all_bdd_obj_det(root):
    for key, (image_root, json_file) in _PREDEFINED_SPLITS_BDD_OBJ_DET.items():
        # Assume pre-defined datasets live in `./datasets`.
        register_coco_instances(
            key,
            _get_bdd_obj_det_meta(),
            os.path.join(root, json_file) if "://" not in json_file else json_file,
            os.path.join(root, image_root),
            dataset_name_in_dict="bdd_det"
        )
 

def register_all_bdd_inst_seg(root):
    for key, (image_root, json_file) in _PREDEFINED_SPLITS_BDD_INST_SEG.items():
        # Assume pre-defined datasets live in `./datasets`.
        register_coco_instances(
            key,
            _get_bdd_inst_seg_meta(),
            os.path.join(root, json_file) if "://" not in json_file else json_file,
            os.path.join(root, image_root),
            dataset_name_in_dict="bdd_inst"
        )


def register_all_bdd_box_track(root):
    for key, (image_root, json_file) in _PREDEFINED_SPLITS_BDD_BOX_TRACK.items():
        # Assume pre-defined datasets live in `./datasets`.
        register_ytvis_instances(
            key,
            _get_bdd_obj_track_meta(),
            os.path.join(root, json_file) if "://" not in json_file else json_file,
            os.path.join(root, image_root),
            dataset_name_in_dict="bdd_track_box",
            has_mask = False
        )


def register_all_bdd_seg_track(root):
    for key, (image_root, json_file) in _PREDEFINED_SPLITS_BDD_SEG_TRACK.items():
        # Assume pre-defined datasets live in `./datasets`.
        register_ytvis_instances(
            key,
            _get_bdd_obj_track_meta(),
            os.path.join(root, json_file) if "://" not in json_file else json_file,
            os.path.join(root, image_root),
            dataset_name_in_dict="bdd_track_seg"
        )

# def register_all_bdd_det_trk_mix(root):
#     for key, (image_root, json_file) in _PREDEFINED_SPLITS_BDD_DET_TRK_MIXED.items():
#         register_coco_instances(
#             key,
#             _get_bdd_obj_det_meta(),
#             os.path.join(root, json_file) if "://" not in json_file else json_file,
#             os.path.join(root, image_root),
#         )


_PREDEFINED_SPLITS_SOT = {
    # "sot_got10k_train": ("GOT10K/train", "GOT10K/train.json"),
    # "sot_got10k_val": ("GOT10K/val", "GOT10K/val.json"),
    # "sot_got10k_test": ("GOT10K/test", "GOT10K/test.json"),
    # "sot_lasot_train": ("LaSOT", "LaSOT/train.json"),
    # "sot_lasot_test": ("LaSOT", "LaSOT/test.json"),
    # "sot_lasot_ext_test": ("LaSOT_extension_subset", "LaSOT_extension_subset/test.json"),
    # "sot_trackingnet_train": ("TrackingNet", "TrackingNet/TRAIN.json"),
    # "sot_trackingnet_test": ("TrackingNet", "TrackingNet/TEST.json"),
    # "sot_coco_2017_train": ("coco/train2017", "coco/annotations/instances_train2017_video_sot.json"),
    # "sot_coco_2017_val": ("coco/val2017", "coco/annotations/instances_val2017_video_sot.json"),
    "ytbvos18_train": ("ytbvos18/train/JPEGImages", "ytbvos18/train/train.json"),
    "ytbvos18_val": ("ytbvos18/val/JPEGImages", "ytbvos18/val/val.json"),
    "mose_train": ("mose/train/JPEGImages", "mose/train/train.json"),
    "mose_val": ("mose/val/JPEGImages", "mose/val/val.json"),
    # "sot_davis17_val": ("DAVIS/JPEGImages/480p", "DAVIS/2017_val.json"),
    # "sot_nfs": ("nfs/sequences", "nfs/nfs.json"),
    # "sot_uav123": ("UAV123/data_seq/UAV123", "UAV123/UAV123.json"),
    # "sot_tnl2k_test": ("TNL-2K", "TNL-2K/test.json")
}

SOT_CATEGORIES = [{"color": [220, 20, 60], "isthing": 1, "id": 1, "name": "object"}] # only one class for visual grounding

_PREDEFINED_SPLITS_ODinW13_image = {
    # "odinw13_AerialDrone": ("/opt/tiger/odinw/dataset/AerialMaritimeDrone/large/valid/", "/opt/tiger/odinw/dataset/AerialMaritimeDrone/large/valid/annotations_without_background.json"),
    # "odinw13_Aquarium":  ("/opt/tiger/odinw/dataset/Aquarium/Aquarium Combined.v2-raw-1024.coco/valid" , "/opt/tiger/odinw/dataset/Aquarium/Aquarium Combined.v2-raw-1024.coco/valid/annotations_without_background.json"),
    # "odinw13_Rabbits":   ("/opt/tiger/odinw/dataset/CottontailRabbits/valid" , "/opt/tiger/odinw/dataset/CottontailRabbits/valid/annotations_without_background.json"),
    # "odinw13_EgoHands":  ("/opt/tiger/odinw/dataset/EgoHands/generic/mini_val" , "/opt/tiger/odinw/dataset/EgoHands/generic/mini_val/annotations_without_background.json"),
    # "odinw13_Mushrooms":  ("/opt/tiger/odinw/dataset/NorthAmericaMushrooms/North American Mushrooms.v1-416x416.coco/valid" , "/opt/tiger/odinw/dataset/NorthAmericaMushrooms/North American Mushrooms.v1-416x416.coco/valid/annotations_without_background.json"),
    # "odinw13_Packages":  ("/opt/tiger/odinw/dataset/Packages/Raw/valid" , "/opt/tiger/odinw/dataset/Packages/Raw/valid/annotations_without_background.json"),
    "odinw13_PascalVOC":  ("odinw/PascalVOC/valid/" , "odinw/PascalVOC/valid/annotations_without_background.json"),
    # "odinw13_Pistols":  ("/opt/tiger/odinw/dataset/pistols/export" , "/opt/tiger/odinw/dataset/pistols/export/val_annotations_without_background.json"),
    # "odinw13_Pothole":  ("/opt/tiger/odinw/dataset/pothole/valid" , "/opt/tiger/odinw/dataset/pothole/valid/annotations_without_background.json"),
    # "odinw13_Raccoon":  ("/opt/tiger/odinw/dataset/Raccoon/Raccoon.v2-raw.coco/valid" , "/opt/tiger/odinw/dataset/Raccoon/Raccoon.v2-raw.coco/valid/annotations_without_background.json"),
    # "odinw13_Shellfish":  ("/opt/tiger/odinw/dataset/ShellfishOpenImages/raw/valid" , "/opt/tiger/odinw/dataset/ShellfishOpenImages/raw/valid/annotations_without_background.json"),
    # "odinw13_Thermal":  ("/opt/tiger/odinw/dataset/thermalDogsAndPeople/valid" , "/opt/tiger/odinw/dataset/thermalDogsAndPeople/valid/annotations_without_background.json"),
    # "odinw13_Vehicles":  ("/opt/tiger/odinw/dataset/VehiclesOpenImages/416x416/mini_val" , "/opt/tiger/odinw/dataset/VehiclesOpenImages/416x416/mini_val/annotations_without_background.json"),
}
 
def register_all_odinw_image(root):
    for key, (image_root, json_file) in _PREDEFINED_SPLITS_ODinW13_image.items():
        # Assume pre-defined datasets live in `./datasets`.
        register_coco_instances(
            key,
            _get_odinw_image_meta(key),
            os.path.join(root, json_file) if "://" not in json_file else json_file,
            os.path.join(root, image_root),
            dataset_name_in_dict = key
        )

def _get_sot_meta():
    thing_ids = [k["id"] for k in SOT_CATEGORIES if k["isthing"] == 1]
    thing_colors = [k["color"] for k in SOT_CATEGORIES if k["isthing"] == 1]
    assert len(thing_ids) == 1, len(thing_ids)

    thing_dataset_id_to_contiguous_id = {k: i for i, k in enumerate(thing_ids)}
    thing_classes = [k["name"] for k in SOT_CATEGORIES if k["isthing"] == 1]
    ret = {
        "thing_dataset_id_to_contiguous_id": thing_dataset_id_to_contiguous_id,
        "thing_classes": thing_classes,
        "thing_colors": thing_colors,
    }
    return ret

def register_all_sot(root):
    for key, (image_root, json_file) in _PREDEFINED_SPLITS_SOT.items():
        has_mask = ("coco" in key) or ("vos" in key) or ("davis" in key)
        # Assume pre-defined datasets live in `./datasets`.
        register_ytvis_instances(
            key,
            _get_sot_meta(),
            os.path.join(root, json_file) if "://" not in json_file else json_file,
            os.path.join(root, image_root),
            dataset_name_in_dict='ytbvos',
            has_mask=has_mask,
            sot=True
        )


if __name__.endswith(".builtin"):
    # Assume pre-defined datasets live in `./datasets`.
    _root = os.getenv("DETECTRON2_DATASETS", "datasets")
    # refcoco/g/+
    register_all_refcoco(_root)
    register_all_sa1b(_root)
    # register_all_sa1b_part(_root)
    register_all_obj365v2(_root)
    register_all_obj365v1(_root)
    register_all_openimage(_root)
    register_all_vg(_root)
    register_all_grit(_root)

    # zero-shot
    register_all_odinw_image(_root)

    register_all_lvvis(_root)
    register_all_ytvis_2019(_root)
    register_all_ytvis_2021(_root)
    register_all_ovis(_root)
    register_all_UVO_image(_root)
    register_all_UVO_dense_video(_root)
    register_all_BURST_video(_root)
    register_all_TAO_video(_root)
    register_all_refytbvos_videos(_root)

    # vis image format
    register_all_YTVIS19_image(_root)
    register_all_YTVIS21_image(_root)
    register_all_OVIS_image(_root)
    register_all_TAO_image(_root)
    register_all_BURST_image(_root)

    # BDD100K
    register_all_bdd_obj_det(_root)
    register_all_bdd_inst_seg(_root)
    register_all_bdd_box_track(_root)
    register_all_bdd_seg_track(_root)

    # VOS/SOT
    register_all_sot(_root)
    
    # PACO
    register_all_paco(_root)
    
    # PASCAL_JOINT
    register_all_pascal_joint(_root)
    # # PASCAL_PART
    # register_all_pascal_part(_root)
    
    # PartImageNet
    register_all_partimagenet(_root)
    
    # COCO-panoptic
    register_all_coco_panoptic_annos_sem_seg(_root)
    
    # ade20k-joint
    register_all_ade20k_joint(_root)
    
    # ade20k-base
    register_all_ade20k_base(_root)
    
    # pascalvoc-base
    register_all_pascalvoc_base(_root)
    
    # pascalvoc-joint
    register_all_pascalvoc_joint(_root)
    
    # partimagenet-joint
    register_all_partimagenet_joint(_root)
    
    # pascal_part_open_vocabulary
    register_all_pascal_part_open_vocabulary(_root)
    
    # vg-joint [split for object and part]
    # register_all_vg_joint(_root)
    
    # voc-2010
    register_all_pascal_voc(_root)
    
    # ade20k-part-234
    register_ade20k_part_234(_root)
    
    # pascal-part-116
    register_pascal_part_116(_root)
    
    # seginw
    register_all_seginw(_root)
    
    # partimagenet_semseg
    register_all_partimagenet_annos_sem_seg(_root)