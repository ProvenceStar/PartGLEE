# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
import math
from os import DirEntry
import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from detectron2.structures import Boxes, ImageList, Instances, BitMasks
from detectron2.utils.memory import retry_if_cuda_oom
from detectron2.data import MetadataCatalog
from .models._CONSTANTS import LVIS_CATEGORIES, OBJ365_CATEGORIESV2, OPENIMAGE_CATEGORIES, OPENIMAGES_PART_CATEGORIES, OPENIMAGES_OBJECT_CATEGORIES, \
                              BURST_CATEGORIES, YTVIS_CATEGORIES_2019, OVIS_CATEGORIES, YTVIS_CATEGORIES_2021, LVVIS_CATEGORIES, \
                              UVO_CATEGORIES, BDD_DET_CATEGORIES, BDD_INST_CATEGORIES, BDD_TRACK_CATEGORIES, VG_name_list, \
                              TAO_CATEGORIES, VG_OBJECT_CATEGORIES, VG_PART_CATEGORIES, TAO_CATEGORIES, odinw_category_dict, \
                              PACO_CATEGORIES, ADE20K_OBJECT_BASE_CLASS_NAMES, ADE20K_BASE_CLASS_NAMES, ADE20K_OBJECT_CLASS_NAMES, ADE20K_PART_CLASS_NAMES, \
                              PASCALVOC_OBJECT_CLASS_NAMES, PASCALVOC_PART_CLASS_NAMES, PASCALVOC_OBJECT_BASE_CLASS_NAMES, PASCALVOC_BASE_CLASS_NAMES, PASCAL_BASE_OBJECT_CATEGORIES, \
                              PASCAL_JOINT_CATEGORIES, PASCAL_OBJECT_CATEGORIES, PASCAL_PART_CATEGORIES, PARTIMAGENET_JOINT_CATEGORIES, PARTIMAGENET_OBJECT_CATEGORIES, PARTIMAGENET_PART_CATEGORIES, \
                              COCO_PANOPTIC_CATEGORIES, PASCAL_OPEN_VOCABULARY_JOINT_BASE_CATEGORIES, PASCAL_BASE_PART_CATEGORIES, \
                              PASCAL_VAL_PART_CATEGORIES, PASCAL_VAL_OBJECT_CATEGORIES, PASCAL_OPEN_VOCABULARY_JOINT_VAL_CATEGORIES, SEGINW_CATEGORIES, PASCAL_VOC_CATEGORIES, OBJECT_LEVEL_DATASETS
from .models._INFERENCE_CONSTANTS import DATASET_SPECIFIC_CATEGORIES, PART_DATASETS_OBJECT_CATEGORY_INDEX, PART_DATASETS_PART_CATEGORY_INDEX, DATASET_CATEGORY_SET
import torchvision.ops as ops
import random
from PIL import Image
from detectron2.layers import ShapeSpec
from detectron2.modeling import META_ARCH_REGISTRY, build_backbone, detector_postprocess
from detectron2.structures import Boxes, ImageList, Instances, BitMasks, PolygonMasks

from .models.PartGLEE_Model import PartGLEE_Model
from .models.matcher import HungarianMatcher
from .models.criterion import SetCriterion
from .models.matcher import HungarianMatcher

from detectron2.modeling.postprocessing import sem_seg_postprocess
from .utils import box_ops
from scipy.optimize import linear_sum_assignment
import os

__all__ = ["PartGLEE"]

@META_ARCH_REGISTRY.register()
class PartGLEE(nn.Module):
    """
    Implement PartGLEE
    """
    def __init__(self, cfg):
        super().__init__()
        # loss weights
        class_weight = cfg.MODEL.MaskDINO.CLASS_WEIGHT
        cost_class_weight = cfg.MODEL.MaskDINO.COST_CLASS_WEIGHT
        cost_dice_weight = cfg.MODEL.MaskDINO.COST_DICE_WEIGHT
        dice_weight = cfg.MODEL.MaskDINO.DICE_WEIGHT  #
        cost_mask_weight = cfg.MODEL.MaskDINO.COST_MASK_WEIGHT  #
        mask_weight = cfg.MODEL.MaskDINO.MASK_WEIGHT
        cost_box_weight = cfg.MODEL.MaskDINO.COST_BOX_WEIGHT
        box_weight = cfg.MODEL.MaskDINO.BOX_WEIGHT  #
        cost_giou_weight = cfg.MODEL.MaskDINO.COST_GIOU_WEIGHT
        giou_weight = cfg.MODEL.MaskDINO.GIOU_WEIGHT  #
        
        self.sem_seg_postprocess_before_inference = cfg.MODEL.MaskDINO.TEST.SEM_SEG_POSTPROCESSING_BEFORE_INFERENCE or cfg.MODEL.MaskDINO.TEST.PANOPTIC_ON or cfg.MODEL.MaskDINO.TEST.INSTANCE_ON
        
        # building matcher
        matcher = HungarianMatcher(
            cost_class=cost_class_weight,
            cost_mask=cost_mask_weight,
            cost_dice=cost_dice_weight,
            cost_box=cost_box_weight,
            cost_giou=cost_giou_weight,
            num_points=cfg.MODEL.MaskDINO.TRAIN_NUM_POINTS,
        )
        
        self.pseudo_video = cfg.MODEL.PSEUDO_VIDEO

        self.device = torch.device(cfg.MODEL.DEVICE)
        
        uvo_calss_name = [cat['name']+', object'  for cat in UVO_CATEGORIES[:-1] ]  + [UVO_CATEGORIES[-1]['name']]
        coco_class_name = ['person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train', 'truck', 'boat', 'traffic light', 'fire hydrant', 'stop sign', 'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse', 'sheep', 'cow', 'elephant', 'bear', 'zebra', 'giraffe', 'backpack', 'umbrella', 'handbag', 'tie', 'suitcase', 'frisbee', 'skis', 'snowboard', 'sports ball', 'kite', 'baseball bat', 'baseball glove', 'skateboard', 'surfboard', 'tennis racket', 'bottle', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple', 'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake', 'chair', 'couch', 'potted plant', 'bed', 'dining table', 'toilet', 'tv', 'laptop', 'mouse', 'remote', 'keyboard', 'cell phone', 'microwave', 'oven', 'toaster', 'sink', 'refrigerator', 'book', 'clock', 'vase', 'scissors', 'teddy bear', 'hair drier', 'toothbrush']
        openimages_object_indices = [cat['id'] for cat in OPENIMAGES_OBJECT_CATEGORIES]
        openimages_part_indices = [cat['id'] for cat in OPENIMAGES_PART_CATEGORIES]
        
        self.LVIS_class_names = [cat['name'] for cat in LVIS_CATEGORIES]
        self.paco_class_names = [cat['name'] for cat in PACO_CATEGORIES]
        self.ade20k_class_names = ADE20K_OBJECT_CLASS_NAMES + ADE20K_PART_CLASS_NAMES
        self.ade20k_base_class_names = ADE20K_OBJECT_BASE_CLASS_NAMES + ADE20K_BASE_CLASS_NAMES
        self.pascalvoc_base_class_names = PASCALVOC_OBJECT_BASE_CLASS_NAMES + PASCALVOC_BASE_CLASS_NAMES
        self.pascalvoc_class_names = PASCALVOC_OBJECT_CLASS_NAMES + PASCALVOC_PART_CLASS_NAMES
        self.sa1b_joint_class_names = ['object', 'part']
        self.pascal_joint_class_names = [cat['name'] for cat in PASCAL_JOINT_CATEGORIES]
        self.partimagenet_joint_class_names = [cat['name'] for cat in PARTIMAGENET_JOINT_CATEGORIES]
        self.pascal_open_vocabulary_base_class_names = [cat['name'] for cat in PASCAL_OPEN_VOCABULARY_JOINT_BASE_CATEGORIES]
        self.pascal_open_vocabulary_val_class_names = [cat['name'] for cat in PASCAL_OPEN_VOCABULARY_JOINT_VAL_CATEGORIES]
        self.coco_panoptic_class_names = [cat['name'] for cat in COCO_PANOPTIC_CATEGORIES]
        self.voc_class_names = [cat['name'] for cat in PASCAL_VOC_CATEGORIES]
        self.OBJ365_class_names = [cat['name'] for cat in OBJ365_CATEGORIESV2]
        self.OPENIMAGE_class_names = [cat['name'] for cat in OPENIMAGE_CATEGORIES]
        self.VG_name_list = VG_name_list
        self.vg_object_name_list = VG_OBJECT_CATEGORIES
        self.vg_part_name_list = VG_PART_CATEGORIES
        self.brust_class_names= [cat['name'] for cat in BURST_CATEGORIES] 
        self.openimage_joint_object_mapping = {original_order - 1: idx for idx, original_order in enumerate(openimages_object_indices)}
        self.openimage_joint_part_mapping = {original_order - 1: idx for idx, original_order in enumerate(openimages_part_indices)}
        
        self.category_set = DATASET_CATEGORY_SET
        
        self.dataset_name_dicts = {
            'coco': coco_class_name,
            'voc': self.voc_class_names,
            'coco_clip':coco_class_name,
            'lvis': self.LVIS_class_names,
            'paco': self.paco_class_names,
            'ade20k': self.ade20k_class_names,
            'ade20k_base': self.ade20k_base_class_names,
            'pascalvoc': self.pascalvoc_class_names,
            'pascalvoc_base': self.pascalvoc_base_class_names,
            'sa1b_joint': self.sa1b_joint_class_names,
            'pascal_joint': self.pascal_joint_class_names,
            'partimagenet_joint': self.partimagenet_joint_class_names,
            'partimagenet_parsed': self.partimagenet_joint_class_names,
            'partimagenet_val': self.partimagenet_joint_class_names,
            'pascal_open_vocabulary': self.pascal_open_vocabulary_base_class_names,
            'pascal_part_parsed': self.pascal_joint_class_names,
            'pascal_open_vocabulary_val': self.pascal_open_vocabulary_val_class_names,
            'coco_panoptic': self.coco_panoptic_class_names,
            'obj365': self.OBJ365_class_names,
            'openimage': self.OPENIMAGE_class_names, 
            'openimage_joint': self.OPENIMAGE_class_names, 
            'lvis_clip': self.LVIS_class_names,
            'obj365_clip': self.OBJ365_class_names ,
            'openimage_clip': self.OPENIMAGE_class_names, 
            'bdd_det': [cat['name']  for cat in BDD_DET_CATEGORIES ] ,
            'bdd_inst': [cat['name']  for cat in BDD_INST_CATEGORIES ] ,
            'ytvis19':[cat['name'] for cat in YTVIS_CATEGORIES_2019],
            'image_yt19': [cat['name'] for cat in YTVIS_CATEGORIES_2019],
            'ytvis21': [cat['name'] for cat in YTVIS_CATEGORIES_2021],
            'image_yt21': [cat['name'] for cat in YTVIS_CATEGORIES_2021],
            'ovis': [cat['name'] for cat in OVIS_CATEGORIES],
            'image_o': [cat['name'] for cat in OVIS_CATEGORIES],
            'lvvis':  [cat['name'] for cat in LVVIS_CATEGORIES], 
            'uvo_video':uvo_calss_name,
            'burst':self.brust_class_names,
            'image_bur': self.brust_class_names,
            'image_tao': [cat['name'] for cat in TAO_CATEGORIES],
            'tao_video': [cat['name'] for cat in TAO_CATEGORIES],
            'sa1b': ['object'],
            'sa1b_clip': ['object'],
            'grounding': ['object'],
            'rvos': ['object'],
            'bdd_track_box':  [cat['name']  for cat in BDD_TRACK_CATEGORIES ] ,
            'bdd_track_seg':  [cat['name']  for cat in BDD_TRACK_CATEGORIES ] ,
            'ytbvos': ['object'],
            'seginw_Brain-Tumor': ['tumor'],
        }
        self.object_category_index_mapper = PART_DATASETS_OBJECT_CATEGORY_INDEX
        self.part_category_index_mapper = PART_DATASETS_PART_CATEGORY_INDEX
        self.num_class = DATASET_SPECIFIC_CATEGORIES
        for k,v in odinw_category_dict.items():
            if k == 'odinw13_Rabbits':
                self.dataset_name_dicts.update({k:['rabbits' for cat in v ]})
            elif k == 'odinw13_EgoHands':
                self.dataset_name_dicts.update({k:['hand hands' for cat in v ]})
            elif k == 'odinw13_Mushrooms':
                self.dataset_name_dicts.update({k:['mushroom ' + cat['name']for cat in v ]})
            elif k=='odinw13_Packages':
                self.dataset_name_dicts.update({k:['packages' for cat in v ]})
            else:
                self.dataset_name_dicts.update({k:[cat['name'] for cat in v ]})
            self.num_class.update({k:len(v)})

        # cfg.DATALOADER.DATASET_BS[-1]
        self.video_info = {'bz':cfg.DATALOADER.DATASET_BS[-1], 'len':cfg.INPUT.SAMPLING_FRAME_NUM}

        self.unify_object_part = cfg.MODEL.MaskDINO.UNIFY_OBJECT_PART
        self.partglee = PartGLEE_Model(cfg, matcher, self.device, self.video_info, cfg.MODEL.CONTRAS_MEAN, self.unify_object_part)
        
        self.cate_sampled = False
        if cfg.MODEL.MAX_CATEGORY_LEN is not None:
            self.cate_sampled = True
            self.max_category_len = cfg.MODEL.MAX_CATEGORY_LEN 

        size_divisibility = cfg.MODEL.MaskDINO.SIZE_DIVISIBILITY
        if size_divisibility < 0:
            # use backbone size_divisibility if not set
            size_divisibility = self.partglee.backbone.size_divisibility
        self.size_divisibility = size_divisibility

        pixel_mean = torch.Tensor(cfg.MODEL.PIXEL_MEAN).to(self.device).view(3, 1, 1)
        pixel_std = torch.Tensor(cfg.MODEL.PIXEL_STD).to(self.device).view(3, 1, 1)
        self.normalizer = lambda x: (x - pixel_mean) / pixel_std
        self.to(self.device)

        # Loss parameters:
        deep_supervision = cfg.MODEL.MaskDINO.DEEP_SUPERVISION
        no_object_weight = cfg.MODEL.MaskDINO.NO_OBJECT_WEIGHT

        weight_dict = {"loss_ce": class_weight}
        weight_dict.update({"loss_conf": class_weight})
        weight_dict.update({"loss_mask": mask_weight, "loss_dice": dice_weight})
        weight_dict.update({"loss_bbox":box_weight,"loss_giou":giou_weight})
        weight_dict.update({"track_loss": 2.0})
        weight_dict.update({"dist_loss": 4.0})
        # two stage is the query selection scheme
        if cfg.MODEL.MaskDINO.TWO_STAGE:
            interm_weight_dict = {}
            interm_weight_dict.update({k + f'_interm': v for k, v in weight_dict.items()})
            weight_dict.update(interm_weight_dict)
        # denoising training
        dn = cfg.MODEL.MaskDINO.DN
        if dn == "standard":
            weight_dict.update({k + f"_dn": v for k, v in weight_dict.items() if k!="loss_mask" and k!="loss_dice"})
            dn_losses=["labels", "boxes"]
        elif dn == "seg":
            weight_dict.update({k + f"_dn": v for k, v in weight_dict.items()})
            dn_losses=["labels", "masks", "boxes"]
        else:
            dn_losses=[]
        if deep_supervision:
            dec_layers = cfg.MODEL.MaskDINO.DEC_LAYERS
            aux_weight_dict = {}
            for i in range(dec_layers):
                aux_weight_dict.update({k + f"_{i}": v for k, v in weight_dict.items()})
            weight_dict.update(aux_weight_dict)
        if cfg.MODEL.MaskDINO.UNIFY_OBJECT_PART:
            object_weight_dict = weight_dict.copy()
            for key in object_weight_dict.keys():
                weight_dict.update({"part_" + key: weight_dict[key]})
            if cfg.MODEL.MaskDINO.USE_BOX_RESTRICTIONS:
                weight_dict.update({"part_loss_box_restriction": 0.1})
                if cfg.MODEL.MaskDINO.TWO_STAGE:
                    weight_dict.update({"part_loss_box_restriction_interm": 0.1})
            if deep_supervision:
                dec_layers = cfg.MODEL.MaskDINO.DEC_LAYERS
                aux_weight_dict = {}
                for i in range(dec_layers):
                    aux_weight_dict.update({k + f"_{i}": v for k, v in weight_dict.items() if 'loss_box_restriction' in k})
                weight_dict.update(aux_weight_dict)
        if cfg.MODEL.MaskDINO.BOX_LOSS:
            losses = ["labels", "masks", "boxes", "conf"]
        else:
            losses = ["labels", "masks", "conf"]
        # building criterion
        self.criterion = SetCriterion(
            num_classes=1,
            matcher=matcher,
            weight_dict=weight_dict,
            eos_coef=no_object_weight,
            losses=losses,
            num_points=cfg.MODEL.MaskDINO.TRAIN_NUM_POINTS,
            oversample_ratio=cfg.MODEL.MaskDINO.OVERSAMPLE_RATIO,
            importance_sample_ratio=cfg.MODEL.MaskDINO.IMPORTANCE_SAMPLE_RATIO,
            dn=cfg.MODEL.MaskDINO.DN,
            dn_losses=dn_losses,
            panoptic_on=cfg.MODEL.MaskDINO.PANO_BOX_LOSS,
            semantic_ce_loss=cfg.MODEL.MaskDINO.TEST.SEMANTIC_ON and cfg.MODEL.MaskDINO.SEMANTIC_CE_LOSS and not cfg.MODEL.MaskDINO.TEST.PANOPTIC_ON,
            topk_object_queries=cfg.MODEL.MaskDINO.TOPK_OBJECT_QUERIES,
            num_part_queries=cfg.MODEL.MaskDINO.NUM_PART_QUERIES,
            box_restrictions=cfg.MODEL.MaskDINO.USE_BOX_RESTRICTIONS,
        )

        self.visualize = cfg.MODEL.MaskDINO.TEST.VISUALIZE

        self.test_topk_per_image = cfg.TEST.DETECTIONS_PER_IMAGE
        self.num_queries = cfg.MODEL.MaskDINO.NUM_OBJECT_QUERIES
        self.instance_on = cfg.MODEL.MaskDINO.TEST.INSTANCE_ON
        self.semantic_on = cfg.MODEL.MaskDINO.TEST.SEMANTIC_ON
        self.panoptic_on = cfg.MODEL.MaskDINO.TEST.PANOPTIC_ON
        self.semantic_ce_loss = cfg.MODEL.MaskDINO.TEST.SEMANTIC_ON and cfg.MODEL.MaskDINO.SEMANTIC_CE_LOSS and not cfg.MODEL.MaskDINO.TEST.PANOPTIC_ON
        self.pano_temp = cfg.MODEL.MaskDINO.TEST.PANO_TEMPERATURE
        self.transform_eval = cfg.MODEL.MaskDINO.TEST.PANO_TRANSFORM_EVAL
        self.overlap_threshold = cfg.MODEL.MaskDINO.TEST.OVERLAP_THRESHOLD
        self.object_mask_threshold = cfg.MODEL.MaskDINO.TEST.OBJECT_MASK_THRESHOLD
        self.metadata = MetadataCatalog.get(cfg.DATASETS.TRAIN[0])
        
        self.object_level_datasets = OBJECT_LEVEL_DATASETS

        if not self.semantic_on:
            assert self.sem_seg_postprocess_before_inference
            
        self.visaul_prompt = cfg.MODEL.VISUAL_PROMPT

        self.is_lsj = cfg.INPUT.DATASET_MAPPER_NAME == 'coco_instance_lsj'
        # self.LVIS_class_names = ['person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train', 'truck', 'boat', 'traffic light', 'fire hydrant', 'stop sign', 'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse', 'sheep', 'cow', 'elephant', 'bear', 'zebra', 'giraffe', 'backpack', 'umbrella', 'handbag', 'tie', 'suitcase', 'frisbee', 'skis', 'snowboard', 'sports ball', 'kite', 'baseball bat', 'baseball glove', 'skateboard', 'surfboard', 'tennis racket', 'bottle', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple', 'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake', 'chair', 'couch', 'potted plant', 'bed', 'dining table', 'toilet', 'tv', 'laptop', 'mouse', 'remote', 'keyboard', 'cell phone', 'microwave', 'oven', 'toaster', 'sink', 'refrigerator', 'book', 'clock', 'vase', 'scissors', 'teddy bear', 'hair drier', 'toothbrush']

        # VIS  
        self.is_multi_cls = True
        self.apply_cls_thres  = 0.01

        self.save_path_prefix = os.path.join(cfg.OUTPUT_DIR, "Annotations")

        if cfg.MODEL.FREEZE_WHOLE:
            for p in self.partglee.parameters():
                p.requires_grad = False
            for name, param in self.partglee.named_parameters():
                if 'part' in name:
                    print('require_grad:', name)
                    param.requires_grad = True

    def vg_category_name(self, targets, task, batched_inputs):
        all_pos_categories = []
        for tgt, batchinput in zip(targets, batched_inputs):
            assert len(batchinput['object_descriptions']) == len(tgt['labels'])
            all_pos_categories = all_pos_categories + batchinput['object_descriptions']

        all_pos_set = set(all_pos_categories)

        if task == 'vg': 
            # 每次采样200个word出来：
            dataset_neg_category_names = random.sample(self.VG_name_list, 200)
            dataset_neg_category_names = set(dataset_neg_category_names)
            
            rest_cate = dataset_neg_category_names - all_pos_set
            label_category = list(all_pos_set)
            assert self.max_category_len*2 >= len(label_category)
            sample_cate = random.sample(rest_cate, self.max_category_len*2-len(all_pos_set))  # 采样的category id
            batch_name_list = label_category+sample_cate
        elif task == 'grit': 
            dataset_neg_category_names = random.sample(self.VG_name_list, 100)
            dataset_neg_category_names = set(dataset_neg_category_names)
            
            rest_cate = dataset_neg_category_names - all_pos_set
            label_category = list(all_pos_set)
            assert self.max_category_len >= len(label_category)
            sample_cate = random.sample(rest_cate, self.max_category_len-len(all_pos_set))  # 采样的category id
            batch_name_list = label_category+sample_cate
        random.shuffle(batch_name_list)

        for tgt, batchinput in zip(targets,batched_inputs): # 重新将GT里的id 顺序映射到0~100内，每个batch动态调整
            # tgt_ids = tgt["labels"]
            gt_new_ids = [batch_name_list.index(l) for l in batchinput['object_descriptions']] #原来的id在new_cate_idx里的顺序
            # ori_names = [self.OBJ365_class_names[i] for i in label_list]
            # now_names = [batch_name_list[i] for i in gt_new_ids]
            tgt['labels'] = torch.tensor(gt_new_ids).to(tgt['labels'])

        return batch_name_list, targets
 
    def vg_joint_category_name_sampling(self, targets, task, batched_inputs):
        all_pos_object_categories = []
        all_pos_part_categories = []
        
        object_descriptions = []
        part_descriptions = []
        object_tgt_map_indices = []
        part_tgt_map_indices = []
        
        for tgt, batched_input in zip(targets, batched_inputs):
            assert len(batched_input['object_descriptions']) == len(tgt['labels'])
            object_description = [description for description in batched_input['object_descriptions'] if description in self.vg_object_name_list]
            part_description = [description for description in batched_input['object_descriptions'] if description in self.vg_part_name_list]
            object_tgt_map_indice = torch.tensor(list(map(lambda x: x in object_description, batched_input['object_descriptions']))).to(tgt['labels'].device)
            object_tgt_map_indices.append(object_tgt_map_indice)
            part_tgt_map_indice = torch.tensor(list(map(lambda x: x in part_description, batched_input['object_descriptions']))).to(tgt['labels'].device)
            part_tgt_map_indices.append(part_tgt_map_indice)
            all_pos_object_categories = all_pos_object_categories + object_description
            all_pos_part_categories = all_pos_part_categories + part_description
            object_descriptions.append(object_description)
            part_descriptions.append(part_description)

        assert task == 'vg_joint', "Object and Part Category Sampling must be performed in visual_genome task"
        
       # 每次采样200个word出来：
        dataset_neg_object_category_names = random.sample(self.vg_object_name_list, 100)
        dataset_neg_part_category_names = random.sample(self.vg_part_name_list, 100)
        dataset_neg_category_names = set(dataset_neg_object_category_names + dataset_neg_part_category_names)
        
        # Object Categories
        all_pos_object_set = set(all_pos_object_categories)
        rest_cate = dataset_neg_category_names - all_pos_object_set
        label_category = list(all_pos_object_set)
        assert self.max_category_len*2 >= len(label_category)
        sample_cate = random.sample(rest_cate, self.max_category_len*2-len(all_pos_object_set))  # 采样的category id
        object_batch_name_list = label_category + sample_cate
        random.shuffle(object_batch_name_list)
        
        # Part Categories
        all_pos_part_set = set(all_pos_part_categories)
        rest_cate = dataset_neg_category_names - all_pos_part_set
        label_category = list(all_pos_part_set)
        assert self.max_category_len*2 >= len(label_category)
        sample_cate = random.sample(rest_cate, self.max_category_len*2-len(all_pos_part_set))  # 采样的category id
        part_batch_name_list = label_category + sample_cate
        random.shuffle(part_batch_name_list)

        object_targets = []
        part_targets = []
        
        for i, (tgt, batched_input) in enumerate(zip(targets, batched_inputs)): # 重新将GT里的id 顺序映射到0~100内，每个batch动态调整
            object_gt_new_ids = [object_batch_name_list.index(l) for l in object_descriptions[i]] #原来的id在new_cate_idx里的顺序
            part_gt_new_ids = [part_batch_name_list.index(l) for l in part_descriptions[i]] #原来的id在new_cate_idx里的顺序
            if object_tgt_map_indices[i].shape[0] == 0:
                object_tgt_boxes = torch.empty(size=(0, 4), device=tgt['labels'].device)
            else:
                object_tgt_boxes = tgt['boxes'][object_tgt_map_indices[i]]
            object_targets.append(
                {
                    "labels": torch.tensor(object_gt_new_ids).to(tgt['labels'].device),
                    "masks": None,
                    "boxes": object_tgt_boxes,
                }
            )
            if part_tgt_map_indices[i].shape[0] == 0:
                part_tgt_boxes = torch.empty(size=(0, 4), device=tgt['labels'].device)
            else:
                part_tgt_boxes = tgt['boxes'][part_tgt_map_indices[i]]
            part_targets.append(
                {
                    "labels": torch.tensor(part_gt_new_ids).to(tgt['labels'].device),
                    "masks": None,
                    "boxes": part_tgt_boxes,
                }
            )
        batch_name_list = object_batch_name_list + part_batch_name_list

        return batch_name_list, object_targets, part_targets

    def category_name_sampling(self, targets, task):
        # all_set = self.OPENIMAGE_set if task == 'openimage' else self.OBJ365_set
        all_set = self.category_set[task]
        dataset_category_names = self.dataset_name_dicts[task]
        # dataset_category_names = self.OPENIMAGE_class_names if   task == 'openimage'  else self.OBJ365_class_names
        # 每个batch里采样一次，而不是每张图
        tgt_ids = torch.cat([v["labels"] for v in targets]).tolist()

        label_category_set = set(tgt_ids)
        rest_cate = all_set - label_category_set
        label_category = list(label_category_set)
        assert self.max_category_len >= len(label_category)
        sample_cate = random.sample(rest_cate, self.max_category_len-len(label_category))  # 采样的category id
        new_cate_idx = label_category+sample_cate
        batch_name_list = [dataset_category_names[i] for i in new_cate_idx]
        
        for tgt in targets: # 重新将GT里的id 顺序映射到0~100内，每个batch动态调整
            # tgt_ids = tgt["labels"]
            label_list = tgt['labels'].tolist()
            gt_new_ids = [label_category.index(l) for l in label_list] #原来的id在new_cate_idx里的顺序
            # ori_names = [self.OBJ365_class_names[i] for i in label_list]
            # now_names = [batch_name_list[i] for i in gt_new_ids]
            # import pdb;pdb.set_trace()
            tgt['labels'] = torch.tensor(gt_new_ids).to(tgt['labels'])
            
        return batch_name_list, targets
    
    def forward(self, batched_inputs):
        prompt_list = {}
        if 'dataset_name' in batched_inputs[0]:
            # print([x['dataset_name'] for x in batched_inputs])
            if 'rvos' in  batched_inputs[0]['dataset_name']:
                task = 'rvos'
                prompt_list["grounding"] = []
                for x in batched_inputs:
                    prompt_list["grounding"] += x["expressions"]
            elif 'obj365' in batched_inputs[0]['dataset_name']:
                task = 'obj365'
                if self.pseudo_video and self.training:
                    task = 'obj365_clip'
            else:
                task = batched_inputs[0]['dataset_name'] # [ovis, ytvis19, ytvis21, uvo_video, bdd_det, bdd_inst, pascal_joint, pascal_part(_open_vocabulary), coco_panoptic]
                if task == 'UVO_image':
                    task = 'sa1b'
                    if self.pseudo_video and self.training:
                        task = 'sa1b_clip'
        elif "expressions" in batched_inputs[0]:
            task = 'grounding'
            prompt_list["grounding"] = [x["expressions"] for x in batched_inputs]
        elif 'task' in batched_inputs[0]:
            if batched_inputs[0]['task'] == 'sa1b':
                task = 'sa1b'
                if self.pseudo_video and self.training:
                    task = 'sa1b_clip'
            else:
                task = batched_inputs[0]['task']
        elif 'not_exhaustive_category_ids' in batched_inputs[0]:
            task = 'lvis'
        else:
            if 'custom_object_categories' in batched_inputs[0].keys() and 'custom_part_categories' in batched_inputs[0].keys():
                task = 'custom'
                custom_object_categories = batched_inputs[0]['custom_object_categories']
                custom_part_categories = batched_inputs[0]['custom_part_categories']
            else:
                task = 'undefined'
            
        if task == 'coco' and self.pseudo_video and self.training:
            task = 'coco_clip'

        if 'pseudo_video' in batched_inputs[0] and  batched_inputs[0]['pseudo_video']  == True  and self.pseudo_video and self.training:
            if task in ['lvis', 'openimage']:
                task = task+'_clip'

        batch_name_list = None

        if self.training:
            images = self.preprocess_image(batched_inputs, task)
            if task in ['vis', 'ovis', 'ytvis19' ,'ytvis21', 'lvvis', 'rvos','ytbvos', 'uvo_video','burst', 'coco_clip','obj365_clip' ,'sa1b_clip','lvis_clip','openimage_clip','bdd_track_box','bdd_track_seg']:
                gt_instances = [x["instances"] for x in batched_inputs]
            else:
                gt_instances = [x["instances"].to(self.device) for x in batched_inputs]
            
            if task in ['sa1b_joint', 'paco', 'ade20k', 'ade20k_base', 'pascalvoc', 'pascalvoc_base', 'pascal_joint', 'partimagenet_joint', 'pascal_open_vocabulary', 'openimage_joint', 'partimagenet_parsed', 'pascal_part_parsed',] and self.unify_object_part:
                object_targets, part_targets = self.prepare_targets_object_part(batched_inputs, gt_instances, images, task)
            else:
                targets, prompt_list = self.prepare_targets(batched_inputs, gt_instances, images, prompt_list, task)
            
            if task in ['obj365', 'openimage', 'lvis','obj365_clip', 'lvis_clip', 'openimage_clip']:
                batch_name_list, targets = self.category_name_sampling(targets, task)
            elif task in ['vg', 'grit']:
                batch_name_list, targets = self.vg_category_name(targets, task, batched_inputs)
            elif task in ['vg_joint']:
                batch_name_list, object_targets, part_targets = self.vg_joint_category_name_sampling(targets, task, batched_inputs)
            else:
                batch_name_list = self.dataset_name_dicts[task]
            # batch_name_list = ["Dinosaur", "Dinosaur:head"]
            if task in ['vis', 'ovis', 'ytvis19' ,'ytvis21', 'lvvis', 'uvo_video', 'rvos','ytbvos', 'coco_clip','obj365_clip','sa1b_clip','lvis_clip','openimage_clip','bdd_track_box','bdd_track_seg']:
                if 'spatial' in prompt_list:
                    (outputs, mask_dict), track_loss, dist_loss  = self.partglee.video_visualP(images, prompt_list, task, targets, batch_name_list)
                else:
                    (outputs, mask_dict), track_loss, dist_loss  = self.partglee(images, prompt_list, task, targets, batch_name_list)
                losses = self.criterion(outputs, targets, mask_dict, task)
                losses.update({"track_loss":track_loss})
                # if dist_loss is not None:
                losses.update({"dist_loss":dist_loss})
            else:
                if self.unify_object_part and task not in self.object_level_datasets:
                    targets = (object_targets, part_targets)
                    losses = self.partglee(images, prompt_list, task, targets, batch_name_list, is_train=True, criterion=self.criterion)
                else:
                    losses = self.partglee(images, prompt_list, task, targets, batch_name_list, is_train=True, criterion=self.criterion)
            
            for k in list(losses.keys()):
                if k in self.criterion.weight_dict:
                    losses[k] *= self.criterion.weight_dict[k]
                else:
                    # remove this loss if not specified in `weight_dict`
                    losses.pop(k)
                
                if ('box' in k or 'giou' in k) and task == 'grit':
                    losses[k] *= 0
            return losses
        
        else:  # testing
            torch.cuda.empty_cache()
            if task in ['vis', 'ovis', 'ytvis19' ,'ytvis21', 'lvvis', 'uvo_video','burst', 'tao_video']:
                return self.MinVIS_inference(batched_inputs, task)
            elif task in ['rvos']:
                self.inference_rvos(batched_inputs, prompt_list, task)
                return 
            elif task in ['omnilabel']:
                return self.omnilabel_inference(batched_inputs, task)
            else:
                custom_object_categories_idx = None
                custom_part_categories_idx = None
                images = self.preprocess_image(batched_inputs, task)
                if task in ['pascal_open_vocabulary', 'partimagenet'] and not self.training:
                    task = task + '_val'
                    batch_name_list = self.dataset_name_dicts[task]
                elif 'seginw' in task:
                    if task == 'seginw_House-Parts':
                        batch_name_list = ['object'] + [cat for cat in SEGINW_CATEGORIES[task]]
                    elif task == 'seginw_Airplane-Parts':
                        batch_name_list = [cat for cat in SEGINW_CATEGORIES[task]]
                    elif task == 'seginw_Bottles':
                        batch_name_list = [cat for cat in SEGINW_CATEGORIES[task]]
                    else:
                        batch_name_list = SEGINW_CATEGORIES[task]
                elif task == 'custom':
                    batch_name_list = custom_object_categories + custom_part_categories
                    custom_object_categories_idx = [idx for idx in range(len(custom_object_categories))]
                    custom_part_categories_idx = [idx + len(custom_object_categories) for idx in range(len(custom_part_categories))]
                else:
                    batch_name_list = self.dataset_name_dicts[task]
                # In inference phase partglee model only returns outputs
                
                outputs = self.partglee(images, prompt_list, task, batch_name_list=batch_name_list, is_train=False, custom_object_categories_idx=custom_object_categories_idx, custom_part_categories_idx=custom_part_categories_idx)

                mask_cls_results = outputs["pred_logits"]
                mask_pred_results = outputs["pred_masks"]
                mask_box_results = outputs["pred_boxes"]
                # upsample masks
                mask_pred_results = F.interpolate(
                    mask_pred_results,
                    size=(images.tensor.shape[-2], images.tensor.shape[-1]),
                    mode="bilinear",
                    align_corners=False,
                )
                del outputs
                # import pdb;pdb.set_trace()
                processed_results = []
                for mask_cls_result, mask_pred_result, mask_box_result, input_per_image, image_size in zip(
                    mask_cls_results, mask_pred_results, mask_box_results, batched_inputs, images.image_sizes
                ):
                    height = input_per_image.get("height", image_size[0])
                    width = input_per_image.get("width", image_size[1])
                    processed_results.append({})
                    new_size = mask_pred_result.shape[-2:]
                    if True:
                        if self.is_lsj:
                            resize_ratio = image_size[0]/max(height, width)
                            crop_size =  (int(height*resize_ratio), int(width*resize_ratio))
                        else:
                            crop_size = image_size
                        mask_pred_result = sem_seg_postprocess(
                            mask_pred_result, crop_size, height, width
                        )
                        mask_cls_result = mask_cls_result.to(mask_pred_result)

                    if self.semantic_on:
                        r = retry_if_cuda_oom(self.semantic_inference)(mask_cls_result, mask_pred_result)
                        if not self.sem_seg_postprocess_before_inference:
                            r = retry_if_cuda_oom(sem_seg_postprocess)(r, image_size, height, width)
                            processed_results[-1]["sem_seg"] = r
                        if task == 'ade20k' or task == 'pascalvoc':
                            # Generalized evaluation post-process
                            if task == 'ade20k':
                                test_obj_classes = ADE20K_OBJECT_CLASS_NAMES
                                test_classes = ADE20K_PART_CLASS_NAMES
                                ignore_label = 65535
                            elif task == 'pascalvoc':
                                test_obj_classes = PASCALVOC_OBJECT_CLASS_NAMES
                                test_classes = PASCALVOC_PART_CLASS_NAMES
                                ignore_label = 255
                            gt_objs = [test_obj_classes[i] for i in torch.unique(batched_inputs[0]["sem_seg"]) if i != ignore_label]
                            part_category_names = []
                            part_inds = []
                            for obj in gt_objs:
                                for i, part in enumerate(test_classes):
                                    if part.split('\'s')[0] == obj:
                                        part_category_names.append(part.replace('\'s', ''))
                                        part_inds.append(i)
                            no_part_ids = [i for i in range(len(test_classes)) if i not in part_inds]
                            preds = r
                            preds[no_part_ids] = 0.0
                            processed_results[-1]["sem_seg"] = preds
                        else:
                            processed_results[-1]["sem_seg"] = r

                    # panoptic segmentation inference
                    if self.panoptic_on:
                        panoptic_r = retry_if_cuda_oom(self.panoptic_inference)(mask_cls_result, mask_pred_result, task)
                        processed_results[-1]["panoptic_seg"] = panoptic_r
                        
                    # instance segmentation inference
                    if self.instance_on:
                        mask_box_result = mask_box_result.to(mask_pred_result)
                        height = new_size[0]/crop_size[0]*height
                        width = new_size[1]/crop_size[1]*width
                        mask_box_result = self.box_postprocess(mask_box_result, height, width)
                        instance_r = retry_if_cuda_oom(self.instance_inference)(mask_cls_result, mask_pred_result, mask_box_result, task)
                        processed_results[-1]["instances"] = instance_r
                        
                return processed_results
 

    def prepare_targets(self, batched_inputs, targets, images, prompt_list, task):
        img_long_size = max(images.tensor.shape[-2:])  #以0.4的概率把video数据集进入prompt模式
        if img_long_size<1000  and  np.random.rand() > 0.6 and self.visaul_prompt and task in ['ovis', 'ytvis19' ,'ytvis21', 'uvo_video', 'bdd_track_seg', 'coco_clip','sa1b_clip','lvis_clip']:  # switch into visual prompt mode
            if task in ['ovis', 'ytvis19' ,'ytvis21', 'uvo_video', 'sa1b_clip','coco_clip','lvis_clip','bdd_track_seg']:
                all_first_frame_num_objs = [(targets_i[0].gt_ids != -1).sum() for targets_i in targets]
                all_first_frame_num_objs = torch.stack(all_first_frame_num_objs)>0
                if all_first_frame_num_objs.all(): # each clip has a valid object in first frame
                    prompt_flag = True
                    prompt_list["spatial"] = []
                else:
                    prompt_flag = False
            else:
                prompt_flag = False

        else:
            prompt_flag = False

        if task in ['ytbvos']:
            all_first_frame_num_objs = [(targets_i[0].gt_ids != -1).sum() for targets_i in targets]
            all_first_frame_num_objs = torch.stack(all_first_frame_num_objs)>0
            if img_long_size<1000 and all_first_frame_num_objs.all(): # each clip has a valid object in first frame
                prompt_flag = True
                prompt_list["spatial"] = []
            else:
                prompt_flag = False
        if task in  ['vis', 'ovis', 'ytvis19' ,'ytvis21', 'uvo_video', 'rvos', 'coco_clip','obj365_clip','sa1b_clip','lvis_clip','openimage_clip', 'ytbvos','bdd_track_box','bdd_track_seg']:
           
            video_targets  = []
            for batched_inputs_i, targets_i in zip(batched_inputs, targets):
                h_pad, w_pad = images.tensor.shape[-2:]
                new_targets = []


                if prompt_flag :  # first frame has object
                    first_frame_valid_num = (targets_i[0].gt_ids != -1).sum().item() # 第一帧的有效物体数
                    if first_frame_valid_num==0:
                        import pdb;pdb.set_trace()
                    assert first_frame_valid_num>0

                    # num_prompts = random.randint(1, min(first_frame_valid_num,5) ) # keep random objects
                    num_prompts = 1 # 给定一个prompt，只分割一个物体
                    sample_idx = random.sample(list(range(0, first_frame_valid_num)), num_prompts)  #sample index for this video
                    visualP_ids = targets_i[0].gt_ids[targets_i[0].gt_ids != -1][sample_idx].to(self.device)
                
                for frame_i, targets_per_image in enumerate(targets_i):
                    targets_per_image = targets_per_image.to(self.device)
                    if  'gt_masks' not in targets_per_image._fields.keys():
                        padded_masks = None
                    else:
                        gt_masks = targets_per_image.gt_masks.tensor
                        padded_masks = torch.zeros((gt_masks.shape[0], h_pad, w_pad), dtype=gt_masks.dtype, device=gt_masks.device)
                        padded_masks[:, : gt_masks.shape[1], : gt_masks.shape[2]] = gt_masks
                    gt_classes = targets_per_image.gt_classes
                    
                    image_size_xyxy = torch.as_tensor([w_pad, h_pad, w_pad, h_pad], dtype=torch.float, device=self.device)
                    gt_boxes = box_ops.box_xyxy_to_cxcywh(targets_per_image.gt_boxes.tensor)/image_size_xyxy
                    gt_boxes = torch.clamp(gt_boxes,0,1)

                    inst_ids = targets_per_image.gt_ids
                    if prompt_flag : # inst_id 中不在 visualP_ids 里的全部置为-1 被mask掉
                        not_in_prompt = [ inst_ids_i not in visualP_ids for inst_ids_i in inst_ids]
                        inst_ids[not_in_prompt] = -1
                        
                    valid_id = inst_ids!=-1  # if a object is disappeared，its gt_ids is -1
                    if 'ori_id' in targets_per_image._fields.keys():
                        ori_id = [int(oriid) for oriid in targets_per_image.ori_id]
                    else:
                        ori_id = None

                    if padded_masks is None:
                        video_targets.append(
                        {
                            "labels": gt_classes[valid_id],
                            'inst_id':inst_ids[valid_id],
                            "masks": None,
                            "boxes":gt_boxes[valid_id],
                            "ori_id":ori_id,
                        }
                        )
                    else:
                        video_targets.append(
                            {
                                "labels": gt_classes[valid_id],
                                'inst_id':inst_ids[valid_id],
                                "masks": padded_masks[valid_id],
                                "boxes":gt_boxes[valid_id],
                                "ori_id":ori_id,
                            }
                        )
                    if prompt_flag and frame_i==0:
                        # import pdb;pdb.set_trace()
                        prompt_list["spatial"].append(padded_masks[valid_id]) # add the first frame gt mask as visual prompt
    
            return video_targets, prompt_list
        
        h_pad, w_pad = images.tensor.shape[-2:]
        new_targets = []

        if np.random.rand() > 0.8 and self.visaul_prompt and task in ['coco', 'sa1b', 'UVO_image', 'image_yt19', 'image_yt21', 'image_o']:  # switch into visual prompt mode
            prompt_flag = True
            prompt_list["spatial"] = []
        else:
            prompt_flag = False
        for targets_per_image in targets:
            #h, w = targets_per_image.image_size   
            #image_size_xyxy = torch.as_tensor([w, h, w, h], dtype=torch.float, device=self.device)
            if task in ['obj365','bdd_det', 'bdd_track_box','openimage','vg', 'vg_joint', 'grit', 'partimagenet_parsed', 'pascal_part_parsed',] and 'gt_masks' not in targets_per_image._fields.keys():
                padded_masks = None
            else:
                if isinstance(targets_per_image.gt_masks, torch.Tensor):
                    gt_masks = targets_per_image.gt_masks
                else:
                    gt_masks = targets_per_image.gt_masks.tensor
                # pad gt
                padded_masks = torch.zeros((gt_masks.shape[0], h_pad, w_pad), dtype=gt_masks.dtype, device=gt_masks.device)
                padded_masks[:, : gt_masks.shape[1], : gt_masks.shape[2]] = gt_masks
            gt_classes = targets_per_image.gt_classes
            #gt_boxes = box_ops.box_xyxy_to_cxcywh(targets_per_image.gt_boxes.tensor)/image_size_xyxy
            # generate random visual prompt and delet un-selected gt, only keep the prompt ones
            image_size_xyxy = torch.as_tensor([w_pad, h_pad, w_pad, h_pad], dtype=torch.float, device=self.device)
            gt_boxes = box_ops.box_xyxy_to_cxcywh(targets_per_image.gt_boxes.tensor)/image_size_xyxy
            gt_boxes = torch.clamp(gt_boxes,0,1)

            if prompt_flag and len(gt_classes)>0:
                # num_prompts = random.randint(1,len(gt_classes))
                num_prompts = 1 # 给定一个prompt，只分割一个物体
                sample_ids = random.sample(list(range(0, len(gt_classes))), num_prompts)
                if padded_masks is not None:
                    padded_masks = padded_masks[sample_ids]
                gt_classes = gt_classes[sample_ids]
                gt_boxes = gt_boxes[sample_ids]
            else:
                if prompt_flag:
                    prompt_flag = False
                    prompt_list.pop("spatial")
                
            new_targets.append(
                {
                    "labels": gt_classes,
                    "masks": padded_masks,
                    "boxes":gt_boxes,
                }
            )
            if prompt_flag:
                prompt_list["spatial"].append(padded_masks)
        return new_targets, prompt_list

    def prepare_targets_object_part(self, batched_inputs, targets, images, task):
        h_pad, w_pad = images.tensor.shape[-2:]
        object_targets = []
        part_targets = []
        
        for targets_per_image in targets:
            if task in ['openimage_joint', 'partimagenet_parsed', 'pascal_part_parsed',] and 'gt_masks' not in targets_per_image._fields.keys():
                padded_masks = None
            else:
                if isinstance(targets_per_image.gt_masks, torch.Tensor):
                    gt_masks = targets_per_image.gt_masks
                else:
                    gt_masks = targets_per_image.gt_masks.tensor
                # pad gt
                padded_masks = torch.zeros((gt_masks.shape[0], h_pad, w_pad), dtype=gt_masks.dtype, device=gt_masks.device)
                padded_masks[:, : gt_masks.shape[1], : gt_masks.shape[2]] = gt_masks

            # Divide Object and Part indices
            if task in ['sa1b_joint', 'paco', 'ade20k', 'ade20k_base', 'pascalvoc', 'pascalvoc_base', 'pascal_joint', 'partimagenet_joint', 'pascal_open_vocabulary', 'coco_panoptic', 'openimage_joint',] and self.unify_object_part:
                object_level_task = task + '_object'
                part_level_task = task + '_part'
                object_category_indices = torch.tensor(self.object_category_index_mapper[object_level_task]).to(targets_per_image.gt_classes.device)
                part_category_indices = torch.tensor(self.part_category_index_mapper[part_level_task]).to(targets_per_image.gt_classes.device)
            else:
                object_category_indices = None 
            # Prepare Object targets and Part targets seperately
            gt_classes = targets_per_image.gt_classes
            #gt_boxes = box_ops.box_xyxy_to_cxcywh(targets_per_image.gt_boxes.tensor)/image_size_xyxy
            # generate random visual prompt and delet un-selected gt, only keep the prompt ones
            image_size_xyxy = torch.as_tensor([w_pad, h_pad, w_pad, h_pad], dtype=torch.float, device=self.device)
            gt_boxes = box_ops.box_xyxy_to_cxcywh(targets_per_image.gt_boxes.tensor)/image_size_xyxy
            gt_boxes = torch.clamp(gt_boxes,0,1)
            if task in ['partimagenet_parsed', 'pascal_part_parsed'] and object_category_indices is None:
                object_targets.append(
                    {
                        "labels": torch.empty(size=(0, 1), device=gt_boxes.device),
                        "masks": None,
                        "boxes": torch.empty(size=(0, 4), device=gt_boxes.device)
                    }
                )
                part_targets.append(
                    {
                        "labels": gt_classes,
                        "masks": padded_masks,
                        "boxes":gt_boxes,
                    }
                )
            elif object_category_indices is None:
                object_targets.append(
                    {
                        "labels": gt_classes,
                        "masks": padded_masks,
                        "boxes":gt_boxes,
                    }
                )
            else:
                object_category_mask = torch.isin(gt_classes, object_category_indices)
                # part_category_mask = ~object_category_mask
                part_category_mask = torch.isin(gt_classes, part_category_indices)
                if task in ['openimage_joint']:
                    mapped_object_classes = torch.tensor([self.openimage_joint_object_mapping[idx] for idx in gt_classes[object_category_mask].tolist()]).to(gt_classes.device)
                else:
                    mapped_object_classes = gt_classes[object_category_mask]
                if padded_masks is not None:
                    object_targets.append(
                        {
                            "labels": mapped_object_classes,
                            "masks": padded_masks[object_category_mask],
                            "boxes": gt_boxes[object_category_mask],
                        }
                    )
                else:
                    object_targets.append(
                        {
                            "labels": mapped_object_classes,
                            "masks": None,
                            "boxes": gt_boxes[object_category_mask],
                        }
                    )
                # In order to decouple object and part, here we apply a mapping for part labels to transform it into 0-N
                if task in ['openimage_joint']:
                    mapped_part_classes = torch.tensor([self.openimage_joint_part_mapping[idx] for idx in gt_classes[part_category_mask].tolist()]).to(gt_classes.device)
                else:
                    mapped_part_classes = gt_classes[part_category_mask] - len(object_category_indices)
                if padded_masks is not None:
                    part_targets.append(
                        {
                            "labels": mapped_part_classes,
                            "masks": padded_masks[part_category_mask],
                            "boxes": gt_boxes[part_category_mask],
                        }
                    )
                else:
                    part_targets.append(
                        {
                            "labels": mapped_part_classes,
                            "masks": None,
                            "boxes": gt_boxes[part_category_mask],
                        }
                    )
        return object_targets, part_targets

    def preprocess_video(self, batched_inputs):
        """
        Normalize, pad and batch the input images.
        """
        images = []
        for video in batched_inputs:
            for frame in video["image"]:
                images.append(self.normalizer(frame.to(self.device)))
        images = ImageList.from_tensors(images,size_divisibility=self.size_divisibility)
        return images

    def preprocess_image(self, batched_inputs, task):
        """
        Normalize, pad and batch the input images.
        """
        if task in ['vis', 'ovis', 'ytvis19' ,'ytvis21','lvvis', 'uvo_video', 'burst', 'rvos', 'ytbvos', 'coco_clip','obj365_clip','sa1b_clip','lvis_clip','openimage_clip','bdd_track_box','bdd_track_seg']:
            return self.preprocess_video(batched_inputs)  #[bz [frame]]
        images = [self.normalizer(x["image"].to(self.device)) for x in batched_inputs]
        images = ImageList.from_tensors(images,size_divisibility=self.size_divisibility)
        return images

    def instance_inference(self, mask_cls, mask_pred, mask_box_result, task):
        # mask_pred is already processed to have the same shape as original input
        image_size = mask_pred.shape[-2:]
        if task == 'grounding':
            max_inst = 1
            prob = mask_cls.sigmoid()
            topk_values, topk_indexes = torch.topk(prob.view(-1), max_inst, dim=0)
            scores = topk_values
            topk_boxes = torch.div(topk_indexes, mask_cls.shape[1], rounding_mode='floor')
            labels = topk_indexes % mask_cls.shape[1]
            scores_per_image = scores
            labels_per_image = labels
            mask_pred = mask_pred[topk_boxes]
            mask_box_result = mask_box_result[topk_boxes]
            
        elif task == 'sa1b':
            max_inst = 100
            prob = mask_cls.sigmoid()
            topk_values, topk_indexes = torch.topk(prob.view(-1), max_inst, dim=0)
            scores = topk_values
            topk_boxes = torch.div(topk_indexes, mask_cls.shape[1], rounding_mode='floor')
            labels = topk_indexes % mask_cls.shape[1]
            scores_per_image = scores
            labels_per_image = labels
            mask_pred = mask_pred[topk_boxes]
            mask_box_result = mask_box_result[topk_boxes]
        else: 
            # [Q, K]
            if task in ['lvis', 'image_tao', 'image_bur', 'paco'] and self.visualize == False:
                self.test_topk_per_image = 300
            scores = mask_cls.sigmoid()  # [100, 80]
            if task == 'custom':
                labels = torch.arange(mask_cls.shape[1], device=self.device).unsqueeze(0).repeat(self.num_queries, 1).flatten(0, 1)
            else:
                labels = torch.arange(self.num_class[task], device=self.device).unsqueeze(0).repeat(self.num_queries, 1).flatten(0, 1)
            scores_per_image, topk_indices = scores.flatten(0, 1).topk(self.test_topk_per_image, sorted=False)  # select 100
            labels_per_image = labels[topk_indices]
            
            if task == 'custom':
                topk_indices = topk_indices // mask_cls.shape[1]
            else:
                topk_indices = topk_indices // self.num_class[task]
            mask_pred = mask_pred[topk_indices]
            
            # if this is panoptic segmentation, we only keep the "thing" classes
            if self.panoptic_on:
                keep = torch.zeros_like(scores_per_image).bool()
                for i, lab in enumerate(labels_per_image):
                    keep[i] = lab in self.metadata.thing_dataset_id_to_contiguous_id.values()
                scores_per_image = scores_per_image[keep]
                labels_per_image = labels_per_image[keep]
                mask_pred = mask_pred[keep]      
            mask_box_result = mask_box_result[topk_indices]      
        result = Instances(image_size)
        # mask (before sigmoid)
        result.pred_masks = (mask_pred > 0).float()
        # half mask box half pred box
        # mask_box_result = mask_box_result[topk_indices]
        if self.panoptic_on:
            mask_box_result = mask_box_result[keep]
        # result.pred_boxes = Boxes(torch.zeros(mask_pred.size(0), 4))
        # Uncomment the following to get boxes from masks (this is slow)
        # result.pred_boxes = BitMasks(mask_pred > 0).get_bounding_boxes()
        result.pred_boxes = Boxes(mask_box_result)
        
        # Calculate average mask prob
        mask_scores_per_image = (mask_pred.sigmoid().flatten(1) * result.pred_masks.flatten(1)).sum(1) / (result.pred_masks.flatten(1).sum(1) + 1e-6)
        result.scores = scores_per_image * mask_scores_per_image
        result.pred_classes = labels_per_image
        return result
    
    def semantic_inference(self, mask_cls, mask_pred):
        # if use cross-entropy loss in training, evaluate with softmax
        if self.semantic_ce_loss:
            mask_cls = F.softmax(mask_cls, dim=-1)[..., :-1]
            mask_pred = mask_pred.sigmoid()
            semseg = torch.einsum("qc,qhw->chw", mask_cls, mask_pred)
            return semseg
        # if use focal loss in training, evaluate with sigmoid. As sigmoid is mainly for detection and not sharp
        # enough for semantic and panoptic segmentation, we additionally use use softmax with a temperature to
        # make the score sharper.
        else:
            T = self.pano_temp
            mask_cls = mask_cls.sigmoid()
            if self.transform_eval:
                mask_cls = F.softmax(mask_cls / T, dim=-1)  # already sigmoid
            mask_pred = mask_pred.sigmoid()
            semseg = torch.einsum("qc,qhw->chw", mask_cls, mask_pred)
            return semseg

    def panoptic_inference(self, mask_cls, mask_pred, task):
        # As we use focal loss in training, evaluate with sigmoid. As sigmoid is mainly for detection and not sharp
        # enough for semantic and panoptic segmentation, we additionally use use softmax with a temperature to
        # make the score sharper.
        prob = 0.5
        T = self.pano_temp
        scores, labels = mask_cls.sigmoid().max(-1)
        mask_pred = mask_pred.sigmoid()
        keep = labels.ne(self.num_class[task]) & (scores > self.object_mask_threshold)
        # added process
        if self.transform_eval:
            scores, labels = F.softmax(mask_cls.sigmoid() / T, dim=-1).max(-1)
        cur_scores = scores[keep]
        cur_classes = labels[keep]
        cur_masks = mask_pred[keep]
        cur_prob_masks = cur_scores.view(-1, 1, 1) * cur_masks

        h, w = cur_masks.shape[-2:]
        panoptic_seg = torch.zeros((h, w), dtype=torch.int32, device=cur_masks.device)
        segments_info = []

        current_segment_id = 0

        if cur_masks.shape[0] == 0:
            # We didn't detect any mask :(
            return panoptic_seg, segments_info
        else:
            # take argmax
            cur_mask_ids = cur_prob_masks.argmax(0)
            stuff_memory_list = {}
            for k in range(cur_classes.shape[0]):
                pred_class = cur_classes[k].item()
                isthing = pred_class in self.metadata.thing_dataset_id_to_contiguous_id.values()
                mask_area = (cur_mask_ids == k).sum().item()
                original_area = (cur_masks[k] >= prob).sum().item()
                mask = (cur_mask_ids == k) & (cur_masks[k] >= prob)

                if mask_area > 0 and original_area > 0 and mask.sum().item() > 0:
                    if mask_area / original_area < self.overlap_threshold:
                        continue

                    # merge stuff regions
                    if not isthing:
                        if int(pred_class) in stuff_memory_list.keys():
                            panoptic_seg[mask] = stuff_memory_list[int(pred_class)]
                            continue
                        else:
                            stuff_memory_list[int(pred_class)] = current_segment_id + 1

                    current_segment_id += 1
                    panoptic_seg[mask] = current_segment_id

                    segments_info.append(
                        {
                            "id": current_segment_id,
                            "isthing": bool(isthing),
                            "category_id": int(pred_class),
                        }
                    )

            return panoptic_seg, segments_info
    
    def box_postprocess(self, out_bbox, img_h, img_w):
        # postprocess box height and width
        boxes = box_ops.box_cxcywh_to_xyxy(out_bbox)
        scale_fct = torch.tensor([img_w, img_h, img_w, img_h])
        scale_fct = scale_fct.to(out_bbox)
        boxes = boxes * scale_fct
        return boxes

    def match_from_embds(self, tgt_embds, cur_embds):
        cur_embds = cur_embds / cur_embds.norm(dim=1)[:, None]
        tgt_embds = tgt_embds / tgt_embds.norm(dim=1)[:, None]
        cos_sim = torch.mm(cur_embds, tgt_embds.transpose(0,1))

        cost_embd = 1 - cos_sim

        C = 1.0 * cost_embd
        C = C.cpu()

        indices = linear_sum_assignment(C.transpose(0, 1))  # target x current
        indices = indices[1]  # permutation that makes current aligns to target

        return indices
    
    def MinVIS_inference(self, batched_inputs, task):
        video_len = len(batched_inputs[0]['file_names'])

        video_len = len(batched_inputs[0]['file_names'])
        clip_length = 5 # self.batch_infer_len
        batch_name_list = self.dataset_name_dicts[task]

        #split long video into clips to form a batch input 
        # if video_len > clip_length:
        num_clips = math.ceil(video_len/clip_length)
        logits_list, boxes_list, embed_list, points_list, masks_list = [], [], [], [], []
        for c in range(num_clips):
            start_idx = c*clip_length
            end_idx = (c+1)*clip_length
            clip_inputs = [{'image':batched_inputs[0]['image'][start_idx:end_idx]}]
            clip_images = self.preprocess_video(clip_inputs)
            # import pdb;pdb.set_trace()
            (clip_output,_),dist,loss = self.partglee(clip_images, {}, task,  batch_name_list = batch_name_list, is_train= False)
            logits_list.append(clip_output['pred_logits'])
            boxes_list.append(clip_output['pred_boxes'])
            embed_list.append(clip_output['pred_track_embed'])
            masks_list.append(clip_output['pred_masks']) #.to(self.merge_device)
            
        outputs = {
            'pred_logits':torch.cat(logits_list,dim=0).detach(),
            'pred_track_embed':torch.cat(embed_list,dim=0).detach(),
            'pred_masks':torch.cat(masks_list,dim=0).detach(),
            'pred_boxes': torch.cat(boxes_list,dim=0).detach(),
        }    

        # batch_name_list  = self.dataset_name_dicts[task]
        # images = self.preprocess_video(batched_inputs)
        # outputs,_ = self.partglee(images, {}, task,  batch_name_list = batch_name_list, is_train= False)

        pred_logits = list(torch.unbind(outputs['pred_logits']))
        pred_masks = list(torch.unbind(outputs['pred_masks'].cpu()))
        pred_embds = list(torch.unbind(outputs['pred_track_embed']))
        pred_boxes = list(torch.unbind(outputs['pred_boxes']))
        del outputs
        out_logits = []
        out_masks = []
        out_embds = []
        out_boxes = []
        out_logits.append(pred_logits[0])
        out_masks.append(pred_masks[0])
        out_embds.append(pred_embds[0])
        out_boxes.append(pred_boxes[0])
        memory_embedding = out_embds[-1]

        for i in range(1, len(pred_logits)):
            # indices = self.match_from_embds(memory_embedding, pred_embds[i])
            MA_embedding = torch.stack(out_embds[-3:]).mean(0)
            indices = self.match_from_embds(MA_embedding, pred_embds[i])
            out_logits.append(pred_logits[i][indices, :])
            out_masks.append(pred_masks[i][indices, :, :])
            out_embds.append(pred_embds[i][indices, :])
            out_boxes.append(pred_boxes[i][indices, :])
            score_weights = pred_logits[i][indices, :].sigmoid().max(-1)[0][:,None]
            memory_embedding = (memory_embedding+pred_embds[i][indices, :]*score_weights )/(1+score_weights)

        mask_cls_result = sum(out_logits)/len(out_logits)

        out_logits = torch.stack(out_logits, dim=1)  # q numc -> q t numc

        mask_pred_result = torch.stack(out_masks, dim=1) # q h w -> q t h w
        mask_box_result = torch.stack(out_boxes, dim=1) # q 4 -> q t 4
        first_resize_size = (clip_images.tensor.shape[-2], clip_images.tensor.shape[-1])

        input_per_image = batched_inputs[0]
        image_size = clip_images.image_sizes[0]  # image size without padding after data augmentation

        height = input_per_image.get("height", image_size[0])  # raw image size before data augmentation
        width = input_per_image.get("width", image_size[1])
        mask_box_result = self.box_postprocess(mask_box_result, height, width)

        return self.minvis_inference_video(mask_cls_result, mask_pred_result, mask_box_result, image_size, height, width, first_resize_size, task, out_logits, batched_inputs)


    def minvis_inference_video(self, mask_cls, mask_pred, mask_box_result, img_size, output_height, output_width, first_resize_size, task, ori_logits, batched_inputs):
        if task != 'tao_video':
            if len(mask_cls) > 0:
                # import pdb;pdb.set_trace()            
                # keep top-20 predictions

                scores = mask_cls.sigmoid()  # [300, 40]

                num_class = self.num_class[task]
                labels = torch.arange(num_class, device=self.device).unsqueeze(0).repeat(self.num_queries, 1).flatten(0, 1)
                scores_per_image, topk_indices = scores.flatten(0, 1).topk(50, sorted=False)  # select 20
                labels_per_image = labels[topk_indices]
                topk_indices = topk_indices // num_class
                mask_pred = mask_pred[topk_indices]
                mask_box_result = mask_box_result[topk_indices]

                pred_masks = F.interpolate(
                    mask_pred, size=first_resize_size, mode="bilinear", align_corners=False
                )

                pred_masks = pred_masks[:, :, : img_size[0], : img_size[1]]
                pred_masks = F.interpolate(
                    pred_masks, size=(output_height, output_width), mode="bilinear", align_corners=False
                )

                masks = pred_masks > 0.

                out_scores = scores_per_image.tolist()
                out_labels = labels_per_image.tolist()
                out_masks = [m for m in masks.cpu()]
                # import pdb;pdb.set_trace()

                mask_box_result[:,:,2]=mask_box_result[:,:,2]-mask_box_result[:,:,0]
                mask_box_result[:,:,3]=mask_box_result[:,:,3]-mask_box_result[:,:,1]

                # xyxy2 xywh
                mask_box_result = mask_box_result.cpu().long()
                out_boxes = [m for m in mask_box_result]
                # import pdb;pdb.set_trace()
            else:
                out_scores = []
                out_labels = []
                out_masks = []
                out_boxes = []

            video_output = {
                "image_size": (output_height, output_width),
                "pred_scores": out_scores,
                "pred_labels": out_labels,
                "pred_masks": out_masks,
                "pred_boxes":out_boxes,
            }
        else:  # for TAO video  teta metric
            scores = mask_cls.sigmoid()  # [300, numcls]

            topk_num = 50

            num_class = self.num_class[task]

            
            scores_per_video, topk_indices = scores.max(-1)[0].topk(topk_num, sorted=False)  # select 20
            labels_per_video =  scores[topk_indices].max(-1)[1]  # [select_num]

            # import pdb;pdb.set_trace()

            mask_pred = mask_pred[topk_indices]  #[select, len, H, W]
            mask_pred = mask_pred>0

            mask_box_result = mask_box_result[topk_indices]  #[slelct_num, len, 4]
             # xyxy2 xywh
            mask_box_result[:,:,2]=mask_box_result[:,:,2]-mask_box_result[:,:,0]
            mask_box_result[:,:,3]=mask_box_result[:,:,3]-mask_box_result[:,:,1]

            ori_logits = ori_logits[topk_indices].sigmoid()    #[slelct_num, len, num_class]
            
            image_ids = batched_inputs[0]['image_ids']
            video_id = batched_inputs[0]['video_id']
            video_len = len(image_ids)
            track_ids = torch.arange(topk_num).to(scores_per_video) + topk_num*video_id


            video_results = []
            for i,image_id in enumerate(image_ids):
                
                # frame_logits = ori_logits[:,i] # [topk_num,nun_cls]
                # scores_per_frame, labels_per_frames = frame_logits.max(-1)

                frame_boxes = mask_box_result[:,i]  
                frame_masks = mask_pred[:,i]  
                mask_valid = frame_masks.flatten(1,2).sum(-1)>5

                frame_boxes = frame_boxes[mask_valid]
                frame_scores = scores_per_video[mask_valid]
                frame_labels = labels_per_video[mask_valid]
                frame_trackids = track_ids[mask_valid]

                # box nms
                boxes_before_nms = box_ops.box_cxcywh_to_xyxy(frame_boxes)
                keep_indices = ops.nms(boxes_before_nms,frame_scores,0.5)#.tolist()

                frame_boxes = frame_boxes[keep_indices]
                frame_scores = frame_scores[keep_indices]
                frame_labels = frame_labels[keep_indices]
                frame_trackids = frame_trackids[keep_indices]

                
                for box,score,label,trackid in zip(frame_boxes,frame_scores,frame_labels,frame_trackids):
                    video_results.append(
                        {
                            "image_id" : image_id,
                            "category_id" : label.item(),
                            "bbox" : box.tolist(),
                            "score" : score.item(),
                            "track_id": trackid.item(),
                            "video_id": video_id
                        }
                    )

            video_output = video_results

        return video_output



    def inference_rvos(self, batched_inputs, prompt_list, task):
        import cv2

        # import pdb;pdb.set_trace()
        images = self.preprocess_video(batched_inputs)
        (outputs,_),_d,_t  = self.partglee(images, prompt_list, task, is_train= False)

        pred_logits = list(torch.unbind(outputs['pred_logits']))
        pred_masks = list(torch.unbind(outputs['pred_masks']))

        for frame_idx, (mask_pred_result, mask_cls_result) in  enumerate(zip(pred_masks, pred_logits)):
            
            prob = mask_cls_result.sigmoid().max(-1)[0]
            topk_values, topk_indexes = torch.topk(prob, 1, dim=0)
            mask_pred_result = mask_pred_result[topk_indexes]

            mask_pred_result = mask_pred_result.unsqueeze(0)
            
            mask_pred_result = F.interpolate(
                mask_pred_result,
                size=(images.tensor.shape[-2], images.tensor.shape[-1]),
                mode="bilinear",
                align_corners=False,
            )
            
            image_size = images.image_sizes
            height = batched_inputs[0].get("height", image_size[0])
            width = batched_inputs[0].get("width", image_size[1])
            mask_pred_result = sem_seg_postprocess(
                mask_pred_result, image_size[0], height, width
            )

            final_mask = mask_pred_result[0]>0
            final_mask = final_mask.cpu().numpy()
            video_name, exp_id = batched_inputs[0]["video"], batched_inputs[0]["exp_id"]
            # save binary image
            save_path = os.path.join(self.save_path_prefix, video_name, exp_id)
            if not os.path.exists(save_path):
                os.makedirs(save_path)
            frame_name = batched_inputs[0]["file_names"][frame_idx].split("/")[-1].replace(".jpg", "")
            mask = final_mask.astype(np.float32) 
            mask = Image.fromarray(mask * 255).convert('L')
            save_file = os.path.join(save_path, frame_name + ".png")
            mask.save(save_file)
        return 