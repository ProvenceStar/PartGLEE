#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates.

"""
PartGLEE Training Script.
"""

import os
import sys
import itertools
import time
from typing import Any, Dict, List, Set
import yaml
import torch

import detectron2.utils.comm as comm
from detectron2.checkpoint import DetectionCheckpointer
from detectron2.config import get_cfg
from detectron2.data import MetadataCatalog
from detectron2.engine import DefaultTrainer, default_argument_parser, default_setup, launch
from detectron2.evaluation import COCOEvaluator, verify_results, DatasetEvaluators, LVISEvaluator, SemSegEvaluator, COCOPanopticEvaluator
from detectron2.solver.build import maybe_add_gradient_clipping
from detectron2.projects.partglee import add_partglee_config, build_detection_train_loader, build_detection_test_loader
from detectron2.projects.partglee.backbone.eva01 import get_vit_lr_decay_rate
from detectron2.projects.partglee.data import (
    get_detection_dataset_dicts, RefCOCODatasetMapper, DetrDatasetMapper, YTVISDatasetMapper, Joint_Image_LSJDatasetMapper,\
    COCOInstanceNewBaselineDatasetMapper, COCO_CLIP_DatasetMapper,UnivideoimageDatasetMapper, UnivideopseudoDatasetMapper, PACOEvaluator, \
    PascalPartNewBaselineDatasetMapper, PASCALPARTEvaluator, JointNewBaselineDatasetMapper, COCOPanopticNewBaselineDatasetMapper, SemanticObjPartDatasetMapper, \
    InstanceSegEvaluator, SeginWDatasetMapper
)
from detectron2.projects.partglee.data.ade20k_evaluation import GeneralizedSemSegEvaluator
from detectron2.projects.partglee.data.ade20k_evaluation.utils.post_process_utils import dense_crf_post_process
from detectron2.projects.partglee.data import build_custom_train_loader

os.environ['CUDA_LAUNCH_BLOCKING'] = '1'

class Trainer(DefaultTrainer):
    """
    Extension of the Trainer class adapted.
    """

    @classmethod
    def build_evaluator(cls, cfg, dataset_name, output_folder=None):
        """
        Create evaluator(s) for a given dataset.
        This uses the special metadata "evaluator_type" associated with each builtin dataset.
        For your own dataset, you can simply create an evaluator manually in your
        script and do not have to worry about the hacky if-else logic here.
        """
        if output_folder is None:
            output_folder = os.path.join(cfg.OUTPUT_DIR, "inference")
            os.makedirs(output_folder, exist_ok=True)
        evaluator_list = []
        # import pdb;pdb.set_trace()
        evaluator_type = MetadataCatalog.get(dataset_name).evaluator_type
        if evaluator_type == "lvis":
            evaluator_list.append(LVISEvaluator(dataset_name, cfg, True, output_folder))    
        if evaluator_type == "paco":
            evaluator_list.append(PACOEvaluator(dataset_name, cfg, True, output_folder))
        if evaluator_type == "pascal_part":
            evaluator_list.append(PASCALPARTEvaluator(dataset_name, cfg, True, output_folder))
        if evaluator_type in [
            "coco_panoptic_seg",
            "ade20k_panoptic_seg",
            "cityscapes_panoptic_seg",
            "mapillary_vistas_panoptic_seg",
        ]:
            if cfg.MODEL.MaskDINO.TEST.PANOPTIC_ON:
                evaluator_list.append(COCOPanopticEvaluator(dataset_name, output_folder))
        # COCO
        if evaluator_type == "coco_panoptic_seg" and cfg.MODEL.MaskDINO.TEST.INSTANCE_ON:
            evaluator_list.append(COCOEvaluator(dataset_name, output_dir=output_folder))
        # SEGINW
        if evaluator_type == "seginw":
            evaluator_list.append(InstanceSegEvaluator(dataset_name, output_dir=output_folder))
        if evaluator_type == "coco_panoptic_seg" and cfg.MODEL.MaskDINO.TEST.SEMANTIC_ON:
            evaluator_list.append(SemSegEvaluator(dataset_name, distributed=True, output_dir=output_folder))
        elif evaluator_type == "coco":
            if "objects365" in dataset_name or "openimage" in dataset_name or "bdd_det" in dataset_name:
                force_tasks = {"bbox"}
            else:
                force_tasks = None
            # import pdb;pdb.set_trace()
            if "refcoco" in dataset_name:
                evaluator_list.append(COCOEvaluator(dataset_name, cfg, True, output_folder, force_tasks=force_tasks, refcoco=True))
            # elif "coco" in dataset_name or "objects365" in dataset_name or "sa1b" in dataset_name or "UVO" in dataset_name:
            else:
                evaluator_list.append(COCOEvaluator(dataset_name, cfg, True, output_folder, force_tasks=force_tasks, refcoco=False))

        if evaluator_type in ["sem_seg"]:
            evaluator_list.append(GeneralizedSemSegEvaluator(dataset_name, distributed=True, output_dir=output_folder, post_process_func=dense_crf_post_process if cfg.TEST.DENSE_CRF else None,))
            
        if len(evaluator_list) == 0:
            raise NotImplementedError(
                "no Evaluator for the dataset {} with the type {}".format(
                    dataset_name, evaluator_type
                )
            )
        elif len(evaluator_list) == 1:
            return evaluator_list[0]
        return DatasetEvaluators(evaluator_list)

    @classmethod
    def build_train_loader(cls, cfg):
        if cfg.DATALOADER.SAMPLER_TRAIN == "MultiDatasetSampler":
            # multiple datasets (for example, detection & grounding)
            # import pdb;pdb.set_trace()
            datasetname = ' '.join(list(cfg.DATASETS.TRAIN))
            # if 'ytvis' in datasetname:
            #     mapper = YTVISDatasetMapper(cfg, is_train=True)
            # else:
            #     mapper = RefCOCODatasetMapper(cfg, is_train=True)
            if cfg.MODEL.PSEUDO_VIDEO:   
                mapper = UnivideopseudoDatasetMapper(cfg, is_train=True)   
            else:  
                if cfg.INPUT.DATASET_MAPPER_NAME == 'coco_instance_lsj':
                    mapper = Joint_Image_LSJDatasetMapper(cfg, is_train=True)
                elif cfg.INPUT.DATASET_MAPPER_NAME == 'joint_lsj':
                    mapper = JointNewBaselineDatasetMapper(cfg, is_train=True)
                elif cfg.INPUT.DATASET_MAPPER_NAME == 'joint_image_lsj':
                    mapper = Joint_Image_LSJDatasetMapper(cfg, is_train=True)
                else:
                    # mapper = RefCOCODatasetMapper(cfg, is_train=True)
                    mapper = UnivideoimageDatasetMapper(cfg, is_train=True)   
            # elif 'vg' in datasetname:
            #     mapper = RefCOCODatasetMapper(cfg, is_train=True)
            data_loader = build_custom_train_loader(cfg, mapper=mapper)   
            return data_loader
        else:
            dataset_name = cfg.DATASETS.TRAIN[0]
            # if dataset_name.startswith('coco'):
            #     mapper = DetrDatasetMapper(cfg, is_train=True)
            # elif dataset_name.startswith('refcoco') or dataset_name.startswith('sa1b') :
            #     mapper = RefCOCODatasetMapper(cfg, is_train=True)
            # import pdb;pdb.set_trace()
            if cfg.MODEL.PSEUDO_VIDEO:
                mapper = COCO_CLIP_DatasetMapper(cfg, is_train=True)
            elif 'ytvis' in dataset_name or 'ovis' in dataset_name or 'video' in dataset_name or 'rvos' in dataset_name:
                mapper = YTVISDatasetMapper(cfg, is_train=True)
            else:
                # import pdb;pdb.set_trace()
                if cfg.INPUT.DATASET_MAPPER_NAME == 'coco_instance_lsj':
                    mapper = Joint_Image_LSJDatasetMapper(cfg, is_train=True)
                elif cfg.INPUT.DATASET_MAPPER_NAME == 'joint_image_lsj':
                    mapper = Joint_Image_LSJDatasetMapper(cfg, is_train=True)
                elif cfg.INPUT.DATASET_MAPPER_NAME == 'obj_part_semantic':
                    mapper = SemanticObjPartDatasetMapper(cfg, is_train=True)
                elif cfg.INPUT.DATASET_MAPPER_NAME == 'pascal_part_lsj':
                    mapper = PascalPartNewBaselineDatasetMapper(cfg, is_train=True)
                elif cfg.INPUT.DATASET_MAPPER_NAME == 'coco_panoptic_lsj':
                    mapper = COCOPanopticNewBaselineDatasetMapper(cfg, is_train=True)
                elif cfg.INPUT.DATASET_MAPPER_NAME == 'joint_lsj':
                    mapper = JointNewBaselineDatasetMapper(cfg, is_train=True)
                elif cfg.INPUT.DATASET_MAPPER_NAME == 'univideo_image_mapper':
                    mapper = UnivideoimageDatasetMapper(cfg, is_train=True)
                else:
                    mapper = RefCOCODatasetMapper(cfg, is_train=True)
                # if '365' in dataset_name or 'openimage' in dataset_name or 'vg' in dataset_name:
                #     mapper = RefCOCODatasetMapper(cfg, is_train=True)
                # else:
                #     mapper = Joint_Image_LSJDatasetMapper(cfg, is_train=True)
                
            # import pdb;pdb.set_trace()
            dataset_dict = get_detection_dataset_dicts(
                dataset_name,
                filter_empty=cfg.DATALOADER.FILTER_EMPTY_ANNOTATIONS,
                proposal_files=cfg.DATASETS.PROPOSAL_FILES_TRAIN if cfg.MODEL.LOAD_PROPOSALS else None,
            )

            return build_detection_train_loader(cfg, mapper=mapper, dataset=dataset_dict)

    @classmethod
    def build_test_loader(cls, cfg, dataset_name):
        if  'ytvis' in dataset_name or 'ovis' in dataset_name or 'video' in dataset_name or 'rvos' in dataset_name or 'ytbvos' in dataset_name or 'lvvis' in dataset_name:
            mapper = YTVISDatasetMapper(cfg, is_train=False)
        else:
            # if dataset_name.startswith('coco'):
            #     # mapper = CocoDatasetMapper(cfg, is_train=False)
            #     mapper = DetrDatasetMapper(cfg, is_train=False)
        # elif dataset_name.startswith('refcoco'):
            if cfg.INPUT.DATASET_MAPPER_NAME == 'coco_instance_lsj':
                mapper = Joint_Image_LSJDatasetMapper(cfg, is_train=False)
            elif cfg.MODEL.MaskDINO.TEST.OBJECT_PART_TEST_AUGMENTATIONS == True and cfg.MODEL.MaskDINO.TEST.ORACLE == True:
                mapper = SemanticObjPartDatasetMapper(cfg, is_train=False)
            # elif cfg.MODEL.MaskDINO.TEST.OBJECT_PART_TEST_AUGMENTATIONS == True and cfg.MODEL.MaskDINO.TEST.ORACLE == False:
            #     mapper = None
            elif 'seginw' in dataset_name:
                mapper = SeginWDatasetMapper(cfg, False)
                return build_detection_test_loader(cfg, dataset_name, mapper=mapper)
            elif dataset_name.startswith('lvis'):
                return build_detection_test_loader(cfg, dataset_name)
            else:
                mapper = RefCOCODatasetMapper(cfg, is_train=False)
        loader = build_detection_test_loader(cfg, dataset_name, mapper=mapper)
        
        return loader

    @classmethod
    def build_optimizer(cls, cfg, model):
        params: List[Dict[str, Any]] = []
        memo: Set[torch.nn.parameter.Parameter] = set()

        for key, value in model.named_parameters(recurse=True):
            if not value.requires_grad:
                continue
            # Avoid duplicating parameters
            if value in memo:
                continue
            memo.add(value)
            lr = cfg.SOLVER.BASE_LR
            weight_decay = cfg.SOLVER.WEIGHT_DECAY
            if "backbone" in key:
                lr = lr * cfg.SOLVER.BACKBONE_MULTIPLIER
                if cfg.SOLVER.LR_DECAY_RATE is not None:
                    backbone_decay = get_vit_lr_decay_rate(key, cfg.SOLVER.LR_DECAY_RATE, cfg.SOLVER.LR_DECAY_RATE_NUM_LAYERS)
                    lr = lr * backbone_decay
                    # print(key, ' lr decay=',backbone_decay)
            elif 'part' not in key:
                lr = lr * cfg.SOLVER.BACKBONE_MULTIPLIER
                    
            if "text_encoder" in key:
                lr = lr * cfg.SOLVER.TEXTENCODER_MULTIPLIER
            params += [{"params": [value], "lr": lr, "weight_decay": weight_decay}]

        def maybe_add_full_model_gradient_clipping(optim):  # optim: the optimizer class
            # detectron2 doesn't have full model gradient clipping now
            clip_norm_val = cfg.SOLVER.CLIP_GRADIENTS.CLIP_VALUE
            enable = (
                cfg.SOLVER.CLIP_GRADIENTS.ENABLED
                and cfg.SOLVER.CLIP_GRADIENTS.CLIP_TYPE == "full_model"
                and clip_norm_val > 0.0
            )

            class FullModelGradientClippingOptimizer(optim):
                def step(self, closure=None):
                    all_params = itertools.chain(*[x["params"] for x in self.param_groups])
                    torch.nn.utils.clip_grad_norm_(all_params, clip_norm_val)
                    super().step(closure=closure)

            return FullModelGradientClippingOptimizer if enable else optim

        optimizer_type = cfg.SOLVER.OPTIMIZER
        if optimizer_type == "SGD":
            optimizer = maybe_add_full_model_gradient_clipping(torch.optim.SGD)(
                params, cfg.SOLVER.BASE_LR, momentum=cfg.SOLVER.MOMENTUM
            )
        elif optimizer_type == "ADAMW":
            optimizer = maybe_add_full_model_gradient_clipping(torch.optim.AdamW)(
                params, cfg.SOLVER.BASE_LR
            )
        else:
            raise NotImplementedError(f"no optimizer type {optimizer_type}")
        if not cfg.SOLVER.CLIP_GRADIENTS.CLIP_TYPE == "full_model":
            optimizer = maybe_add_gradient_clipping(cfg, optimizer)
        return optimizer

def load_config_dict_to_opt(opt, config_dict):
    """
    Load the key, value pairs from config_dict to opt, overriding existing values in opt
    if there is any.
    """
    if not isinstance(config_dict, dict):
        raise TypeError("Config must be a Python dictionary")
    for k, v in config_dict.items():
        k_parts = k.split('.')
        pointer = opt
        for k_part in k_parts[:-1]:
            if k_part not in pointer:
                pointer[k_part] = {}
            pointer = pointer[k_part]
            assert isinstance(pointer, dict), "Overriding key needs to be inside a Python dict."
        ori_value = pointer.get(k_parts[-1])
        pointer[k_parts[-1]] = v
        # if ori_value:
        #     logger.warning(f"Overrided {k} from {ori_value} to {pointer[k_parts[-1]]}")


def load_opt_from_config_files(conf_file):
    """
    Load opt from the config files, settings in later files can override those in previous files.

    Args:
        conf_files: config file path

    Returns:
        dict: a dictionary of opt settings
    """
    opt = {}
    with open(conf_file, encoding='utf-8') as f:
        config_dict = yaml.safe_load(f)

    load_config_dict_to_opt(opt, config_dict)

    return opt

def setup(args):
    """
    Create configs and perform basic setups.
    """
    cfg = get_cfg()
    add_partglee_config(cfg)
    cfg.merge_from_file(args.config_file)
    cfg.merge_from_list(args.opts)
    cfg.freeze()  
    default_setup(cfg, args)
    return cfg


def main(args):
    cfg = setup(args)
    if args.eval_only:
        model = Trainer.build_model(cfg)
        DetectionCheckpointer(model, save_dir=cfg.OUTPUT_DIR).resume_or_load(cfg.MODEL.WEIGHTS, resume=args.resume)
        res = Trainer.test(cfg, model)
        if comm.is_main_process():
            verify_results(cfg, res)
        return res
    if_resume = args.resume
    if   hasattr(cfg, "HDFS_DIR") and cfg.HDFS_DIR.startswith('hdfs') and comm.get_local_rank()==0:
        os.system(f"mkdir -p {cfg.OUTPUT_DIR}")
        os.system(f'hdfs dfs -get {cfg.HDFS_DIR}/{cfg.OUTPUT_DIR}/last_checkpoint {cfg.OUTPUT_DIR}')
        last_checkpoint_path = os.path.join(cfg.OUTPUT_DIR, "last_checkpoint")
        if os.path.exists(last_checkpoint_path):
            if_resume = True
            with open(last_checkpoint_path, "r") as f:
                last_checkpoint = f.read()
                print("Downloading last_checkpoint:", last_checkpoint)
            os.system(f"hdfs dfs -get {cfg.HDFS_DIR}/{cfg.OUTPUT_DIR}/{last_checkpoint} {cfg.OUTPUT_DIR}")


    trainer = Trainer(cfg)
    # import pdb;pdb.set_trace()
    trainer.resume_or_load(resume=if_resume)
    return trainer.train()


if __name__ == "__main__":
    args = default_argument_parser().parse_args()
    print("Command Line Args:", args)
    launch(
        main,
        args.num_gpus,
        num_machines=args.num_machines,
        machine_rank=args.machine_rank,
        dist_url=args.dist_url,
        args=(args,),
    )
