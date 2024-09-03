from ._CONSTANTS import PACO_CATEGORIES, ADE20K_OBJECT_CLASS_NAMES, ADE20K_PART_CLASS_NAMES, PASCALVOC_OBJECT_CLASS_NAMES, \
                        PASCALVOC_PART_CLASS_NAMES, PARTIMAGENET_OBJECT_CATEGORIES, PARTIMAGENET_PART_CATEGORIES, PASCAL_OBJECT_CATEGORIES, PASCAL_PART_CATEGORIES, \
                        OPENIMAGES_OBJECT_CATEGORIES, ADE20K_OBJECT_BASE_CLASS_NAMES, PASCALVOC_OBJECT_BASE_CLASS_NAMES, PASCAL_VAL_OBJECT_CATEGORIES, \
                        OPENIMAGES_PART_CATEGORIES, ADE20K_BASE_CLASS_NAMES, PASCALVOC_OBJECT_BASE_CLASS_NAMES, PASCALVOC_BASE_CLASS_NAMES, PASCAL_VAL_OBJECT_CATEGORIES, PASCAL_VAL_PART_CATEGORIES

DATASET_SPECIFIC_CATEGORIES = {            
            'lvis': 1203,
            'paco': 531,
            'ade20k': 278,
            'ade20k_base': 209,
            'voc': 20,
            'pascalvoc': 136,
            'pascalvco_base': 89,
            'partimagenet_joint': 51,
            'sa1b_joint': 2,
            'pascal_joint': 113,
            'pascal_open_vocabulary': 89,
            'pascal_open_vocabulary_val': 93,
            'partimagenet_val': 40,
            'coco_panoptic': 133,
            'obj365':365,
            'openimage':601,
            'openimage_joint':601,
            'coco':80,
            'grounding': 1,
            'bdd_det':10,
            'bdd_inst':8,
            'ytvis19':40,
            'image_yt19':40,
            'ovis':25,
            'image_o':25,
            'ytvis21':40,
            'image_yt21':40,
            'lvvis': 1196,
            'uvo_video': 81,
            'burst': 482,
            'image_bur': 482,
            'image_tao': 302,
            'tao_video': 302,
            'seginw_House-Parts': 12,
            'seginw_Airplane-Parts': 5,
            'seginw_Bottles': 3,
            'seginw_Cows': 1,
            'seginw_Elephants': 1,
            'seginw_Brain-Tumor': 1,
            'seginw_Chicken': 1,
            'seginw_Rail': 1,
            'seginw_Electric-Shaver': 1,
            'seginw_Fruits': 5,
            'seginw_Garbage': 4,
            'seginw_Ginger-Garlic': 2,
            'seginw_Hand': 2,
            'seginw_Hand-Metal': 2,
            'seginw_HouseHold-Items': 12,
            'seginw_Nutterfly-Squireel': 2,
            'seginw_Phones': 1,
            'seginw_Poles': 1,
            'seginw_Puppies': 1,
            'seginw_Salmon-Fillet': 1,
            'seginw_Strawberry': 2,
            'seginw_Tablets': 1,
            'seginw_Toolkits': 8,
            'seginw_Trash': 22,
            'seginw_Watermelon': 1,
}

# We treat each part-level datasets into a object-level branch subset and a part-level branch subset
# Here we use '{dataset_name}_object' to denote the number object-level categories in the dataset, '{dataset_name}_part' to denote the number of part-level categories
CRITERION_NUM_CLASSES = {'coco':80, 'coconomask':80, 'coco_clip':80, 'obj365':100, 'obj365_clip':100, 'lvis':100, 'lvis_clip':100, \
                    'openimage':100, 'openimage_clip':100, 'grit':100, 'vg':200, 'grounding':1,  'ytbvos':1, 'rvos':1, 'sa1b':1, \
                    'sa1b_clip':1, 'ytvis19':40, 'image_yt19':40, 'ytvis21':40,'image_yt21':40,'ovis':25, 'image_o':25, 'uvo_video':81, \
                    'burst':1,'bdd_det':10, 'bdd_inst':8,'bdd_track_seg':8, 'bdd_track_box':8, 'paco': 531, 'partimagenet': 40, \
                    'coco_panoptic': 133, 'paco_object': 75, 'paco_part': 456, 'ade20k_object': 44, 'ade20k_part': 234, \
                    'pascalvoc_object': 20, 'pascalvoc_part': 116, 'sa1b_joint_object': 1, 'sa1b_joint_part': 1, \
                    'pascal_joint_object': 20, 'pascal_joint_part': 93, 'partimagenet_joint_object': 11, 'partimagenet_joint_part': 40, \
                    'pascal_open_vocabulary_object': 12, 'pascal_open_vocabulary_part': 77, 'coco_panoptic_object': 80, 'coco_panoptic_part': 53, \
                    'vg_joint_object': 200, 'vg_joint_part': 200, 'voc': 20, 'openimage_joint_object': 588, 'openimage_joint_part': 13, \
                    'ade20k_base_object': 33, 'ade20k_base_part': 176, 'pascalvoc_base_object': 15, 'pascalvoc_base_part': 74, \
                    'seginw_Brain-Tumor': 1, 'partimagenet_parsed_object': 11, 'partimagenet_parsed_part': 40, 'pascal_part_parsed_object': 20, 'pascal_part_parsed_part': 93
}

TEST_TOPK_PER_IMAGE = {
            'paco': 300,
            'ade20k': 100,
            'pascalvoc': 100,
            'partimagenet_joint': 100,
            'partimagenet_val': 100,
            'sa1b_joint': 11,   # vis
            'pascal_joint': 100,
            'pascal_open_vocabulary_val': 100,
            'coco': 100,
            'voc': 100,
            'lvis': 300,
            'grounding':1,
}

DATASET_SPECIFIC_OBJECT_NUMS = {
            'paco': 100,
            'ade20k': 0,
            'pascalvoc': 0,
            'partimagenet_joint': 5,
            'sa1b_joint': 1,    # vis
            'pascal_joint': 20,
            'pascal_open_vocabulary_val': 0,
            'partimagenet_val': 0,
            'coco': 100,
            'voc': 100,
            'lvis': 300,
            'seginw_Airplane-Parts': 5,
            'seginw_Bottles':90,
            'grounding': 1,
}

DATASET_SPECIFIC_PART_NUMS = {
            'paco': 200,
            'ade20k': 100,
            'pascalvoc': 100,
            'partimagenet_joint': 95,
            'sa1b_joint': 10,   # vis
            'pascal_joint': 80,
            'pascal_open_vocabulary_val': 100,
            'partimagenet_val': 100,
            'coco': 0,
            'voc': 0,
            'lvis': 0,
            'seginw_Airplane-Parts': 95,
            'seginw_Bottles': 10,
            'grounding': 0,
}

PART_DATASETS_OBJECT_CATEGORY_INDEX = {
            'paco_object': [idx for idx in range(len(PACO_CATEGORIES)) if PACO_CATEGORIES[idx]['supercategory']=='OBJECT'],
            'openimage_joint_object': [cat['id'] - 1 for cat in OPENIMAGES_OBJECT_CATEGORIES],
            'ade20k_object': [idx for idx in range(len(ADE20K_OBJECT_CLASS_NAMES))],
            'ade20k_base_object': [idx for idx in range(len(ADE20K_OBJECT_BASE_CLASS_NAMES))],
            'pascalvoc_object': [idx for idx in range(len(PASCALVOC_OBJECT_CLASS_NAMES))],
            'pascalvoc_base_object': [idx for idx in range(len(PASCALVOC_OBJECT_BASE_CLASS_NAMES))],
            'partimagenet_joint_object': [idx for idx in range(len(PARTIMAGENET_OBJECT_CATEGORIES))],
            'sa1b_joint_object': [0],
            'pascal_joint_object': [idx for idx in range(len(PASCAL_OBJECT_CATEGORIES))],
            'pascal_open_vocabulary_object': [idx for idx in range(12)],
            'pascal_open_vocabulary_val_object': [idx for idx in range(len(PASCAL_VAL_OBJECT_CATEGORIES))],
            'seginw_Airplane-Parts_object': [0],
            'seginw_Bottles_object': [0, 1],
}

PART_DATASETS_PART_CATEGORY_INDEX = {
            'paco_part': [idx for idx in range(len(PACO_CATEGORIES)) if PACO_CATEGORIES[idx]['supercategory']=='PART'],
            'openimage_joint_part': [cat['id'] - 1 for cat in OPENIMAGES_PART_CATEGORIES],
            'ade20k_part': [idx + 44 for idx in range(len(ADE20K_PART_CLASS_NAMES))],
            'ade20k_base_part': [idx + len(ADE20K_OBJECT_BASE_CLASS_NAMES) for idx in range(len(ADE20K_BASE_CLASS_NAMES))],
            'pascalvoc_part': [idx + 20 for idx in range(len(PASCALVOC_PART_CLASS_NAMES))],
            'pascalvoc_base_part': [idx + len(PASCALVOC_OBJECT_BASE_CLASS_NAMES) for idx in range(len(PASCALVOC_BASE_CLASS_NAMES))],
            'partimagenet_joint_part': [idx + 11 for idx in range(len(PARTIMAGENET_PART_CATEGORIES))],
            'sa1b_joint_part': [1],
            'pascal_joint_part': [idx + 20 for idx in range(len(PASCAL_PART_CATEGORIES))],
            'pascal_open_vocabulary_part': [idx + 12 for idx in range(77)],
            'pascal_open_vocabulary_val_part': [idx + len(PASCAL_VAL_OBJECT_CATEGORIES) for idx in range(len(PASCAL_VAL_PART_CATEGORIES))],
            'seginw_Airplane-Parts_part': [idx + 1 for idx in range(4)],
            'seginw_Bottles_part': [2],
}

DATASET_CATEGORY_SET = {
            'obj365': set(list(range(365))),
            'openimage': set(list(range(601))),
            'openimage_joint': set(list(range(601))),
            'lvis': set(list(range(1203))),
            'paco': set(list(range(531))),
            'ade20k': set(list(range(278))),
            'ade20k_base': set(list(range(209))),
            'pascalvoc': set(list(range(136))),
            'pascalvoc_base': set(list(range(89))),
            'sa1b_joint': set(list(range(2))),
            'pascal_joint': set(list(range(113))),
            'partimagenet_joint': set(list(range(51))),
            'pascal_open_vocabulary': set(list(range(89))),
            'pascal_open_vocabulary_val': set(list(range(107))),
            'coco_panoptic': set(list(range(133))),
            'obj365_clip': set(list(range(365))),
            'openimage_clip': set(list(range(601))),
            'lvis_clip': set(list(range(1203))),
}

TEST_PART_ONLY_DATASETS = ['pascal_open_vocabulary_val', 'partimagenet_val', 'pascalvoc', 'ade20k', 'seginw_House-Parts']