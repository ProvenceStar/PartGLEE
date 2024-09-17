# Copyright (c) Facebook, Inc. and its affiliates.
import logging
import os
from detectron2.data import DatasetCatalog, MetadataCatalog
from detectron2.data.datasets.coco import load_coco_json, register_coco_instances

OBJ_CLASS_NAMES = [
    "aeroplane",
    "bicycle",
    "bird",
    "boat",
    "bottle",
    "bus",
    "car",
    "cat",
    "chair",
    "cow",
    "diningtable",
    "dog",
    "horse",
    "motorbike",
    "person",
    "pottedplant",
    "sheep",
    "sofa",
    "train",
    "tvmonitor",
]
OBJ_BASE_CLASS_NAMES = [
    c for i, c in enumerate(OBJ_CLASS_NAMES) if c not in ["bird", "car", "dog", "sheep", "motorbike"]
]
CLASS_NAMES = ["aeroplane's body", "aeroplane's stern", "aeroplane's wing", "aeroplane's tail", "aeroplane's engine", "aeroplane's wheel", "bicycle's wheel", "bicycle's saddle", "bicycle's handlebar", "bicycle's chainwheel", "bicycle's headlight", 
               "bird's wing", "bird's tail", "bird's head", "bird's eye", "bird's beak", "bird's torso", "bird's neck", "bird's leg", "bird's foot", "bottle's body", "bottle's cap", "bus's wheel", "bus's headlight", "bus's front", "bus's side", "bus's back", 
               "bus's roof", "bus's mirror", "bus's license plate", "bus's door", "bus's window", "car's wheel", "car's headlight", "car's front", "car's side", "car's back", "car's roof", "car's mirror", "car's license plate", "car's door", "car's window", 
               "cat's tail", "cat's head", "cat's eye", "cat's torso", "cat's neck", "cat's leg", "cat's nose", "cat's paw", "cat's ear", "cow's tail", "cow's head", "cow's eye", "cow's torso", "cow's neck", "cow's leg", "cow's ear", "cow's muzzle", "cow's horn", 
               "dog's tail", "dog's head", "dog's eye", "dog's torso", "dog's neck", "dog's leg", "dog's nose", "dog's paw", "dog's ear", "dog's muzzle", "horse's tail", "horse's head", "horse's eye", "horse's torso", "horse's neck", "horse's leg", "horse's ear", 
               "horse's muzzle", "horse's hoof", "motorbike's wheel", "motorbike's saddle", "motorbike's handlebar", "motorbike's headlight", "person's head", "person's eye", "person's torso", "person's neck", "person's leg", "person's foot", "person's nose", 
               "person's ear", "person's eyebrow", "person's mouth", "person's hair", "person's lower arm", "person's upper arm", "person's hand","pottedplant's pot", "pottedplant's plant", 
               "sheep's tail", "sheep's head", "sheep's eye", "sheep's torso", "sheep's neck", "sheep's leg", "sheep's ear", "sheep's muzzle", "sheep's horn", "train's headlight", "train's head", "train's front", "train's side", "train's back", "train's roof", 
               "train's coach", "tvmonitor's screen"]
BASE_CLASS_NAMES = [
    c for i, c in enumerate(CLASS_NAMES) if c not in ["bird's wing", "bird's tail", "bird's head", "bird's eye", "bird's beak", "bird's torso", "bird's neck", "bird's leg", "bird's foot",
                                                      "car's wheel", "car's headlight", "car's front", "car's side", "car's back", "car's roof", "car's mirror", "car's license plate", "car's door", "car's window",
                                                      "dog's tail", "dog's head", "dog's eye", "dog's torso", "dog's neck", "dog's leg", "dog's nose", "dog's paw", "dog's ear", "dog's muzzle",
                                                      "sheep's tail", "sheep's head", "sheep's eye", "sheep's torso", "sheep's neck", "sheep's leg", "sheep's ear", "sheep's muzzle", "sheep's horn",
                                                      "motorbike's wheel", "motorbike's saddle", "motorbike's handlebar", "motorbike's headlight"]
]

def _get_partimagenet_metadata():
    PASCAL_BASE_CLASS_NAMES = OBJ_BASE_CLASS_NAMES + BASE_CLASS_NAMES
    PASCAL_BASE_CATEGORIES = [{"id": idx, "name": PASCAL_BASE_CLASS_NAMES[idx]} for idx in range(len(PASCAL_BASE_CLASS_NAMES))]
    id_to_name = {x['id']: x['name'] for x in PASCAL_BASE_CATEGORIES}
    thing_dataset_id_to_contiguous_id = {
        x: i for i, x in enumerate(sorted(id_to_name))}
    thing_classes = [id_to_name[k] for k in sorted(id_to_name)]
    return {
        "thing_dataset_id_to_contiguous_id": thing_dataset_id_to_contiguous_id,
        "thing_classes": thing_classes}

_PASCAL_BASE = {
    "pascalvoc_base_train": ("PascalPart116/images/train", "PascalPart116/pascalvoc_base_train.json"),
}

def register_all_pascalvoc_base(root):
    for key, (image_root, json_file) in _PASCAL_BASE.items():
        register_coco_instances(
            key,
            _get_partimagenet_metadata(),
            os.path.join(root, json_file) if "://" not in json_file else json_file,
            os.path.join(root, image_root),
            dataset_name_in_dict="pascalvoc_base",
            evaluator_type='coco'
        )

# _root = os.getenv("DETECTRON2_DATASETS", "datasets")
# register_all_pascal_joint(_root)