# Tutorial for Testing

## Object-level Detection and Segmentation 

PartGLEE can directly perform evaluations on COCO, LVIS, and the RefCOCO series based on Detectron2. Please first download our weights from [MODEL_ZOO.md](MODEL_ZOO.md) and put them under your `"checkpoint"` folder, by default we set the `"checkpoint"` folder under the path: `"projects/PartGLEE/checkpoint"`.
To inference on COCO:

```bash
# RN50
python3 projects/PartGLEE/train_net.py --config-file projects/PartGLEE/configs/Inference/RN50/coco.yaml  --num-gpus 8 --eval-only

# SwinL
python3 projects/PartGLEE/train_net.py --config-file projects/PartGLEE/configs/Inference/Swin-L/coco.yaml  --num-gpus 8 --eval-only
```


You can also customize your checkpoint folder path, after modification please add `"path/to/downloaded/weights"` with the actual path of pretrained model weights to the command and use `"DATASETS.TEST"` to specific the dataset you wish to evaluate on.

We note that `'("coco_2017_val",)'` can be replace by:

```bash
# RN50
python3 projects/PartGLEE/train_net.py --config-file projects/PartGLEE/configs/Inference/RN50/coco.yaml  --num-gpus 8 --eval-only  MODEL.WEIGHTS path/to/PartGLEE_weights.pth  DATASETS.TEST 
'("coco_2017_val",)'
'("lvis_v1_minival",)'
'("lvis_v1_val",)'
'("objects365_v2_val",)'
'("refcoco-unc-val",)'
'("refcoco-unc-testA",)'
'("refcoco-unc-testB",)'
'("refcocoplus-unc-val",)'
'("refcocoplus-unc-testA",)'
'("refcocoplus-unc-testB",)'
'("refcocog-umd-val",)'
'("refcocog-umd-test",)'
# Alternatively, to infer across all tasks at once:
'("coco_2017_val","lvis_v1_minival","lvis_v1_val","objects365_v2_val","refcoco-unc-val","refcoco-unc-testA","refcoco-unc-testB","refcocoplus-unc-val","refcocoplus-unc-testA","refcocoplus-unc-testB","refcocog-umd-val","refcocog-umd-test",)'
```

## Part-level Detection and Segmentation 

PartGLEE extends the recognition capability of GLEE to part-level instances. It can perform evaluations on PACO, PASCAL-PART, PartImageNet, ADE20K-Part-234 as well as Pascal-Part-116 (refined by [OV-PARTS](https://github.com/OpenRobotLab/OV_PARTS)) based on Detectron2. Please first download our weights from [MODEL_ZOO.md](MODEL_ZOO.md) and put them under your `"checkpoint"` folder, by default we set the `"checkpoint"` folder under the path: `"projects/PartGLEE/checkpoint"`.

To inference on PACO:

```bash
# RN50
python3 projects/PartGLEE/train_net.py --config-file projects/PartGLEE/configs/Inference/RN50/paco.yaml  --num-gpus 8 --eval-only

# SwinL
python3 projects/PartGLEE/train_net.py --config-file projects/PartGLEE/configs/Inference/Swin-L/paco.yaml  --num-gpus 8 --eval-only
```

You can also customize your checkpoint folder path, after modification please add `"path/to/downloaded/weights"` with the actual path of pretrained model weights to the command and use `"DATASETS.TEST"` to specific the dataset you wish to evaluate on.

We note that `'("coco_2017_val",)'` can be replace by:

```bash
# RN50
python3 projects/PartGLEE/train_net.py --config-file projects/PartGLEE/configs/Inference/RN50/paco.yaml  --num-gpus 8 --eval-only  MODEL.WEIGHTS path/to/PartGLEE_weights.pth  DATASETS.TEST 
'("partimagenet_val",)'
'("pascal_part_open_vocabulary_val",)'
# Alternatively, to infer across all tasks at once:
'("paco_lvis_v1_val","partimagenet_val","pascal_part_open_vocabulary_val",)'
```

We note that the inference process of ADE20K-Part-234 and Pascal-Part-116 follows the semantic segmentation inference pipeline. Consequently, their inference configs are different from the others. If you want to perform inference on these two datasets, use the following command:

```bash
# ADE20K-Part-234
python3 projects/PartGLEE/train_net.py --config-file projects/PartGLEE/configs/Inference/RN50/ade20k-part-234.yaml --num-gpus 8 --eval-only

# Pascal-Part-116
python3 projects/PartGLEE/train_net.py --config-file projects/PartGLEE/configs/Inference/RN50/pascal-part-116.yaml --num-gpus 8 --eval-only
```