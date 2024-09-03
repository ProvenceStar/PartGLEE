# Data Preparation

**Here we provide details on how to prepare all the datasets used in the training and testing stages of PartGLEE.**

Please first download our refined hierarchical annotations for 4 different datasets from [PartGLEE-Refined-Annotations](https://drive.google.com/file/d/1jLiG0WBQqxCOtgvBofiaCEk_w2FmvVjo/view?usp=drive_link). You are expected to unzip it and put these refined annotations to their corresponding folders following the instructions below. Please check the file tree of every dataset we provide to make sure the refined annotations are put under the right path.

After first create a `datasets` folder under `${PartGLEE_ROOT}` and then unzip the refined annotations, you are expected to get the annotations organized as below:
```
${PartGLEE_ROOT}
    -- datasets
        -- ADE20KPart234
            -- ade20k_joint_train.json
            -- ade20k_joint_val.json
        -- partimagenet
            -- partimagenet_train_joint.json
            -- partimagenet_val_joint.json
        -- pascal_part
            -- train_base_joint.json
            -- voc2010_train_joint.json
            -- voc2010_train_segm.json
            -- voc2010_val_joint.json
            -- voc2010_val_segm.json
        -- PascalPart116
            -- pascalvoc_joint_train.json
            -- pascalvoc_joint_val.json
```

## For Training

### COCO

Please download [COCO](https://cocodataset.org/#home) from the offical website. We use [train2017.zip](http://images.cocodataset.org/zips/train2017.zip), [train2014.zip](http://images.cocodataset.org/zips/train2014.zip), [val2017.zip](http://images.cocodataset.org/zips/val2017.zip), [test2017.zip](http://images.cocodataset.org/zips/test2017.zip) & [annotations_trainval2017.zip](http://images.cocodataset.org/annotations/annotations_trainval2017.zip), [image_info_test2017.zip](http://images.cocodataset.org/annotations/image_info_test2017.zip). We expect that the data is organized as below.

```
${PartGLEE_ROOT}
    -- datasets
        -- coco
            -- annotations
            -- train2017
            -- val2017
            -- test2017
```

### PACO

Download paco annotations and images according to [paco](https://github.com/facebookresearch/paco).
The PACO folder should look like:

```
${PartGLEE_ROOT}
    -- datasets
        -- paco
            -- annotations
                -- paco_lvis_v1_test.json
                -- paco_lvis_v1_train.json
                -- paco_lvis_v1_val.json
```

### PASCAL-PART

Please download Pascal-Part images and annotations following the scripts below:

```
wget http://roozbehm.info/pascal-parts/trainval.tar.gz
wget http://host.robots.ox.ac.uk/pascal/VOC/voc2010/VOCtrainval_03-May-2010.tar
```

The Pascal-Part folder should look like:
```
${PartGLEE_ROOT}
    -- datasets
        -- pascal_part/
            -- Annotations_Part/
                    2008_000002.mat
                    2008_000003.mat
                    ...
                    2010_006086.mat
            -- VOCdevkit/
                    VOC2010/
            -- voc2010_train_joint.json
            -- voc2010_val_joint.json
            -- voc2010_train_segm.json
            -- voc2010_val_segm.json
            -- train_base_joint.json
            -- train_base.json
            -- val.json
```

Then following [VLPart](https://github.com/facebookresearch/VLPart?tab=readme-ov-file) to convert their original annotations into coco format:

```
cd $PartGLEE_ROOT/
python projects/PartGLEE/tools/pascal_part_mat2json.py
python projects/PartGLEE/tools/pascal_part_mat2json.py --split train.txt --ann_out datasets/pascal_part/train.json
python projects/PartGLEE/tools/pascal_part_mat2json.py --only_base --split train.txt --ann_out datasets/pascal_part/train_base.json
python projects/PartGLEE/tools/pascal_part_one_json.py
python projects/PartGLEE/tools/pascal_part_one_json.py --only_base --part_path datasets/pascal_part/train_base.json --out_path datasets/pascal_part/train_base_one.json
```

### VOC

Download VOC2007 from:

```
wget http://host.robots.ox.ac.uk/pascal/VOC/voc2007/VOCtrainval_06-Nov-2007.tar
```

The VOC2007 folder is in `VOCdevkit/VOC2007`.

### PARTIMAGENET

Please download [PartImageNet](https://github.com/TACJu/PartImageNet) images and annotations from their offical repo.

The PartImageNet folder should look like:

```
${PartGLEE_ROOT}
    -- datasets 
        -- partimagenet/
            -- train/
                    n01440764
                    n01443537
                    ...            
            -- val/
                    n01484850
                    n01614925
                    ...
            -- train.json
            -- val.json
            -- partimagenet_train_joint.json
            -- partimagenet_val_joint.json
```

Convert the original annotations into coco annotation format following [VLPart](https://github.com/facebookresearch/VLPart?tab=readme-ov-file):
```
cd $PartGLEE_ROOT/
python projects/PartGLEE/tools/partimagenet_format_json.py --old_path datasets/partimagenet/train.json --new_path datasets/partimagenet/train_format.json
python projects/PartGLEE/tools/partimagenet_format_json.py --old_path datasets/partimagenet/val.json --new_path datasets/partimagenet/val_format.json
```

### LVIS

Please download [LVISv1](https://www.lvisdataset.org/dataset) from the offical website. LVIS uses the COCO 2017 train, validation, and test image sets, so only Annotation needs to be downloaded：[lvis_v1_train.json.zip](https://dl.fbaipublicfiles.com/LVIS/lvis_v1_train.json.zip),  [lvis_v1_val.json.zip](https://dl.fbaipublicfiles.com/LVIS/lvis_v1_val.json.zip), [lvis_v1_minival_inserted_image_name.json](https://huggingface.co/GLIPModel/GLIP/resolve/main/lvis_v1_minival_inserted_image_name.json). We expect that the data is organized as below.

```
${PartGLEE_ROOT}
    -- datasets
        -- lvis
            -- lvis_v1_train.json
            -- lvis_v1_val.json
            -- lvis_v1_minival_inserted_image_name.json
```

### VisualGenome

Please download [VisualGenome](https://homes.cs.washington.edu/~ranjay/visualgenome/api.html) images from the offical website:  [part 1 (9.2 GB)](https://cs.stanford.edu/people/rak248/VG_100K_2/images.zip), [part 2 (5.47 GB)](https://cs.stanford.edu/people/rak248/VG_100K_2/images2.zip), and download our preprocessed annotation file: [train.json](https://huggingface.co/spaces/Junfeng5/GLEE_demo/resolve/main/annotations/VisualGenome/train.json), [train_from_objects.json](https://huggingface.co/spaces/Junfeng5/GLEE_demo/resolve/main/annotations/VisualGenome/train_from_objects.json) . We expect that the data is organized as below.

```
${PartGLEE_ROOT}
    -- datasets
        -- visual_genome
            -- images
            	-- *.jpg
            			...
            -- annotations
              -- train_from_objects.json
              -- train.json
            -- vg_all_categories.txt
            -- vg_object_category.txt
            -- vg_part_category.txt
```

### OpenImages

Please download [OpenImages v6](https://storage.googleapis.com/openimages/web/download_v6.html) images from the offical website, all detection annotations need to be preprocessed into coco format. We expect that the data is organized as below. 

```
${PartGLEE_ROOT}
    -- datasets
        -- openimages
            -- detection
            -- openimages_v6_train_bbox.json
```

### VIS

Download YouTube-VIS [2019](https://codalab.lisn.upsaclay.fr/competitions/6064#participate-get_data), [2021](https://codalab.lisn.upsaclay.fr/competitions/7680#participate-get_data), [OVIS](https://codalab.lisn.upsaclay.fr/competitions/4763#participate) dataset for video instance segmentation task, and it is necessary to convert their video annotation into coco format in advance for image-level joint-training by run: ```python3 conversion/conver_vis2coco.py```  We expect that the data is organized as below. 

```
${PartGLEE_ROOT}
    -- datasets
        -- ytvis_2019
            -- train
            -- val
            -- annotations
            		-- instances_train_sub.json
            		-- instances_val_sub.json
            		-- ytvis19_cocofmt.json
        -- ytvis_2021
            -- train
            -- val
            -- annotations
            		-- instances_train_sub.json
            		-- instances_val_sub.json
            		-- ytvis21_cocofmt.json
        -- ovis
            -- train
            -- val		
            -- annotations_train.json
            -- annotations_valid.json
            -- ovis_cocofmt.json
```

### SA-1B

We downloaded data from the [SA1B](https://ai.meta.com/datasets/segment-anything-downloads/) official website, and only use [sa_000000.tar ~ sa_000050.tar] to  preprocess into the required format and train the model. First, we utilize a NMS-like method on each sa_n directory using the command below, this operation is taken to categorize the larger instance masks into objects, while the others are classified as parts:

```python
python3 convert_sam2coco_rewritresa1b.py  --src sa_000000
python3 convert_sam2coco_rewritresa1b.py  --src sa_000001
python3 convert_sam2coco_rewritresa1b.py  --src sa_000002
python3 convert_sam2coco_rewritresa1b.py  --src sa_000003
...
python3 convert_sam2coco_rewritresa1b.py  --src sa_000050
```

then merge all the annotations by running xxx.py.  

``` python
python3 merge_sa1b.py 
```

We expect that the data is organized as below. 

```
${PartGLEE_ROOT}
    -- datasets
        -- SA1B
            -- images
            		-- sa_000000
            				-- sa_1.jpg
            				-- sa_1.json
            				-- ...
            		-- sa_000001
            		-- ...
            -- sa1b_subtrain_500k.json
            -- sa1b_subtrain_1m.json
            -- sa1b_subtrain_2m.json
```

### Objects365 and others

Following UNINEXT, we prepare **Objects365, RefCOCO series, YouTubeVOS, Ref-YouTubeVOS, and BDD** data, and we expect that they are organized as below:

```
${PartGLEE_ROOT}
    -- datasets
        -- Objects365v2
            -- annotations
                -- zhiyuan_objv2_train_new.json
                -- zhiyuan_objv2_val_new.json
            -- images
        -- annotations
            -- refcoco-unc
            -- refcocog-umd
            -- refcocoplus-unc
        -- ytbvos18
            -- train
            -- val
        -- ref-youtube-vos
            -- meta_expressions
            -- train
            -- valid
            -- train.json
            -- valid.json
            -- RVOS_refcocofmt.json
        -- bdd
            -- images
                -- 10k
                -- 100k
                -- seg_track_20
                -- track
            -- labels
                -- box_track_20
                -- det_20
                -- ins_seg
                -- seg_track_20


```

RVOS_refcocofmt.json is the conversion of the annotation of ref-youtube-vos into the format of RefCOCO, which is used for image-level training. It can be converted by run ```python3 conversion/ref-ytbvos-conversion.py```

### ADE20K-Part-234 & Pascal-Part-116

Please follow [OV-PARTS](https://github.com/OpenRobotLab/OV_PARTS) to download their refined datasets and create the corresponding annotations. 

We expect the data is organized as below.

```
# ADE20K-Part-234
${PartGLEE_ROOT}
    -- datasets
        -- ADE20KPart234
            -- annotations_detectron2_part
                -- training
                -- validation
            -- images
                -- training
                -- validation
            -- ade20k_base_train.json
            -- ade20k_instance_train.json
            -- ade20k_instance_val.json
            -- ade20k_joint_train.json
            -- ade20k_joint_val.json
            -- train_16shot.json

# Pascal-Part-116
${PartGLEE_ROOT}
    -- datasets
        -- PascalPart116
            -- annotations_detectron2_obj
                -- train
                -- val
            -- annotations_detectron2_part
                -- train
                -- val
                -- train_obj_label_count.json
                -- train_part_label_count.json
                -- val_obj_label_count.json
                -- val_part_label_count.json
            -- images
                -- train
                -- val
            -- pascalvoc_base_train.json
            -- pascalvoc_joint_train.json
            -- pascalvoc_joint_val.json
            -- train_16shot.json
```

## For Evaluation Only

The following datasets are only used for zero-shot evaluation, and are not used in joint-training. 

### SeginW

Please follow [X-Decoder](https://github.com/microsoft/X-Decoder/tree/seginw) to prepare the corresponding images and annotations.

We expect the data is organized as below.
```
${PartGLEE_ROOT}
└── seginw/
    ├── Airplane-Parts/
    │   ├── train/
    │   │   ├── *.jpg
    │   │   └── _annotations_min1cat.coco.json
    │   ├── train_10shot/
    │   │   └── ...
    │   └── valid/
    │       └── ...
    ├── Bottles/
    │   └── ...
    └── ...
```