# Copyright (c) 2024 ByteDance. All Rights Reserved.
# PartGLEE Model.
# PartGLEE: A Foundation Model for Recognizing and Parsing Any Objects (ECCV 2024)
# https://arxiv.org/abs/2407.16696

import torch
import torch.nn.functional as F
from torch import nn

from detectron2.modeling import build_backbone
from ..language import build_language_encoder

from .pixel_decoder.maskdino_encoder import build_pixel_decoder
from .transformer_decoder.maskdino_decoder import build_transformer_decoder
from .transformer_decoder.maskdino_part_decoder import build_part_transformer_decoder
from timm.models.layers import trunc_normal_
from transformers import CLIPTokenizer,CLIPTextModel
from .vos_utils import masks_to_boxes, FeatureFuser
from .transformer_decoder.transformer import QFormer, QFormerDecoderLayer
from ._CONSTANTS import OBJECT_LEVEL_DATASETS
from ._INFERENCE_CONSTANTS import TEST_TOPK_PER_IMAGE, DATASET_SPECIFIC_OBJECT_NUMS, DATASET_SPECIFIC_PART_NUMS, DATASET_SPECIFIC_CATEGORIES, PART_DATASETS_OBJECT_CATEGORY_INDEX, PART_DATASETS_PART_CATEGORY_INDEX, TEST_PART_ONLY_DATASETS

def rand_sample(x, max_len):
    if x.shape[1] <= max_len:
        return x
    else:
        rand_idx = torch.randperm(x.shape[1])[:max_len]
        return x[:,rand_idx]


def agg_lang_feat(features, mask, pool_type="average"):
    """average pooling of language features"""
    # feat: (bs, seq_len, C)
    # mask: (bs, seq_len)
    if pool_type == "average":
        embedded = features * mask.unsqueeze(-1).float() # use mask to zero out invalid token features
        aggregate = embedded.sum(1) / (mask.sum(-1).unsqueeze(-1).float())
    elif pool_type == "max":
        out = []
        for i in range(len(features)):
            pool_feat, _ = torch.max(features[i][mask[i]], 0) # (L, C) -> (C, )
            out.append(pool_feat)
        aggregate = torch.stack(out, dim=0) # (bs, C)
    else:
        raise ValueError("pool_type should be average or max")
    return aggregate

class PartGLEE_Model(nn.Module):
    """
    Main class for mask classification semantic segmentation architectures.
    """
    def __init__(self, cfg, matcher, device, video_info, contras_mean, unify_object_part):
        super().__init__()
        self.cfg = cfg
        self.matcher = matcher
        self.backbone = build_backbone(cfg)
        output_channels = [v for k,v in self.backbone._out_feature_channels.items()]
        if cfg.MODEL.VISUAL_PROMPT:
            self.sot_fuser = FeatureFuser(output_channels[-3:], 256)
        
        self.text_encode_type = cfg.MODEL.TEXT.ARCH
        if cfg.MODEL.TEXT.ARCH == 'clip_frozen':
            self.tokenizer = CLIPTokenizer.from_pretrained('projects/PartGLEE/clip_vit_base_patch32') 
            self.tokenizer.add_special_tokens({'cls_token': self.tokenizer.eos_token})
            self.text_encoder = CLIPTextModel.from_pretrained('projects/PartGLEE/clip_vit_base_patch32')
            self.lang_encoder = None
            for p in self.text_encoder.parameters():
                p.requires_grad = False
            self.lang_projection = nn.Parameter(torch.rand(cfg.MODEL.LANGUAGE_BACKBONE.LANG_DIM, cfg.MODEL.DIM_PROJ))
        elif cfg.MODEL.TEXT.ARCH == 'clip_unfreeze':
            self.tokenizer = CLIPTokenizer.from_pretrained('projects/PartGLEE/clip_vit_base_patch32') 
            self.tokenizer.add_special_tokens({'cls_token': self.tokenizer.eos_token})
            self.text_encoder = CLIPTextModel.from_pretrained('projects/PartGLEE/clip_vit_base_patch32')
            self.lang_encoder = None
            self.lang_projection = nn.Parameter(torch.rand(cfg.MODEL.LANGUAGE_BACKBONE.LANG_DIM, cfg.MODEL.DIM_PROJ))
            self.text_encode_type = 'clip_frozen'
        elif cfg.MODEL.TEXT.ARCH == 'clip_teacher':
            self.tokenizer = CLIPTokenizer.from_pretrained('projects/PartGLEE/clip_vit_base_patch32') 
            self.tokenizer.add_special_tokens({'cls_token': self.tokenizer.eos_token})
            self.text_encoder = CLIPTextModel.from_pretrained('projects/PartGLEE/clip_vit_base_patch32')
            self.text_encoder_teacher = CLIPTextModel.from_pretrained('projects/PartGLEE/clip_vit_base_patch32')
            self.lang_encoder = None
            for p in self.text_encoder_teacher.parameters():
                p.requires_grad = False
            self.lang_projection = nn.Parameter(torch.rand(cfg.MODEL.LANGUAGE_BACKBONE.LANG_DIM, cfg.MODEL.DIM_PROJ))

        # self.lang_encoder = None     
        self.pixel_decoder = build_pixel_decoder(cfg, self.backbone.output_shape())
        transformer_predictor_in_channels = cfg.MODEL.SEM_SEG_HEAD.CONVS_DIM
        self.use_early_fusion = cfg.MODEL.USE_EARLYFUSION
        self.object_predictor = build_transformer_decoder(cfg, transformer_predictor_in_channels, lang_encoder = self.lang_encoder, mask_classification=True,)
        self.part_predictor = build_part_transformer_decoder(cfg, transformer_predictor_in_channels, lang_encoder = self.lang_encoder, mask_classification=True, is_part_decoder=True)
        
        # Unify object-part tasks
        self.unify_object_part = unify_object_part
        self.use_qformer = cfg.MODEL.MaskDINO.Q_FORMER
        
        if self.unify_object_part:
            # Hyperparameters for Q-Former
            num_object_queries = cfg.MODEL.MaskDINO.NUM_OBJECT_QUERIES
            self.num_object_queries = num_object_queries
            num_part_queries = cfg.MODEL.MaskDINO.NUM_PART_QUERIES
            self.num_part_queries = num_part_queries
            hidden_dim = cfg.MODEL.MaskDINO.HIDDEN_DIM
            self.num_decoder_layer = cfg.MODEL.MaskDINO.OBJECT_PART_DECODER_LAYERS
            self.topk_object_queries_num = cfg.MODEL.MaskDINO.TOPK_OBJECT_QUERIES
            self.part_queries_feat = nn.Embedding(num_part_queries, hidden_dim)
            
            nhead = cfg.MODEL.MaskDINO.NHEADS
            dim_feedforward = cfg.MODEL.MaskDINO.DIM_FEEDFORWARD
            dropout = 0.0
            activation = 'relu'
            self.part_queries_pos = nn.Embedding(num_part_queries, hidden_dim)
            self.obj_queries_pos = nn.Embedding(num_object_queries, hidden_dim)
            qformer_layer = QFormerDecoderLayer(hidden_dim, nhead, dim_feedforward, dropout, activation)
            qformer_norm = nn.LayerNorm(hidden_dim)
            self.qformer = QFormer(qformer_layer, self.num_decoder_layer, qformer_norm, return_intermediate=False)
        
        self.device = device
        self.to(device)
        
        self.visualize = cfg.MODEL.MaskDINO.TEST.VISUALIZE
        
        self.test_topk_per_image = TEST_TOPK_PER_IMAGE
        self.test_object_inst_num = DATASET_SPECIFIC_OBJECT_NUMS
        self.test_part_inst_num = DATASET_SPECIFIC_PART_NUMS
        self.num_classes = DATASET_SPECIFIC_CATEGORIES
        self.object_category_index_mapper = PART_DATASETS_OBJECT_CATEGORY_INDEX
        self.part_category_index_mapper = PART_DATASETS_PART_CATEGORY_INDEX
        
        self.object_level_datasets = OBJECT_LEVEL_DATASETS
        self.test_part_only_datasets = TEST_PART_ONLY_DATASETS
        
        self.video_info = video_info
        self.contras_mean = contras_mean
        self.track_loss_version = cfg.MODEL.TRACK_VERSION

        # for visual prompt
        hidden_dim = 256
        self.max_spatial_len = [512,512,512,512]
        self.mask_sptial_embed = nn.ParameterList([nn.Parameter(torch.empty(hidden_dim, hidden_dim)) for x in range(4)])
        trunc_normal_(self.mask_sptial_embed[0], std=.02)
        trunc_normal_(self.mask_sptial_embed[1], std=.02)
        trunc_normal_(self.mask_sptial_embed[2], std=.02)
        trunc_normal_(self.mask_sptial_embed[3], std=.02)
        # learnable positive negative indicator
        self.pn_indicator = nn.Embedding(2, hidden_dim)
    
    def forward(self, images, prompts, task, targets=None, batch_name_list=None, is_train=True, criterion=None, custom_object_categories_idx=None, custom_part_categories_idx=None):
        extra =  {}
        
        early_semantic = None
        if self.text_encode_type == 'clip_frozen':
            if task not in ['grounding','rvos']:
                assert batch_name_list
                classes_name_list = batch_name_list
                tokenized = self.tokenizer.batch_encode_plus(classes_name_list,
                        max_length=self.cfg.MODEL.LANGUAGE_BACKBONE.MAX_QUERY_LEN, # 256
                        padding='max_length' if self.cfg.MODEL.LANGUAGE_BACKBONE.PAD_MAX else "longest", # max_length
                        return_special_tokens_mask=True,
                        return_tensors='pt',
                        truncation=True).to("cuda")
                texts = (tokenized['input_ids'], tokenized['attention_mask'])
                token_x = self.text_encoder(*texts)['last_hidden_state']
                token_x = token_x @ self.lang_projection
                lang_feat_pool = agg_lang_feat(token_x, tokenized['attention_mask'], pool_type="average")  # (bs, 768)
                extra['class_embeddings'] = lang_feat_pool
                dist_loss =  (lang_feat_pool*0).sum()
                if self.use_early_fusion: # early_fusion
                    gather_all_classtoken = token_x.flatten(0,1)[tokenized['attention_mask'].flatten(0,1)>0]
                    gather_all_classtoken = gather_all_classtoken.unsqueeze(0).repeat(len(images),1,1) #[bs,L,C]
                    gather_all_classtoken_mask = torch.ones_like(gather_all_classtoken[:,:,0])>0  #[bs,L]
                    early_semantic = {"hidden":gather_all_classtoken.float(),"masks":gather_all_classtoken_mask}
                    
        elif self.text_encode_type == "clip_teacher":
            if task not in ['grounding','rvos']:
                assert batch_name_list
                classes_name_list = batch_name_list
                tokenized = self.tokenizer.batch_encode_plus(classes_name_list,
                        max_length=self.cfg.MODEL.LANGUAGE_BACKBONE.MAX_QUERY_LEN, # 256
                        padding='max_length' if self.cfg.MODEL.LANGUAGE_BACKBONE.PAD_MAX else "longest", # max_length
                        return_special_tokens_mask=True,
                        return_tensors='pt',
                        truncation=True).to("cuda")

                texts = (tokenized['input_ids'], tokenized['attention_mask'])
                token_x = self.text_encoder(*texts)['last_hidden_state']

                valid_mask = tokenized['attention_mask'].bool()
                token_x_teacher = self.text_encoder_teacher(*texts)['last_hidden_state']
                dist_loss =  F.mse_loss(token_x[valid_mask], token_x_teacher[valid_mask] )
                token_x = token_x @ self.lang_projection
                lang_feat_pool = agg_lang_feat(token_x, tokenized['attention_mask'], pool_type="average")  # (bs,  768)
                extra['class_embeddings'] = lang_feat_pool 
                if self.use_early_fusion: # early_fusion
                    gather_all_classtoken = token_x.flatten(0,1)[tokenized['attention_mask'].flatten(0,1)>0]
                    gather_all_classtoken = gather_all_classtoken.unsqueeze(0).repeat(len(images),1,1) #[bs,L,C]
                    gather_all_classtoken_mask = torch.ones_like(gather_all_classtoken[:,:,0])>0  #[bs,L]
                    early_semantic = {"hidden":gather_all_classtoken.float(),"masks":gather_all_classtoken_mask} 
                    
        if 'grounding' in prompts:
            if self.text_encode_type == 'vlpencoder':
                gtext = self.object_predictor.lang_encoder.get_text_token_embeddings(prompts['grounding'], name='grounding', token=False, norm=False)
                token_emb = gtext['token_emb']
                tokens = gtext['tokens']
                extra['grounding_tokens'] = token_emb.permute(1,0,2) #[len,bz,C]
                non_zero_query_mask = tokens['attention_mask']
                extra['grounding_nonzero_mask'] = ~non_zero_query_mask.bool()  # [bz,len]
                lang_feat_pool = agg_lang_feat(gtext['token_emb'], tokens['attention_mask'], pool_type="average").unsqueeze(1) # (bs, 1, 768)
                extra['grounding_class'] = lang_feat_pool.squeeze(1) #[bz,C
            elif self.text_encode_type == 'bert':
                tokenized = self.tokenizer.batch_encode_plus(prompts['grounding'],
                        max_length=self.cfg.MODEL.LANGUAGE_BACKBONE.MAX_QUERY_LEN, # 256
                        padding='max_length' if self.cfg.MODEL.LANGUAGE_BACKBONE.PAD_MAX else "longest", # max_length
                        return_special_tokens_mask=True,
                        return_tensors='pt',
                        truncation=True).to("cuda")
                tokenizer_input = {"input_ids": tokenized.input_ids, "attention_mask": tokenized.attention_mask}
                language_dict_features = self.text_encoder(tokenizer_input) # dict with keys: ['aggregate', 'embedded', 'masks', 'hidden']
                lang_feat_pool = agg_lang_feat(language_dict_features["hidden"], language_dict_features["masks"], pool_type="average").unsqueeze(1) # (bs, 1, 768)
                extra['grounding_tokens'] = (language_dict_features["hidden"]@self.lang_projection).permute(1,0,2) #[len,bz,C]
                extra['grounding_nonzero_mask'] = ~language_dict_features["masks"].bool()  # [bz,len]
                extra['grounding_class'] = lang_feat_pool.squeeze(1)@self.lang_projection
            elif self.text_encode_type == 'clip_frozen' or self.text_encode_type == 'clip_teacher':
                tokens = self.tokenizer(
                    prompts['grounding'], padding='max_length', truncation=True, max_length=self.cfg.MODEL.LANGUAGE_BACKBONE.MAX_QUERY_LEN, return_tensors='pt'
                    )
                tokens = {key: value.cuda() for key, value in tokens.items()}

                texts = (tokens['input_ids'], tokens['attention_mask'])
                x = self.text_encoder(*texts)
                token_x = x['last_hidden_state']
                token_x = token_x @ self.lang_projection

                extra['grounding_tokens'] = token_x.permute(1,0,2) #[len,bz,C]

                non_zero_query_mask = tokens['attention_mask']
                lang_feat_pool = agg_lang_feat(token_x, non_zero_query_mask, pool_type="average").unsqueeze(1) # (bs, 1, 768)

                dist_loss =  (lang_feat_pool*0).sum()
                
                extra['grounding_nonzero_mask'] = ~non_zero_query_mask.bool()  # [bz,len]
                extra['grounding_class'] = lang_feat_pool.squeeze(1) #[bz,C
                early_semantic = {"hidden":token_x.float(),"masks":tokens['attention_mask']>0} 

        if isinstance(images,torch.Tensor):
            features = self.backbone(images)
        else:
            features = self.backbone(images.tensor)
        
        mask_features, _, multi_scale_features, zero_loss = self.pixel_decoder.forward_features(features, masks=None, early_fusion = early_semantic)
        ## ensure all params in loss caculation 
        if early_semantic:
            params_zero_loss = zero_loss + (self.pn_indicator.weight*0).sum()
        else:
            zero_loss = 0
            params_zero_loss = zero_loss + (self.pn_indicator.weight*0).sum()
            
        for p in self.mask_sptial_embed:
            params_zero_loss += (p*0).sum()

        params_zero_loss += (self.object_predictor.coco_label_enc.weight*0).sum()  +\
        (self.object_predictor.obj365_label_enc.weight*0).sum() +\
        (self.object_predictor.vg_label_enc.weight*0).sum() +\
        (self.object_predictor.grounding_label_enc.weight*0).sum() +\
        (self.object_predictor.ytvis19_label_enc.weight*0).sum() +\
        (self.object_predictor.ytvis21_label_enc.weight*0).sum() +\
        (self.object_predictor.ovis_label_enc.weight*0).sum() +\
        (self.object_predictor.uvo_label_enc.weight*0).sum() +\
        (self.object_predictor.bdd_det.weight*0).sum() +\
        (self.object_predictor.bdd_inst.weight*0).sum()
        
        if is_train and self.unify_object_part:
            topk = self.topk_object_queries_num
            if task not in self.object_level_datasets:
                # decouple targets into object-level targets and part-level targets
                object_targets, part_targets = targets
                # For Datasets that have both object-level and part-level annotations, redefine task
                object_level_task = task + '_object'
                part_level_task = task + '_part'
            else:
                object_targets = targets
                part_targets = [{"labels": torch.tensor([]).to(object_targets[i]['labels'].device), "boxes": torch.empty(size=(0,4), device=object_targets[i]['labels'].device), "masks": None} for i in range(len(object_targets))]
                object_level_task = task
                part_level_task = task
                
            assert criterion is not None, 'The training phase must acquire criterion to perform matching and obtain the matched indices'
            
            object_outputs, object_mask_dict, object_queries, src_flatten = self.object_predictor(multi_scale_features, mask_features, extra=extra, task=object_level_task, masks=None, targets=object_targets)
            
            fake_object_track_loss = (object_outputs['pred_track_embed']*0).sum()
            # Perform matching in the object-level instances
            object_losses, matched_indices = criterion(object_outputs, object_targets, object_mask_dict, object_level_task)
            object_losses.update({"track_loss":fake_object_track_loss})
            object_losses.update({"dist_loss":dist_loss+params_zero_loss})
            
            # Q-Former
            # In order to generate part queries for each object queries, we transform the dim of object queries into [bs*nq,1,c=256]
            if self.use_qformer:
                bs, num_object_queries, hidden_dim = object_queries.shape
                if self.detach_object_queries:
                    object_queries = object_queries.detach()     
                topk_object_queries_indices = torch.topk(object_outputs['pred_logits'].max(-1)[0], topk, 1)[1]
                # object_quereis:[bs,nq,c=256]->topk_object_queries:[bs,num_topk_queries,c=256]
                topk_object_queries = torch.gather(object_queries, 1, topk_object_queries_indices.unsqueeze(-1).repeat(1, 1, hidden_dim))
                topk_object_queries = topk_object_queries.flatten(0,1).unsqueeze(1)   #[bs*nq,1,c=256]
                bsnq = topk_object_queries.shape[0]
                
                part_queries = self.part_queries_feat.weight
                part_queries = part_queries.repeat(bsnq, 1, 1)
                
                part_queries_pos = self.part_queries_pos.weight.repeat(bsnq, 1, 1)
                object_queries_pos = torch.gather(self.obj_queries_pos.weight.repeat(bs, 1, 1), 1, topk_object_queries_indices.unsqueeze(-1).repeat(1, 1, hidden_dim))
                object_queries_pos = object_queries_pos.flatten(0,1).unsqueeze(1)     #[bs*nq,1,c=256]
                part_queries = self.qformer(
                    tgt=part_queries.transpose(0,1),
                    memory=topk_object_queries.transpose(0,1),
                    memory_mask=None,
                    tgt_key_padding_mask=None,
                    memory_key_padding_mask=None,
                    pos=object_queries_pos.transpose(0,1),
                    query_pos=part_queries_pos.transpose(0,1),
                )       # part_queries:[num_part_queries, bs*num_object_queries, c=256]
                
                part_queries = part_queries.reshape(bs, topk*self.num_part_queries, hidden_dim)

                # Part-level Decoder
                part_outputs, part_mask_dict = self.part_predictor(multi_scale_features, mask_features, extra=extra, task=part_level_task, masks=None, targets=part_targets, \
                                                                    part_queries=part_queries, topk=topk, num_part_queries=self.num_part_queries)
                
                # Perform matching in the part-level instances
                fake_part_track_loss = (part_outputs['pred_track_embed']*0).sum()
                if task in self.object_level_datasets:
                    fake_task = task + '_fake'
                    part_losses, _ = criterion(part_outputs, part_targets, part_mask_dict, fake_task, object_outputs, topk_object_queries_indices)
                else:
                    part_losses, _ = criterion(part_outputs, part_targets, part_mask_dict, part_level_task, object_outputs, topk_object_queries_indices)
                part_losses.update({"track_loss":fake_part_track_loss})
                part_losses.update({"dist_loss":dist_loss+params_zero_loss})

            losses = object_losses
            for key in part_losses.keys():    
                losses.update({"part_" + key: part_losses[key]})
            
            return losses
        
        else:
            # Inference
            outputs = self.hierarchical_inference(batch_name_list, task, targets, multi_scale_features, mask_features, extra, custom_object_categories_idx, custom_part_categories_idx)
            return outputs
        
    def hierarchical_inference(self, batch_name_list, task, targets, multi_scale_features, mask_features, extra, custom_object_categories_idx, custom_part_categories_idx):
        topk = self.topk_object_queries_num
        
        if task not in self.object_level_datasets:
            object_level_task = task + '_object'
            part_level_task = task + '_part'
        else:
            object_level_task = part_level_task = task
            
        object_outputs, _, object_queries, src_flatten = self.object_predictor(multi_scale_features, mask_features, extra=extra, task=object_level_task, masks=None, targets=targets, custom_object_categories_idx=custom_object_categories_idx)
        # Q-Former
        if self.use_qformer:
            # In order to generate part queries for each object queries, we transform the dim of object queries into [bs*nq,1,c=256]
            bs, num_object_queries, hidden_dim = object_queries.shape
            topk_object_queries_indices = torch.topk(object_outputs['pred_logits'].max(-1)[0], topk, 1)[1]
            # object_quereis:[bs,nq,c=256]->topk_object_queries:[bs,num_topk_queries,c=256]
            topk_object_queries = torch.gather(object_queries, 1, topk_object_queries_indices.unsqueeze(-1).repeat(1, 1, hidden_dim))
            topk_object_queries = topk_object_queries.flatten(0,1).unsqueeze(1)   #[bs*nq,1,c=256]
            bsnq = topk_object_queries.shape[0]
            part_queries = self.part_queries_feat.weight.repeat(bsnq, 1, 1)

            part_queries_pos = self.part_queries_pos.weight.repeat(bsnq, 1, 1)
            object_queries_pos = torch.gather(self.obj_queries_pos.weight.repeat(bs, 1, 1), 1, topk_object_queries_indices.unsqueeze(-1).repeat(1, 1, hidden_dim))
            object_queries_pos = object_queries_pos.flatten(0,1).unsqueeze(1)     #[bs*nq,1,c=256]
            part_queries = self.qformer(
                tgt=part_queries.transpose(0,1),
                memory=topk_object_queries.transpose(0,1),
                memory_mask=None,
                tgt_key_padding_mask=None,
                memory_key_padding_mask=None,
                pos=object_queries_pos.transpose(0,1),
                query_pos=part_queries_pos.transpose(0,1),
            )       # part_queries:[num_part_queries, bs*num_object_queries, c=256]
                
            part_queries = part_queries.reshape(bs, topk*self.num_part_queries, hidden_dim)        
            
            # Part-level Decoder
            part_outputs, _ = self.part_predictor(multi_scale_features, mask_features, extra=extra, task=part_level_task, masks=None, \
                                                part_queries=part_queries, topk=topk, num_part_queries=self.num_part_queries, custom_part_categories_idx=custom_part_categories_idx)
        
        # Determine topk predictions
        test_topk = self.inference_topk(task, topk)
        topk_object, topk_part = self.hierarchical_topk(task, topk)
        
        # Get hierarchical predictions
        outputs = self.hierarchical_topk_outputs(task, object_level_task, part_level_task, batch_name_list, object_outputs, part_outputs, test_topk, topk_object, topk_part, custom_object_categories_idx, custom_part_categories_idx)
        
        return outputs

    def hierarchical_topk_outputs(self, task, object_level_task, part_level_task, batch_name_list, object_outputs, part_outputs, test_topk, topk_object, topk_part, custom_object_categories_idx, custom_part_categories_idx):
        outputs = {}
        top_object_outputs_indices = torch.topk(object_outputs['pred_logits'].max(-1)[0], topk_object, 1)[1]
        top_object_outputs = {}
        top_object_outputs['pred_logits'] = torch.gather(object_outputs['pred_logits'], 1, top_object_outputs_indices.unsqueeze(-1).repeat(1, 1, object_outputs['pred_logits'].shape[-1]))
        top_object_outputs['pred_masks'] = torch.gather(object_outputs['pred_masks'], 1, top_object_outputs_indices.unsqueeze(-1).unsqueeze(-1).repeat(1, 1, object_outputs['pred_masks'].shape[-2], object_outputs['pred_masks'].shape[-1]))
        top_object_outputs['pred_boxes'] = torch.gather(object_outputs['pred_boxes'], 1, top_object_outputs_indices.unsqueeze(-1).repeat(1, 1, object_outputs['pred_boxes'].shape[-1]))
        
        if task in self.object_level_datasets:
            return top_object_outputs
        
        top_part_outputs_indices = torch.topk(part_outputs['pred_logits'].max(-1)[0], topk_part, 1)[1]
        top_part_outputs = {}
        top_part_outputs['pred_logits'] = torch.gather(part_outputs['pred_logits'], 1, top_part_outputs_indices.unsqueeze(-1).repeat(1, 1, part_outputs['pred_logits'].shape[-1]))
        top_part_outputs['pred_masks'] = torch.gather(part_outputs['pred_masks'], 1, top_part_outputs_indices.unsqueeze(-1).unsqueeze(-1).repeat(1, 1, part_outputs['pred_masks'].shape[-2], object_outputs['pred_masks'].shape[-1]))
        top_part_outputs['pred_boxes'] = torch.gather(part_outputs['pred_boxes'], 1, top_part_outputs_indices.unsqueeze(-1).repeat(1, 1, part_outputs['pred_boxes'].shape[-1]))
        
        if task in self.test_part_only_datasets:
            return top_part_outputs
        
        for key in top_object_outputs.keys():
            if key == 'pred_logits':
                if task == 'custom':
                    all_logits = torch.full((1, test_topk, len(batch_name_list)), float('-inf')).to(self.device)
                    all_logits[:, :topk_object, custom_object_categories_idx] = top_object_outputs['pred_logits']
                    all_logits[:, topk_object:, custom_part_categories_idx] = top_part_outputs['pred_logits']
                else:
                    all_logits = torch.full((1, test_topk, self.num_classes[task]), float('-inf')).to(self.device)
                    all_logits[:, :topk_object, self.object_category_index_mapper[object_level_task]] = top_object_outputs['pred_logits']
                    all_logits[:, topk_object:, self.part_category_index_mapper[part_level_task]] = top_part_outputs['pred_logits']
                outputs[key] = all_logits
            else:
                outputs[key] = torch.cat([top_object_outputs[key], top_part_outputs[key]], dim=1)
        return outputs
    
    def hierarchical_topk(self, task, topk):
        if self.visualize:
            topk_object = topk
            topk_part = self.num_part_queries
        else:
            if task == 'seginw_House-Parts':
                topk_object = 0
                topk_part = 100
            elif 'seginw' in task and task not in ['seginw_Airplane-Parts', 'seginw_Bottles']:
                topk_object = 100
                topk_part = 0
            else:
                topk_object = self.test_object_inst_num[task]
                topk_part = self.test_part_inst_num[task]
        return topk_object, topk_part
    
    def inference_topk(self, task, topk):
        if 'seginw' in task and not self.visualize:
            test_topk = 100
        else:
            test_topk = self.test_topk_per_image[task] if not self.visualize else topk + self.num_part_queries
        return test_topk

    def _get_src_permutation_idx(self, indices):
        # permute predictions following indices
        batch_idx = torch.cat([torch.full_like(src, i) for i, (src, _) in enumerate(indices)])
        src_idx = torch.cat([src for (src, _) in indices])
        return batch_idx, src_idx

    def _get_tgt_permutation_idx(self, indices):
        # permute targets following indices
        batch_idx = torch.cat([torch.full_like(tgt, i) for i, (_, tgt) in enumerate(indices)])
        tgt_idx = torch.cat([tgt for (_, tgt) in indices])
        return batch_idx, tgt_idx
    