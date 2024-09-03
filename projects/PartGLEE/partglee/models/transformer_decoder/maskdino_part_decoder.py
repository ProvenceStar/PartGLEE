# Copyright (c) 2024 ByteDance. All Rights Reserved.
# PartGLEE part decoder model.
# PartGLEE: A Foundation Model for Recognizing and Parsing Any Objects (ECCV 2024)
# https://arxiv.org/abs/2407.16696
# Modified from MaskDINO https://github.com/IDEA-Research/MaskDINO by Junyi Li

import logging
import fvcore.nn.weight_init as weight_init
import torch
from torch import nn
from torch.nn import functional as F

from detectron2.config import configurable
from detectron2.layers import Conv2d
from detectron2.utils.registry import Registry
from detectron2.structures import BitMasks
from timm.models.layers import trunc_normal_
from .dino_decoder import TransformerDecoder, DeformableTransformerDecoderLayer
from .._CONSTANTS import PACO_CATEGORIES, OPENIMAGES_PART_CATEGORIES
from ...utils.utils import MLP, gen_encoder_output_proposals, inverse_sigmoid
from ...utils import box_ops

TRANSFORMER_DECODER_REGISTRY = Registry("TRANSFORMER_MODULE")
TRANSFORMER_DECODER_REGISTRY.__doc__ = """
Registry for transformer module in MaskDINO.
"""

def build_part_transformer_decoder(cfg, in_channels, lang_encoder, mask_classification=True, is_part_decoder=False):
    """
    Build a instance embedding branch from `cfg.MODEL.INS_EMBED_HEAD.NAME`.
    """
    name = cfg.MODEL.MaskDINO.PART_TRANSFORMER_DECODER_NAME
    return TRANSFORMER_DECODER_REGISTRY.get(name)(cfg, in_channels, lang_encoder, mask_classification,is_part_decoder)


@TRANSFORMER_DECODER_REGISTRY.register()
class MaskDINOPartDecoder(nn.Module):
    @configurable
    def __init__(
            self,
            in_channels,
            lang_encoder,
            mask_classification=True,
            is_part_decoder=False,
            *,
            num_classes: int,
            hidden_dim: int,
            num_queries: int,
            nheads: int,
            dim_feedforward: int,
            dec_layers: int,
            mask_dim: int,
            dim_projection: int,
            enforce_input_project: bool,
            two_stage: bool,
            dn: str,
            noise_scale:float,
            dn_num:int,
            initialize_box_type:bool,
            initial_pred:bool,
            learn_tgt: bool,
            total_num_feature_levels: int = 4,
            dropout: float = 0.0,
            activation: str = 'relu',
            nhead: int = 8,
            dec_n_points: int = 4,
            return_intermediate_dec: bool = True,
            query_dim: int = 4,
            dec_layer_share: bool = False,
            semantic_ce_loss: bool = False,
            cross_track_layer: bool = False,
            unify_object_part: bool = False,
            use_qformer: bool = False,
    ):
        """
        NOTE: this interface is experimental.
        Args:
            in_channels: channels of the input features
            mask_classification: whether to add mask classifier or not
            num_classes: number of classes
            hidden_dim: Transformer feature dimension
            num_queries: number of queries
            nheads: number of heads
            dim_feedforward: feature dimension in feedforward network
            enc_layers: number of Transformer encoder layers
            dec_layers: number of Transformer decoder layers
            pre_norm: whether to use pre-LayerNorm or not
            mask_dim: mask feature dimension
            enforce_input_project: add input project 1x1 conv even if input
                channels and hidden dim is identical
            d_model: transformer dimension
            dropout: dropout rate
            activation: activation function
            nhead: num heads in multi-head attention
            dec_n_points: number of sampling points in decoder
            return_intermediate_dec: return the intermediate results of decoder
            query_dim: 4 -> (x, y, w, h)
            dec_layer_share: whether to share each decoder layer
            semantic_ce_loss: use ce loss for semantic segmentation
        """
        super().__init__()

        assert mask_classification, "Only support mask classification model"
        self.mask_classification = mask_classification
        self.num_feature_levels = total_num_feature_levels
        self.initial_pred = initial_pred

        self.lang_encoder = lang_encoder
        
        # define Transformer decoder here
        self.dn=dn
        self.learn_tgt = learn_tgt
        self.noise_scale=noise_scale
        self.dn_num=dn_num
        self.num_heads = nheads
        self.num_layers = dec_layers
        self.two_stage=two_stage
        self.initialize_box_type = initialize_box_type
        self.total_num_feature_levels = total_num_feature_levels

        self.num_queries = num_queries
        self.semantic_ce_loss = semantic_ce_loss
        # learnable query features
        if not two_stage or self.learn_tgt:
            self.query_feat = nn.Embedding(num_queries, hidden_dim)
        if not two_stage and initialize_box_type == 'no':
            self.query_embed = nn.Embedding(num_queries, 4)
        if two_stage:
            self.enc_output = nn.Linear(hidden_dim, hidden_dim)
            self.enc_output_norm = nn.LayerNorm(hidden_dim)

        self.input_proj = nn.ModuleList()
        for _ in range(self.num_feature_levels):
            if in_channels != hidden_dim or enforce_input_project:
                self.input_proj.append(Conv2d(in_channels, hidden_dim, kernel_size=1))
                weight_init.c2_xavier_fill(self.input_proj[-1])
            else:
                self.input_proj.append(nn.Sequential())
        self.num_classes = {
            'obj365':100,
            'obj365_clip':100, 
            'lvis':100, 
            'paco':531,
            'paco_part':456,
            'openimage_joint_part': 13,
            'ade20k': 278,
            'ade20k_part': 234,
            'ade20k_base_part': 176,
            'pascalvoc': 136,
            'pascalvoc_part': 116,
            'pascalvoc_base_part': 74,
            'partimagenet_joint': 51,
            'partimagenet_joint_part': 40,
            'partimagenet_parsed_part': 40,
            'sa1b_joint': 2,
            'sa1b_joint_part': 1,
            'pascal_joint':113,
            'pascal_joint_part': 93,
            'pascal_open_vocabulary_part': 77,
            'pascal_part_parsed_part': 93,
            # 'pascal_part': 93,
            'coco_panoptic': 133,
            'openimage':100,
            'lvis_clip':100,
            'openimage_clip':100,
            'grit':100,
            'vg':200,
            'vg_joint_part': 200,
            'coco':80,
            'voc': 20,
            'coco_clip':80,
            'grounding':1, 
            'rvos':1, 
            'sa1b':1, 
            'sa1b_clip':1,
            'bdd_det':10,
            'bdd_inst':8,
            'ytvis19':40,
            'image_yt19':40, 
            'image_yt21':40,
            'bdd_track_seg':8,
            'bdd_track_box':8,
            'ovis':25,
            'image_o':25,
            'ytvis21':40,
            'uvo_video': 81,
            'ytbvos':1,
            'seginw_Brain-Tumor': 1,
        }
        self.part_datasets_part_category_idx = {
            'paco_part': [idx for idx in range(len(PACO_CATEGORIES)) if PACO_CATEGORIES[idx]['supercategory']=='PART'],
            'openimage_joint_part': [cat['id'] - 1 for cat in OPENIMAGES_PART_CATEGORIES],
            'ade20k_part': [idx + 44 for idx in range(234)],
            'ade20k_base_part': [idx + 33 for idx in range(176)],
            'pascalvoc_part': [idx + 20 for idx in range(116)],
            'pascalvoc_base_part': [idx + 15 for idx in range(74)],
            'partimagenet_joint_part': [idx + 11 for idx in range(40)],
            'partimagenet_parsed_part': [idx + 11 for idx in range(40)],
            'partimagenet_val_part': [idx + 11 for idx in range(40)],
            'sa1b_joint_part': [1],
            'pascal_joint_part': [idx + 20 for idx in range(93)],
            'pascal_open_vocabulary_part': [idx + 12 for idx in range(77)],
            'pascal_part_parsed_part': [idx + 20 for idx in range(93)],
            'pascal_open_vocabulary_val_part': [idx + 14 for idx in range(93)],
            'seginw_House-Parts_part': [idx + 1 for idx in range(12)],
            'seginw_Airplane-Parts_part': [idx + 1 for idx in range(4)],
            'seginw_Bottles_part': [2],
            'vg_joint_part': [idx + 200 for idx in range(200)],
        }
        # output FFNs
        assert self.mask_classification, "why not class embedding?"
        # if self.mask_classification:
        #     if self.semantic_ce_loss:
        #         self.class_embed = nn.Linear(hidden_dim, num_classes+1)
        #     else:
        #         self.class_embed = nn.Linear(hidden_dim, num_classes)
        
        self.confidence_score =  MLP(hidden_dim, hidden_dim, 1, 2)
        self.category_embed = nn.Parameter(torch.rand(hidden_dim, dim_projection))
        # trunc_normal_(self.category_embed, std=.02)
        # self.track_embed = MLP(hidden_dim, hidden_dim, hidden_dim, 3)

        self.coco_label_enc = nn.Embedding(80,hidden_dim)
        self.voc_label_enc = nn.Embedding(20, hidden_dim)
        # self.pascal_part_label_enc = nn.Embedding(93, hidden_dim)
        # self.pascal_part_open_vocabulary_label_enc = nn.Embedding(77, hidden_dim)
        self.coco_panoptic_label_enc = nn.Embedding(133, hidden_dim)
        self.openimage_part_label_enc = nn.Embedding(13, hidden_dim)
        self.obj365_label_enc = nn.Embedding(100, hidden_dim)
        self.paco_label_enc = nn.Embedding(531, hidden_dim)
        self.paco_part_label_enc = nn.Embedding(456, hidden_dim)
        self.ade20k_label_enc = nn.Embedding(278, hidden_dim)
        self.ade20k_part_label_enc = nn.Embedding(234, hidden_dim)
        self.ade20k_base_part_label_enc = nn.Embedding(176, hidden_dim)
        self.pascalvoc_label_enc = nn.Embedding(136, hidden_dim)
        self.pascalvoc_part_label_enc = nn.Embedding(116, hidden_dim)
        self.pascalvoc_base_part_label_enc = nn.Embedding(74, hidden_dim)
        self.partimagenet_joint_label_enc = nn.Embedding(51, hidden_dim)
        self.partimagenet_joint_part_label_enc = nn.Embedding(40, hidden_dim)
        self.sa1b_joint_label_enc = nn.Embedding(2, hidden_dim)
        self.sa1b_joint_part_label_enc = nn.Embedding(1, hidden_dim)
        self.pascal_joint_label_enc = nn.Embedding(113, hidden_dim)
        self.pascal_joint_part_label_enc = nn.Embedding(113, hidden_dim)
        self.pascal_open_vocabulary_part_label_enc = nn.Embedding(77, hidden_dim)
        self.vg_label_enc = nn.Embedding(200, hidden_dim)
        self.grounding_label_enc = nn.Embedding(1,hidden_dim)
        self.sa1b_part_label_enc = nn.Embedding(2,hidden_dim)
        self.ytvis19_label_enc = nn.Embedding(40,hidden_dim)
        self.ytvis21_label_enc = nn.Embedding(40,hidden_dim)
        self.ovis_label_enc = nn.Embedding(25,hidden_dim)
        self.uvo_label_enc = nn.Embedding(81,hidden_dim)
        self.bdd_det = nn.Embedding(10,hidden_dim)
        self.bdd_inst = nn.Embedding(8,hidden_dim)
        # self.seginw_brain_tumor_label_enc = nn.Embedding(1, hidden_dim)
        self.label_enc = {
            'coco': self.coco_label_enc, 
            'voc': self.voc_label_enc,
            'coco_clip': self.coco_label_enc, 
            'coconomask': self.coco_label_enc,
            'obj365': self.obj365_label_enc,
            'lvis': self.obj365_label_enc,
            'paco': self.paco_label_enc,
            'paco_part': self.paco_part_label_enc,
            'ade20k': self.ade20k_label_enc,
            'ade20k_part': self.ade20k_part_label_enc,
            'ade20k_base_part': self.ade20k_base_part_label_enc,
            'pascalvoc': self.pascalvoc_label_enc,
            'pascalvoc_part': self.pascalvoc_part_label_enc,
            'pascalvoc_base_part': self.pascalvoc_base_part_label_enc,
            'partimagenet_joint': self.partimagenet_joint_label_enc,
            'partimagenet_joint_part': self.partimagenet_joint_part_label_enc,
            'partimagenet_parsed_part': self.partimagenet_joint_part_label_enc,
            'sa1b_joint': self.sa1b_joint_label_enc,
            'sa1b_joint_part': self.sa1b_joint_part_label_enc,
            'pascal_joint': self.pascal_joint_label_enc,
            'pascal_joint_part': self.pascal_joint_part_label_enc,
            'pascal_open_vocabulary_part': self.pascal_open_vocabulary_part_label_enc,
            'pascal_part_parsed_part': self.pascal_joint_part_label_enc,
            # 'pascal_part': self.pascal_part_label_enc,
            'coco_panoptic': self.coco_panoptic_label_enc,
            'openimage': self.obj365_label_enc,
            'openimage_joint_part': self.openimage_part_label_enc,
            'grit': self.obj365_label_enc,
            'vg': self.vg_label_enc,
            'vg_joint_part': self.vg_label_enc,
            'obj365_clip': self.obj365_label_enc,
            'lvis_clip': self.obj365_label_enc,
            'openimage_clip': self.obj365_label_enc,
            'bdd_det':self.bdd_det,
            'bdd_inst':self.bdd_inst,
            'bdd_track_seg':self.bdd_inst, 
            'bdd_track_box':self.bdd_inst,
            'sa1b': self.grounding_label_enc,
            'sa1b_clip': self.grounding_label_enc,
            'grounding': self.grounding_label_enc,
            'rvos': self.grounding_label_enc,
            'uvo_video':self.uvo_label_enc,
            'ytvis19':self.ytvis19_label_enc,
            'image_yt19': self.ytvis19_label_enc,
            'ytvis21':self.ytvis21_label_enc,
            'image_yt21':self.ytvis21_label_enc,
            'ovis':self.ovis_label_enc,
            'image_o': self.ovis_label_enc,
            'burst':self.grounding_label_enc,
            'ytbvos':self.grounding_label_enc,
        #     'seginw_Brain-Tumor': self.seginw_brain_tumor_label_enc,
            }

        self.mask_embed = MLP(hidden_dim, hidden_dim, mask_dim, 3)
        
        # init decoder
        self.decoder_norm = decoder_norm = nn.LayerNorm(hidden_dim)
        decoder_layer = DeformableTransformerDecoderLayer(hidden_dim, dim_feedforward,
                                                          dropout, activation,
                                                          self.num_feature_levels, nhead, dec_n_points)
        self.decoder = TransformerDecoder(decoder_layer, self.num_layers, decoder_norm,
                                          return_intermediate=return_intermediate_dec,
                                          d_model=hidden_dim, query_dim=query_dim,
                                          num_feature_levels=self.num_feature_levels,
                                          dec_layer_share=dec_layer_share,
                                          cross_track_layer = cross_track_layer,
                                          n_levels=self.num_feature_levels, n_heads=nhead, n_points=dec_n_points
                                          )
        self.cross_track_layer = cross_track_layer
        self.hidden_dim = hidden_dim
        self._bbox_embed = _bbox_embed = MLP(hidden_dim, hidden_dim, 4, 3)
        
        nn.init.constant_(_bbox_embed.layers[-1].weight.data, 0)
        nn.init.constant_(_bbox_embed.layers[-1].bias.data, 0)
        box_embed_layerlist = [_bbox_embed for i in range(self.num_layers)]  # share box prediction each layer
        self.bbox_embed = nn.ModuleList(box_embed_layerlist)
        self.decoder.bbox_embed = self.bbox_embed
        
        # self.part_category_embed = nn.Embedding(1, 256)     # Shift CLIP Embeddings from Object to Part
        
        self.is_part_decoder = is_part_decoder
        self.unify_object_part = unify_object_part
        self.use_qformer = use_qformer
        
        self.object_level_datasets = ['coco', 'coco_panoptic', 'lvis', 
          'obj365',   'openimage',  'vg', 'grounding',  'ytbvos', 'rvos', 'sa1b', 'ytvis19', 'image_yt19', 'ytvis21', 'image_yt21','ovis', 'image_o', 'uvo_video','bdd_det', 'bdd_inst','bdd_track_seg', 'bdd_track_box',
        'voc', 'seginw_Cows', 'seginw_Elephants', 'seginw_Brain-Tumor', 'seginw_Chicken', 'seginw_Rail', 'seginw_Electric-Shaver', 'seginw_Fruits', 'seginw_Garbage', 'seginw_Ginger-Garlic', 'seginw_Hand', 'seginw_Hand-Metal', 'seginw_HouseHold-Items', 'seginw_Nutterfly-Squireel', \
                                      'seginw_Phones', 'seginw_Poles', 'seginw_Puppies', 'seginw_Salmon-Fillet', 'seginw_Strawberry', 'seginw_Tablets', 'seginw_Toolkits', 'seginw_Trash', 'seginw_Watermelon', 'odinw13_PascalVOC', ]


    @classmethod
    def from_config(cls, cfg, in_channels, lang_encoder, mask_classification, is_part_decoder):
        ret = {}
        ret["in_channels"] = in_channels
        ret["lang_encoder"] = lang_encoder
        ret["mask_classification"] = mask_classification
        ret["is_part_decoder"] = is_part_decoder
        ret["dim_projection"] = cfg.MODEL.DIM_PROJ
        ret["num_classes"] = cfg.MODEL.SEM_SEG_HEAD.NUM_CLASSES
        ret["hidden_dim"] = cfg.MODEL.MaskDINO.HIDDEN_DIM
        ret["num_queries"] = cfg.MODEL.MaskDINO.NUM_OBJECT_QUERIES
        # Transformer parameters:
        ret["nheads"] = cfg.MODEL.MaskDINO.NHEADS
        ret["dim_feedforward"] = cfg.MODEL.MaskDINO.DIM_FEEDFORWARD
        ret["dec_layers"] = cfg.MODEL.MaskDINO.DEC_LAYERS
        ret["enforce_input_project"] = cfg.MODEL.MaskDINO.ENFORCE_INPUT_PROJ
        ret["mask_dim"] = cfg.MODEL.SEM_SEG_HEAD.MASK_DIM
        ret["two_stage"] =cfg.MODEL.MaskDINO.TWO_STAGE
        ret["initialize_box_type"] = cfg.MODEL.MaskDINO.INITIALIZE_BOX_TYPE  # ['no', 'bitmask', 'mask2box']
        ret["dn"]=cfg.MODEL.MaskDINO.DN
        ret["noise_scale"] =cfg.MODEL.MaskDINO.DN_NOISE_SCALE
        ret["dn_num"] =cfg.MODEL.MaskDINO.DN_NUM
        ret["initial_pred"] =cfg.MODEL.MaskDINO.INITIAL_PRED
        ret["learn_tgt"] = cfg.MODEL.MaskDINO.LEARN_TGT
        ret["total_num_feature_levels"] = cfg.MODEL.SEM_SEG_HEAD.TOTAL_NUM_FEATURE_LEVELS
        ret["semantic_ce_loss"] = cfg.MODEL.MaskDINO.TEST.SEMANTIC_ON and cfg.MODEL.MaskDINO.SEMANTIC_CE_LOSS and ~cfg.MODEL.MaskDINO.TEST.PANOPTIC_ON
        ret["cross_track_layer"] = cfg.MODEL.CROSS_TRACK
        ret["unify_object_part"] = cfg.MODEL.MaskDINO.UNIFY_OBJECT_PART
        ret["use_qformer"] = cfg.MODEL.MaskDINO.Q_FORMER
        return ret

    def prepare_for_dn(self, targets, tgt, refpoint_emb, batch_size, task, topk, num_part_queries):
        """
        modified from dn-detr. You can refer to dn-detr
        https://github.com/IDEA-Research/DN-DETR/blob/main/models/dn_dab_deformable_detr/dn_components.py
        for more details
            :param dn_args: scalar, noise_scale
            :param tgt: original tgt (content) in the matching part
            :param refpoint_emb: positional anchor queries in the matching part
            :param batch_size: bs
            """
        if self.training:
            scalar, noise_scale = self.dn_num,self.noise_scale

            known = [(torch.ones_like(t['labels'])).cuda() for t in targets]
            know_idx = [torch.nonzero(t) for t in known]
            known_num = [sum(k) for k in known]

            # use fix number of dn queries
            if max(known_num)>0:
                scalar = scalar//(int(max(known_num)))
            else:
                scalar = 0
            if scalar == 0:
                input_query_label = None
                input_query_bbox = None
                attn_mask = None
                mask_dict = None
                return input_query_label, input_query_bbox, attn_mask, mask_dict

            # can be modified to selectively denosie some label or boxes; also known label prediction
            unmask_bbox = unmask_label = torch.cat(known)
            labels = torch.cat([t['labels'] for t in targets])
            boxes = torch.cat([t['boxes'] for t in targets])
            batch_idx = torch.cat([torch.full_like(t['labels'].long(), i) for i, t in enumerate(targets)])
            # known
            known_indice = torch.nonzero(unmask_label + unmask_bbox)
            known_indice = known_indice.view(-1)

            # noise
            known_indice = known_indice.repeat(scalar, 1).view(-1)
            known_labels = labels.repeat(scalar, 1).view(-1)
            known_bid = batch_idx.repeat(scalar, 1).view(-1)
            known_bboxs = boxes.repeat(scalar, 1)
            known_labels_expaned = known_labels.clone().long()
            known_bbox_expand = known_bboxs.clone()

            # noise on the label
            if noise_scale > 0:
                p = torch.rand_like(known_labels_expaned.float())
                chosen_indice = torch.nonzero(p < (noise_scale * 0.5)).view(-1).long()  # half of bbox prob
                new_label = torch.randint_like(chosen_indice, 0, self.num_classes[task])  # randomly put a new one here
                
                if known_labels_expaned.dtype != new_label.dtype:
                    known_labels_expaned = known_labels_expaned.to(new_label.dtype)
                    
                known_labels_expaned.scatter_(0, chosen_indice, new_label)
            if noise_scale > 0:
                diff = torch.zeros_like(known_bbox_expand)
                diff[:, :2] = known_bbox_expand[:, 2:] / 2
                diff[:, 2:] = known_bbox_expand[:, 2:]
                known_bbox_expand += torch.mul((torch.rand_like(known_bbox_expand) * 2 - 1.0),
                                               diff).cuda() * noise_scale
                known_bbox_expand = known_bbox_expand.clamp(min=0.0, max=1.0)
            m = known_labels_expaned.long().to('cuda')
            input_label_embed = self.label_enc[task](m)
            input_bbox_embed = inverse_sigmoid(known_bbox_expand)
            single_pad = int(max(known_num))
            pad_size = int(single_pad * scalar)

            padding_label = torch.zeros(pad_size, self.hidden_dim).cuda()
            padding_bbox = torch.zeros(pad_size, 4).cuda()

            if not refpoint_emb is None:
                input_query_label = torch.cat([padding_label, tgt], dim=0).repeat(batch_size, 1, 1)
                input_query_bbox = torch.cat([padding_bbox, refpoint_emb], dim=0).repeat(batch_size, 1, 1)
            else:
                input_query_label=padding_label.repeat(batch_size, 1, 1)
                input_query_bbox = padding_bbox.repeat(batch_size, 1, 1)

            # map
            map_known_indice = torch.tensor([]).to('cuda')
            if len(known_num):
                map_known_indice = torch.cat([torch.tensor(range(num)) for num in known_num])  # [1,2, 1,2,3]
                map_known_indice = torch.cat([map_known_indice + single_pad * i for i in range(scalar)]).long()
            if len(known_bid):
                input_query_label[(known_bid.long(), map_known_indice)] = input_label_embed
                input_query_bbox[(known_bid.long(), map_known_indice)] = input_bbox_embed
            
            if self.unify_object_part and self.use_qformer:
                tgt_size = pad_size + topk*num_part_queries
            else:
                tgt_size = pad_size + self.num_queries
            attn_mask = torch.ones(tgt_size, tgt_size).to('cuda') < 0
            # match query cannot see the reconstruct
            attn_mask[pad_size:, :pad_size] = True
            # reconstruct cannot see each other
            for i in range(scalar):
                if i == 0:
                    attn_mask[single_pad * i:single_pad * (i + 1), single_pad * (i + 1):pad_size] = True
                if i == scalar - 1:
                    attn_mask[single_pad * i:single_pad * (i + 1), :single_pad * i] = True
                else:
                    attn_mask[single_pad * i:single_pad * (i + 1), single_pad * (i + 1):pad_size] = True
                    attn_mask[single_pad * i:single_pad * (i + 1), :single_pad * i] = True
            mask_dict = {
                'known_indice': torch.as_tensor(known_indice).long(),
                'batch_idx': torch.as_tensor(batch_idx).long(),
                'map_known_indice': torch.as_tensor(map_known_indice).long(),
                'known_lbs_bboxes': (known_labels, known_bboxs),
                'know_idx': know_idx,
                'pad_size': pad_size,
                'scalar': scalar,
            }
        else:
            if not refpoint_emb is None:
                input_query_label = tgt.repeat(batch_size, 1, 1)
                input_query_bbox = refpoint_emb.repeat(batch_size, 1, 1)
            else:
                input_query_label=None
                input_query_bbox=None
            attn_mask = None
            mask_dict=None

        # 100*batch*256
        if not input_query_bbox is None:
            input_query_label = input_query_label
            input_query_bbox = input_query_bbox

        return input_query_label,input_query_bbox,attn_mask,mask_dict

    def dn_post_process(self,outputs_class,outputs_score,outputs_coord,mask_dict,outputs_mask):
        """
            post process of dn after output from the transformer
            put the dn part in the mask_dict
            """
        assert mask_dict['pad_size'] > 0
        output_known_class = outputs_class[:, :, :mask_dict['pad_size'], :]
        outputs_class = outputs_class[:, :, mask_dict['pad_size']:, :]
        output_known_score = outputs_score[:, :, :mask_dict['pad_size'], :]
        outputs_score = outputs_score[:, :, mask_dict['pad_size']:, :]

        output_known_coord = outputs_coord[:, :, :mask_dict['pad_size'], :]
        outputs_coord = outputs_coord[:, :, mask_dict['pad_size']:, :]
        if outputs_mask is not None:
            output_known_mask = outputs_mask[:, :, :mask_dict['pad_size'], :]
            outputs_mask = outputs_mask[:, :, mask_dict['pad_size']:, :]
        out = {'pred_logits': output_known_class[-1], 'pred_scores':output_known_score[-1],'pred_boxes': output_known_coord[-1],'pred_masks': output_known_mask[-1]}

        out['aux_outputs'] = self._set_aux_loss(output_known_class, output_known_score, output_known_mask, output_known_coord)
        mask_dict['output_known_lbs_bboxes']=out
        return outputs_class, outputs_score, outputs_coord, outputs_mask

    def get_valid_ratio(self, mask):
        _, H, W = mask.shape
        valid_H = torch.sum(~mask[:, :, 0], 1)
        valid_W = torch.sum(~mask[:, 0, :], 1)
        valid_ratio_h = valid_H.float() / H
        valid_ratio_w = valid_W.float() / W
        valid_ratio = torch.stack([valid_ratio_w, valid_ratio_h], -1)
        return valid_ratio

    def pred_box(self, reference, hs, ref0=None):
        """
        :param reference: reference box coordinates from each decoder layer
        :param hs: content
        :param ref0: whether there are prediction from the first layer
        """
        device = reference[0].device

        if ref0 is None:
            outputs_coord_list = []
        else:
            outputs_coord_list = [ref0.to(device)]
        for dec_lid, (layer_ref_sig, layer_bbox_embed, layer_hs) in enumerate(zip(reference[:-1], self.bbox_embed, hs)):
            layer_delta_unsig = layer_bbox_embed(layer_hs).to(device)
            layer_outputs_unsig = layer_delta_unsig + inverse_sigmoid(layer_ref_sig).to(device)
            layer_outputs_unsig = layer_outputs_unsig.sigmoid()
            outputs_coord_list.append(layer_outputs_unsig)
        outputs_coord_list = torch.stack(outputs_coord_list)
        return outputs_coord_list

    def forward(self, x, mask_features, extra, task, masks, targets=None, part_queries=None, topk=None, num_part_queries=None, custom_part_categories_idx=None):
        """
        :param x: input, a list of multi-scale feature
        :param mask_features: is the per-pixel embeddings with resolution 1/4 of the original image,
        obtained by fusing backbone encoder encoded features. This is used to produce binary masks.
        :param masks: mask in the original image
        :param targets: used for denoising training
        """

        if 'spatial_query_pos_mask' in extra:
            visual_P = True
        else:
            visual_P = False

        assert len(x) == self.num_feature_levels
        device = x[0].device
        size_list = []
        # disable mask, it does not affect performance
        enable_mask = 0
        if masks is not None:
            for src in x:
                if src.size(2) % 32 or src.size(3) % 32:
                    enable_mask = 1
        if enable_mask == 0:
            masks = [torch.zeros((src.size(0), src.size(2), src.size(3)), device=src.device, dtype=torch.bool) for src in x]
        src_flatten = []
        mask_flatten = []
        spatial_shapes = []
        for i in range(self.num_feature_levels):
            idx=self.num_feature_levels-1-i
            bs, c , h, w=x[idx].shape
            size_list.append(x[i].shape[-2:])
            spatial_shapes.append(x[idx].shape[-2:])
            src_flatten.append(self.input_proj[idx](x[idx]).flatten(2).transpose(1, 2))
            mask_flatten.append(masks[i].flatten(1))
        src_flatten = torch.cat(src_flatten, 1)  # bs, \sum{hxw}, c
        mask_flatten = torch.cat(mask_flatten, 1)  # bs, \sum{hxw}
        spatial_shapes = torch.as_tensor(spatial_shapes, dtype=torch.long, device=src_flatten.device)
        level_start_index = torch.cat((spatial_shapes.new_zeros((1,)), spatial_shapes.prod(1).cumsum(0)[:-1]))
        valid_ratios = torch.stack([self.get_valid_ratio(m) for m in masks], 1)
        
        predictions_federate = []
        predictions_score = []
        predictions_class = []
        predictions_mask = []
        if self.two_stage:
            if self.use_qformer:
                assert part_queries is not None, "Unify object part must input part queries"
                part_queries_feat = self.enc_output_norm(self.enc_output(part_queries))
                enc_outputs_coord_unselected = self._bbox_embed(part_queries_feat)  # (bs, h*w, 4) unsigmoid
                # part queries
                refpoint_embed_undetach = enc_outputs_coord_unselected  # unsigmoid
                
                refpoint_embed = refpoint_embed_undetach.detach() #[bz,num_q,4]
                # refpoint_embed = refpoint_embed_undetach #[bz,num_q,4]
                
                tgt_undetach = part_queries_feat  # unsigmoid  #[bz,num_q,256]
                    
                conf_score, outputs_class, outputs_mask,_ = self.forward_prediction_heads(tgt_undetach.transpose(0, 1), mask_features, task, extra, mask_dict = None, custom_part_categories_idx=custom_part_categories_idx)
                # Q-Former do not detach for optimization
                tgt = tgt_undetach
                # tgt = tgt_undetach.detach()
            
            else:
                output_memory, output_proposals = gen_encoder_output_proposals(src_flatten, mask_flatten, spatial_shapes)
                output_memory = self.enc_output_norm(self.enc_output(output_memory))
                
                if task in ['grounding','rvos']:
                    class_embed = output_memory @ self.category_embed 
                    enc_outputs_class_unselected = torch.einsum("bqc,bc->bq", class_embed, extra['grounding_class']).unsqueeze(-1) #[bz,numq,1]
                
                else:
                    class_embed = output_memory @ self.category_embed  # [bz,num_q,projectdim]
                    if self.unify_object_part and task not in self.object_level_datasets:
                        if 'custom' in task:
                            assert custom_part_categories_idx != None, "Customized Task must input part categories indices!"
                            part_category_idx = custom_part_categories_idx
                        else:
                            part_category_idx = self.part_datasets_part_category_idx[task]
                        part_class_embed = extra['class_embeddings'][part_category_idx, :]
                        enc_outputs_class_unselected = torch.einsum("bqc,nc->bqn", class_embed, part_class_embed)  #[bz,n,80]
                    else:
                        enc_outputs_class_unselected = torch.einsum("bqc,nc->bqn", class_embed, extra['class_embeddings'])  #[bz,n,80]
                    
                enc_outputs_coord_unselected = self._bbox_embed(
                    output_memory) + output_proposals  # (bs, \sum{hw}, 4) unsigmoid
                
                # object queries
                topk = self.num_queries
                topk_proposals = torch.topk(enc_outputs_class_unselected.max(-1)[0], topk, dim=1)[1]
                refpoint_embed_undetach = torch.gather(enc_outputs_coord_unselected, 1,
                                                    topk_proposals.unsqueeze(-1).repeat(1, 1, 4))  # unsigmoid
                refpoint_embed = refpoint_embed_undetach.detach() #[bz,num_q,4]
                
                tgt_undetach = torch.gather(output_memory, 1,
                                    topk_proposals.unsqueeze(-1).repeat(1, 1, self.hidden_dim))  # unsigmoid  #[bz,num_q,256]
                    
                conf_score, outputs_class, outputs_mask,_ = self.forward_prediction_heads(tgt_undetach.transpose(0, 1), mask_features, task, extra, mask_dict = None, custom_part_categories_idx=custom_part_categories_idx)
                tgt = tgt_undetach.detach()
            
            if self.learn_tgt:
                tgt = self.query_feat.weight[None].repeat(bs, 1, 1)
            interm_outputs=dict()
            interm_outputs['pred_logits'] = outputs_class
            interm_outputs['pred_scores'] = conf_score
            interm_outputs['pred_boxes'] = refpoint_embed_undetach.sigmoid()
            interm_outputs['pred_masks'] = outputs_mask

            # if self.initialize_box_type != 'no':
            #     # convert masks into boxes to better initialize box in the decoder
            #     assert self.initial_pred
            #     flaten_mask = outputs_mask.detach().flatten(0, 1)
            #     h, w = outputs_mask.shape[-2:]
            #     if self.initialize_box_type == 'bitmask':  # slower, but more accurate
            #         refpoint_embed = BitMasks(flaten_mask > 0).get_bounding_boxes().tensor.to(device)
            #     elif self.initialize_box_type == 'mask2box':  # faster conversion
            #         refpoint_embed = box_ops.masks_to_boxes(flaten_mask > 0).to(device)
            #     else:
            #         assert NotImplementedError
            #     refpoint_embed = box_ops.box_xyxy_to_cxcywh(refpoint_embed) / torch.as_tensor([w, h, w, h],
            #                                                                                   dtype=torch.float).to(device)
            #     refpoint_embed = refpoint_embed.reshape(outputs_mask.shape[0], outputs_mask.shape[1], 4)
            #     refpoint_embed = inverse_sigmoid(refpoint_embed)
        elif not self.two_stage:
            tgt = self.query_feat.weight[None].repeat(bs, 1, 1)
            refpoint_embed = self.query_embed.weight[None].repeat(bs, 1, 1)
        # import pdb;pdb.set_trace()
        tgt_mask = None
        mask_dict = None
        if self.dn != "no" and self.training:
            assert targets is not None
            input_query_label, input_query_bbox, tgt_mask, mask_dict = \
                self.prepare_for_dn(targets, None, None, x[0].shape[0], task, topk, num_part_queries)
            if mask_dict is not None:
                tgt=torch.cat([input_query_label, tgt],dim=1)
        # import pdb;pdb.set_trace()
        # direct prediction from the matching and denoising part in the begining
        if self.initial_pred:
            conf_score, outputs_class, outputs_mask, pred_federat = self.forward_prediction_heads(tgt.transpose(0, 1), mask_features, task, extra, mask_dict, self.training, custom_part_categories_idx=custom_part_categories_idx)
            predictions_score.append(conf_score)
            predictions_class.append(outputs_class)
            predictions_mask.append(outputs_mask)
            predictions_federate.append(pred_federat)
        if self.dn != "no" and self.training and mask_dict is not None:
            refpoint_embed=torch.cat([input_query_bbox,refpoint_embed],dim=1)
            
        hs, references, cross_track_embed = self.decoder(
            tgt=tgt.transpose(0, 1),
            memory=src_flatten.transpose(0, 1),
            memory_key_padding_mask=mask_flatten,
            pos=None,
            refpoints_unsigmoid=refpoint_embed.transpose(0, 1),
            level_start_index=level_start_index,
            spatial_shapes=spatial_shapes,
            valid_ratios=valid_ratios,
            tgt_mask=tgt_mask,
            task=task,
            extra=extra,
        )
        
        # import pdb;pdb.set_trace()
        for i, output in enumerate(hs):
            conf_score, outputs_class, outputs_mask, pred_federat = self.forward_prediction_heads(output.transpose(0, 1), mask_features, task, extra, mask_dict, self.training or (i == len(hs)-1), custom_part_categories_idx=custom_part_categories_idx)
            predictions_score.append(conf_score)
            predictions_class.append(outputs_class)
            predictions_mask.append(outputs_mask)
            predictions_federate.append(pred_federat)

        # iteratively box prediction
        if self.initial_pred:
            out_boxes = self.pred_box(references, hs, refpoint_embed.sigmoid())
            assert len(predictions_class) == self.num_layers + 1
        else:
            out_boxes = self.pred_box(references, hs)
        # import pdb;pdb.set_trace()
        if mask_dict is not None:
            predictions_mask=torch.stack(predictions_mask)
            predictions_class=torch.stack(predictions_class)
            predictions_score = torch.stack(predictions_score)
            predictions_class, predictions_score, out_boxes, predictions_mask=\
                self.dn_post_process(predictions_class, predictions_score, out_boxes,mask_dict,predictions_mask)

            predictions_class,  predictions_score, predictions_mask=list(predictions_class), list(predictions_score), list(predictions_mask)
        elif self.training:  # this is to insure self.label_enc participate in the model
            predictions_class[-1] += 0.0*self.label_enc[task].weight.sum()
        if mask_dict is not None:
            # track_embed = self.track_embed(cross_track_embed[:, mask_dict['pad_size']:, :]) if self.cross_track_layer else self.track_embed(hs[-1][:, mask_dict['pad_size']:, :])
            # track_embed = self.track_embed(hs[-1][:, mask_dict['pad_size']:, :])  #if task in ['vis', 'ovis', 'ytvis19' ,'ytvis21', 'uvo_video', 'rvos','burst',  'coco_clip', 'obj365_clip'] else None
            track_embed =  hs[-1][:, mask_dict['pad_size']:, :] # if task in ['vis', 'ovis', 'ytvis19' ,'ytvis21', 'uvo_video', 'rvos','burst',  'coco_clip', 'obj365_clip'] else None
        else:
            # track_embed = self.track_embed(cross_track_embed) if self.cross_track_layer else self.track_embed(self.track_embed(hs[-1]) )
            # track_embed = self.track_embed(hs[-1])  #if task in ['vis', 'ovis', 'ytvis19' ,'ytvis21', 'uvo_video', 'rvos', 'burst', 'coco_clip', 'obj365_clip' ] else None
            track_embed =  hs[-1] # if task in ['vis', 'ovis', 'ytvis19' ,'ytvis21', 'uvo_video', 'rvos', 'burst', 'coco_clip', 'obj365_clip' ] else None

        out = {
            'pred_federat':predictions_federate[-1],
            'pred_logits': predictions_class[-1],
            'pred_scores': predictions_score[-1],
            'pred_masks': predictions_mask[-1],
            'pred_boxes':out_boxes[-1],
            'pred_track_embed': track_embed,
            'visual_P': visual_P,
            'aux_outputs': self._set_aux_loss(
                predictions_class if self.mask_classification else None, predictions_score, predictions_mask, out_boxes, predictions_federate, visual_P
            )
        }
        if self.two_stage:
            out['interm_outputs'] = interm_outputs
        return out, mask_dict

    def forward_prediction_heads(self, output, mask_features, task, extra, mask_dict, pred_mask=True, visual_P=False, custom_part_categories_idx=None):
        decoder_output = self.decoder_norm(output)
        decoder_output = decoder_output.transpose(0, 1)
        # outputs_class = self.class_embed(decoder_output)

        conf_score = self.confidence_score(decoder_output) # if visual_P else None
        # import pdb;pdb.set_trace()

        class_embed = decoder_output @ self.category_embed  # [bz,num_q,projectdim]
        if task in ['grounding', 'rvos']:
            outputs_class =  torch.einsum("bqc,bc->bq", class_embed, extra['grounding_class']).unsqueeze(-1) #[bz,numq,1]
        else:
            if self.unify_object_part and task not in self.object_level_datasets:
                if 'custom' in task:
                    assert custom_part_categories_idx != None, "Customized Task must input part categories indices!"
                    part_category_idx = custom_part_categories_idx
                else:
                    part_category_idx = self.part_datasets_part_category_idx[task]
                # part_class_embed = extra['class_embeddings'][part_category_idx, :] + self.part_category_embed.weight
                part_class_embed = extra['class_embeddings'][part_category_idx, :]
                outputs_class = torch.einsum("bqc,nc->bqn", class_embed, part_class_embed)  #[bz,n,80]
            else:
                outputs_class = torch.einsum("bqc,nc->bqn", class_embed, extra['class_embeddings'])  #[bz,n,80]
        # elif task == 'sa1b':
        #     outputs_class = conf_score
        # else:
        #     print('error')
        # # import pdb;pdb.set_trace()
        outputs_mask = None
        if pred_mask:
            mask_embed = self.mask_embed(decoder_output)
            outputs_mask = torch.einsum("bqc,bchw->bqhw", mask_embed, mask_features)

        # if task == 'burst':
        #     if mask_dict:
        #         class_embed = class_embed[:,mask_dict['pad_size']:]
        #     bz=len(class_embed)
        #     assert bz == len(extra['federated']['neg'])
        #     pred_federat = {'neg':[],'ex':[], 'noex':[]}
        #     for embed,neg, ex, noex in zip(class_embed,extra['federated']['neg'],extra['federated']['ex'],extra['federated']['noex']):
        #         pred_federat['neg'].append(embed@neg.permute(1,0))
        #         pred_federat['ex'].append(embed@ex.permute(1,0))
        #         pred_federat['noex'].append(embed@noex.permute(1,0))
        #     return conf_score, outputs_class, outputs_mask, pred_federat


        return conf_score, outputs_class, outputs_mask, None

    @torch.jit.unused
    def _set_aux_loss(self, outputs_class, outputs_score, outputs_seg_masks, out_boxes, predictions_federate=None, visual_P=False):
        # this is a workaround to make torchscript happy, as torchscript
        # doesn't support dictionary with non-homogeneous values, such
        # as a dict having both a Tensor and a list.
        # if self.mask_classification:
        if predictions_federate is None:
            return [
                {"pred_logits": a, "pred_scores": b, "pred_masks": c, "pred_boxes":d, 'visual_P': visual_P}
                for a, b, c, d in zip(outputs_class[:-1], outputs_score[:-1], outputs_seg_masks[:-1], out_boxes[:-1])
            ]
        else:
            return [
                {"pred_logits": a, "pred_scores": b, "pred_masks": c, "pred_boxes":d, 'pred_federat':e,'visual_P': visual_P}
                for a, b, c, d, e in zip(outputs_class[:-1], outputs_score[:-1], outputs_seg_masks[:-1], out_boxes[:-1], predictions_federate[:-1])
            ]