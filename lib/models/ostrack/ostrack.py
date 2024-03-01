"""
Basic OSTrack model.
"""
import math
import os
from typing import List

import torch
from torch import nn
from torch.nn.modules.transformer import _get_clones

from lib.models.layers.head import build_box_head
from lib.models.ostrack.vit import vit_base_patch16_224
from lib.models.ostrack.vit_ce import vit_large_patch16_224_ce, vit_base_patch16_224_ce
from lib.utils.box_ops import box_xyxy_to_cxcywh


from lib.models.ostrack.vit import Fearure_Mixer_base
from lib.models.ostrack.efficientformer import efficientformerv2_s0, efficientformerv2_s1, efficientformerv2_s2, efficientformerv2_l
from lib.models.ostrack.efficientformer import efficientformerv2_s0_16_16, efficientformerv2_s0_16_4

import time

class OSTrack(nn.Module):
    """ This is the base class for OSTrack """

    def __init__(self, transformer, feature_mixer, box_head, aux_loss=False, head_type="CORNER"):
        """ Initializes the model.
        Parameters:
            transformer: torch module of the transformer architecture.
            aux_loss: True if auxiliary decoding losses (loss at each decoder layer) are to be used.
        """
        super().__init__()
        self.backbone = transformer
        self.box_head = box_head
        self.feature_mixer = feature_mixer

        self.aux_loss = aux_loss
        self.head_type = head_type
        if head_type == "CORNER" or head_type == "CENTER":
            self.feat_sz_s = int(box_head.feat_sz)
            self.feat_len_s = int(box_head.feat_sz ** 2)

        if self.aux_loss:
            self.box_head = _get_clones(self.box_head, 6)

    def forward(self, template: torch.Tensor,
                search: torch.Tensor,
                test = False
                ):
        # time_start = time.time()
        if not test:
            template_feature = self.backbone(template)
        else:
            template_feature = template
        search_feature =  self.backbone(search)
        # time_1 = time.time()

        # 现在是在通道上拼接，也可以换一种拼接方式
        fusion = torch.concat((template_feature, search_feature), dim=1).flatten(2)
        # fusion = torch.concat((template_feature.flatten(2), search_feature.flatten(2)), dim=2).permute(0, 2, 1)

        # time_2 = time.time()
        fusion = fusion.permute(0,2,1)
        fusion_feature = self.feature_mixer(fusion)
        # time_3 = time.time()
        # Forward head
        feat_last = fusion_feature[:, -196:, :]

        out = self.forward_head(feat_last, None)
        # time_4 = time.time()
        # print(f"time feature is :{(time_1- time_start)/(time_4-time_start)*100:.3f}")
        # print(f"time concat is :{(time_2 - time_1)/(time_4-time_start)*100:.3f}")
        # print(f"time fusion is :{(time_3 - time_2)/(time_4-time_start)*100:.3f}")
        # print(f"time head is :{(time_4 - time_3)/(time_4-time_start)*100:.3f}")
        return out

    def forward_head(self, cat_feature, gt_score_map=None):
        """
        cat_feature: output embeddings of the backbone, it can be (HW1+HW2, B, C) or (HW2, B, C)
        """
        enc_opt = cat_feature[:, -self.feat_len_s:]  # encoder output for the search region (B, HW, C)
        opt = (enc_opt.unsqueeze(-1)).permute((0, 3, 2, 1)).contiguous()
        bs, Nq, C, HW = opt.size()
        opt_feat = opt.view(-1, C, self.feat_sz_s, self.feat_sz_s)

        if self.head_type == "CORNER":
            # run the corner head
            pred_box, _, _ = self.box_head(opt_feat, True)
            outputs_coord = box_xyxy_to_cxcywh(pred_box)
            outputs_coord_new = outputs_coord.view(bs, Nq, 4)
            # score_map = None
            out = {'pred_boxes': outputs_coord_new,
                   # 'score_map': score_map,
                   }
            return out

        elif self.head_type == "CENTER":
            # run the center head
            score_map_ctr, bbox, size_map, offset_map = self.box_head(opt_feat, gt_score_map)
            # outputs_coord = box_xyxy_to_cxcywh(bbox)
            outputs_coord = bbox
            outputs_coord_new = outputs_coord.view(bs, Nq, 4)
            out = {'pred_boxes': outputs_coord_new,
                   'score_map': score_map_ctr,
                   'size_map': size_map,
                   'offset_map': offset_map}
            return out
        else:
            raise NotImplementedError


def build_ostrack(cfg, training=True):
    current_dir = os.path.dirname(os.path.abspath(__file__))  # This is your Project Root
    pretrained_path = os.path.join(current_dir, '../../../pretrained_models')
    if cfg.MODEL.PRETRAIN_FILE and ('OSTrack' not in cfg.MODEL.PRETRAIN_FILE) and training:
        pretrained = os.path.join(pretrained_path, cfg.MODEL.PRETRAIN_FILE)
    else:
        pretrained = ''

    if cfg.MODEL.BACKBONE.TYPE == 'efficientformerv2_s0':
        backbone = efficientformerv2_s0()

    elif cfg.MODEL.BACKBONE.TYPE == 'efficientformerv2_s1':
        backbone = efficientformerv2_s1()

    elif cfg.MODEL.BACKBONE.TYPE == 'efficientformerv2_s2':
        backbone = efficientformerv2_s2()

    elif cfg.MODEL.BACKBONE.TYPE == 'efficientformerv2_l':
        backbone = efficientformerv2_l()

    elif cfg.MODEL.BACKBONE.TYPE == 'efficientformerv2_s0_16_16':
        backbone = efficientformerv2_s0_16_16()
    elif cfg.MODEL.BACKBONE.TYPE == 'efficientformerv2_s0_16_4':
        backbone = efficientformerv2_s0_16_4()
    else:
        raise NotImplementedError

    hidden_dim = cfg.MODEL.EMBED_DIM

    feature_mixer = Fearure_Mixer_base(embed_dim = hidden_dim, depth=cfg.MODEL.FUSION_DEPTH)

    box_head = build_box_head(cfg, hidden_dim)

    model = OSTrack(
        backbone,
        feature_mixer,
        box_head,
        aux_loss=False,
        head_type=cfg.MODEL.HEAD.TYPE,
    )

    if 'eformer' in cfg.MODEL.PRETRAIN_FILE and training:
        #checkpoint_dir = os.path.join("/home/zca/Downloads/code/OSTrack/pretrained_models", cfg.MODEL.PRETRAIN_FILE)
        checkpoint = torch.load(pretrained, map_location="cpu")
        from collections import OrderedDict
        new_checkpoint = OrderedDict()
        for k, v in checkpoint['model'].items():
            new_key = 'backbone.' + k
            new_checkpoint[new_key] = v
        missing_keys, unexpected_keys = model.load_state_dict(new_checkpoint, strict=False)
        print('Load pretrained model from: ' + cfg.MODEL.PRETRAIN_FILE)

    return model
