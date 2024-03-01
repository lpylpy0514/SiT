import os
import sys

prj_path = os.path.join(os.path.dirname(__file__), '..')
if prj_path not in sys.path:
    sys.path.append(prj_path)

import argparse
import torch
from lib.utils.misc import NestedTensor
from thop import profile
from thop.utils import clever_format
import time
import importlib
import lib.models.ostrack.utils as utils

def parse_args():
    """
    args for training.
    """
    parser = argparse.ArgumentParser(description='Parse args for training')
    # for train
    parser.add_argument('--script', type=str, default='ostrack', choices=['ostrack'],
                        help='training script name')
    parser.add_argument('--config', type=str, default=['s0_s0_resolution_16_4', 's0_s0_resolution', 's1_s1_resolution', 'L_L_resolution',], help='yaml configure file name')
    args = parser.parse_args()

    return args


def evaluate_vit(model, template, search, dim=0):
    # dim = 192 # 192 240 384
    '''Speed Test'''
    macs, params = profile(model.backbone, inputs=(template.unsqueeze(0)), custom_ops=None, verbose=False)
    macs, params = clever_format([macs, params], "%.3f")
    print('backbone macs is ', macs)
    print('backbone params is ', params)

    fusion = torch.randn((1,1,196,dim)).to(template.device)
    macs, params = profile(model.feature_mixer, inputs=(fusion), custom_ops=None, verbose=False)
    macs, params = clever_format([macs, params], "%.3f")
    print('fusion macs is ', macs)
    print('fusion params is ', params)

    fusion = torch.randn((1, 1 ,dim,14, 14)).to(template.device)
    macs, params = profile(model.box_head, inputs=(fusion), custom_ops=None, verbose=False)
    macs, params = clever_format([macs, params], "%.3f")
    print('head macs is ', macs)
    print('head params is ', params)

    template = model.backbone(template)
    macs1, params1 = profile(model, inputs=(template, search, True),
                             custom_ops=None, verbose=False)
    macs1, params1 = clever_format([macs1, params1], "%.3f")
    print('overall macs is ', macs1)
    print('overall params is ', params1)

    T_w = 500
    T_t = 1000
    print("testing speed ...")
    torch.cuda.synchronize()
    with torch.no_grad():
        # overall
        for i in range(T_w):
            _ = model(template, search, True)
            # _ = model.backbone(search)
        start = time.time()
        for i in range(T_t):
            _ = model(template, search, True)
            # _ = model.backbone(search)
        # torch.cuda.synchronize()
        end = time.time()
        avg_lat = (end - start) / T_t
        print("The average overall latency is %.2f ms" % (avg_lat * 1000))
        print("FPS is %.2f fps" % (1. / avg_lat))
        # for i in range(T_w):
        #     _ = model(template, search)
        # start = time.time()
        # for i in range(T_t):
        #     _ = model(template, search)
        # end = time.time()
        # avg_lat = (end - start) / T_t
        # print("The average backbone latency is %.2f ms" % (avg_lat * 1000))


def evaluate_vit_separate(model, template, search):
    '''Speed Test'''
    T_w = 50
    T_t = 500
    print("testing speed ...")
    z = model.forward_backbone(template, image_type='template')
    x = model.forward_backbone(search, image_type='search')
    with torch.no_grad():
        # overall
        for i in range(T_w):
            _ = model.forward_backbone(search, image_type='search')
            _ = model.forward_cat(z, x)
        start = time.time()
        for i in range(T_t):
            _ = model.forward_backbone(search, image_type='search')
            _ = model.forward_cat(z, x)
        end = time.time()
        avg_lat = (end - start) / T_t
        print("The average overall latency is %.2f ms" % (avg_lat * 1000))


def get_data(bs, sz):
    img_patch = torch.randn(bs, 3, sz, sz)
    att_mask = torch.rand(bs, sz, sz) > 0.5
    return NestedTensor(img_patch, att_mask)


if __name__ == "__main__":
    device = "cuda:0"
    # device = 'cpu'
    dims = [192, 240, 384]
    # torch.set_num_threads(1)
    # torch.cuda.set_device(device)

    # Compute the Flops and Params of our STARK-S model
    args = parse_args()
    for i, config in enumerate(args.config):
        dim = dims[i]
        print(config)
        '''update cfg'''
        yaml_fname = './experiments/%s/%s.yaml' % (args.script, config)
        config_module = importlib.import_module('lib.config.%s.config' % args.script)
        cfg = config_module.cfg
        config_module.update_config_from_file(yaml_fname)
        '''set some values'''
        bs = 1
        z_sz = cfg.TEST.TEMPLATE_SIZE
        x_sz = cfg.TEST.SEARCH_SIZE

        if args.script == "ostrack":
            model_module = importlib.import_module('lib.models')
            model_constructor = model_module.build_ostrack
            model = model_constructor(cfg, training=False)


            # merge conv+bn to one operator
            utils.replace_batchnorm(model.backbone)
            # merge layernorm
            # utils.replace_layernorm(model.backbone)

            # get the template and search
            template = torch.randn(bs, 3, z_sz, z_sz)
            search = torch.randn(bs, 3, x_sz, x_sz)
            # transfer to device
            model = model.to(device)
            template = template.to(device)
            search = search.to(device)

            merge_layer = cfg.MODEL.BACKBONE.MERGE_LAYER
            print(merge_layer)
            if merge_layer <= 0:
                evaluate_vit(model, template, search, dim)
            else:
                evaluate_vit_separate(model, template, search)

        else:
            raise NotImplementedError
