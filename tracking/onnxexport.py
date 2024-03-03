import os
import sys

prj_path = os.path.join(os.path.dirname(__file__), '..')
if prj_path not in sys.path:
    sys.path.append(prj_path)

import argparse
import torch
from torch import nn
from lib.utils.misc import NestedTensor
from thop import profile
from thop.utils import clever_format
import time
import importlib
import lib.models.ostrack.utils as utils

import onnx
import onnxruntime
import numpy as np

def to_numpy(tensor):
    return tensor.detach().cpu().numpy() if tensor.requires_grad else tensor.cpu().numpy()

def evaluate_vit_onnx(session, ort_inputs):

    T_w = 500
    T_t = 1000
    print("testing speed ...")
    torch.cuda.synchronize()
    with torch.no_grad():
        # overall
        for i in range(T_w):
            ort_outs = ort_session.run(None, ort_inputs)
        start = time.time()
        for i in range(T_t):
            ort_outs = ort_session.run(None, ort_inputs)
        end = time.time()
        avg_lat = (end - start) / T_t
        print("The average overall latency is %.2f ms" % (avg_lat * 1000))
        print("FPS is %.2f fps" % (1. / avg_lat))



def parse_args():
    """
    args for training.
    """
    parser = argparse.ArgumentParser(description='Parse args for training')
    # for train
    parser.add_argument('--script', type=str, default='ostrack', choices=['ostrack'],
                        help='training script name')
    parser.add_argument('--config', type=str, default=['s0_s0_resolution_16_16', 's0_s0_resolution_16_4', 's0_s0_resolution', 's1_s1_resolution', 'L_L_resolution','levit_384'], help='yaml configure file name')

    parser.add_argument('--device', type=str, default="cuda:0", help='device to inference')
    args = parser.parse_args()

    return args

if __name__ == "__main__":
    dims = [192, 192, 192, 240, 384, 768]
    args = parse_args()
    device = args.device
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
            save_dir = './output'
            # checkpoint_name = os.path.join(save_dir,
            #                             "./checkpoints/train/%s/%s/OSTrack_ep%04d.pth.tar"
            #                             % (args.script, config, cfg.TEST.EPOCH))
            # model.load_state_dict(torch.load(checkpoint_name, map_location='cpu')['net'], strict=True)
            model.eval()

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
            
            # export model into onnx
            save_name1 = "Sit_" + config + "_backbone.onnx"
            save_name2 = "Sit_" + config + "_network.onnx"
            torch.onnx.export(model.backbone,  # model being run
                    (template),  # model input (a tuple for multiple inputs)
                    save_name1,  # where to save the model (can be a file or file-like object)
                    export_params=True,  # store the trained parameter weights inside the model file
                    opset_version=11,  # the ONNX version to export the model to
                    do_constant_folding=True,  # whether to execute constant folding for optimization
                    input_names=['template'],  # model's input names
                    output_names=['template_feature'],  # the model's output names
                    # dynamic_axes={'input': {0: 'batch_size'},  # variable length axes
                    #               'output': {0: 'batch_size'}}
                    )
            template = model.backbone(template)
            res = model(template, search, True)
            print(template.shape)
            torch.onnx.export(model,  # model being run
                    (template, search, True),  # model input (a tuple for multiple inputs)
                    save_name2,  # where to save the model (can be a file or file-like object)
                    export_params=True,  # store the trained parameter weights inside the model file
                    opset_version=11,  # the ONNX version to export the model to
                    do_constant_folding=True,  # whether to execute constant folding for optimization
                    input_names=['template', 'search'],  # model's input names
                    output_names=['outputs_coord_new'],  # the model's output names
                    # dynamic_axes={'input': {0: 'batch_size'},  # variable length axes
                    #               'output': {0: 'batch_size'}}
                    )

            if device == "cpu":
                ort_session = onnxruntime.InferenceSession(save_name2,providers=['CPUExecutionProvider'])
            else:
                ort_session = onnxruntime.InferenceSession(save_name2,providers=['CUDAExecutionProvider'])

            ort_inputs = {'search': to_numpy(search).astype(np.float32),
                        'template': to_numpy(template).astype(np.float32)
                        }
            
            evaluate_vit_onnx(ort_session, ort_inputs)


        else:
            raise NotImplementedError
