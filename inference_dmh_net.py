import argparse
import glob
import json
import os
import shutil
import tempfile
import warnings

import numpy as np
import torch
from PIL import Image

from config import cfg, cfg_from_yaml_file
from model import DMHNet
from misc.utils import pipeload
from perspective_dataset import PerspectiveDataset
from postprocess.postprocess2 import postProcess

CAMERA_H = 1.6


def parse_args():
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--cfg', required=True, help='DMH-Net config yaml (paired with ckpt)')
    parser.add_argument('--pth', required=True, help='checkpoint file')
    parser.add_argument('--img', required=True,
                        help='panorama path or glob (equirectangular, will be resized to 1024x512)')
    parser.add_argument('--output_dir', default='results', help='where to save json results')
    parser.add_argument('--no_cuda', action='store_true', help='disable cuda')
    parser.add_argument('--no_rnn', action='store_true')
    return parser.parse_args()


def resolve_path(p, base):
    if os.path.isabs(p):
        return p
    cand = os.path.join(base, p)
    return cand if os.path.exists(cand) else os.path.abspath(p)


def main():
    np.random.seed(594277)
    torch.manual_seed(594277)
    args = parse_args()
    dmh_root = os.path.dirname(os.path.abspath(__file__))

    cfg_from_yaml_file(resolve_path(args.cfg, dmh_root), cfg)
    cfg.POST_PROCESS.METHOD = 'pred_only'
    cfg.DATA.PREFIX = None

    device = torch.device('cpu' if args.no_cuda or not torch.cuda.is_available() else 'cuda')

    tmp_dir = tempfile.mkdtemp(prefix='dmh_infer_')
    img_out_dir = os.path.join(tmp_dir, 'test', 'img')
    os.makedirs(img_out_dir)
    try:
        paths = sorted(glob.glob(args.img))
        assert len(paths) > 0, 'no images found: %s' % args.img
        for p in paths:
            name = os.path.splitext(os.path.basename(p))[0] + '.png'
            img = Image.open(p).convert('RGB')
            if img.size != (1024, 512):
                img = img.resize((1024, 512), Image.LANCZOS)
            img.save(os.path.join(img_out_dir, name))

        cfg.DATA.ROOT_DIR = tmp_dir
        dataset = PerspectiveDataset(cfg, 'test')
        assert len(dataset) > 0, ('no samples in dataset: check images and '
                                  'PREFIX/USE_CORNER filters in %s' % args.cfg)

        net = DMHNet(cfg, cfg.MODEL.get('BACKBONE', {}).get('NAME', 'drn38'), not args.no_rnn).to(device)
        net = torch.nn.DataParallel(net)
        state_dict = pipeload(resolve_path(args.pth, dmh_root), map_location='cpu')['state_dict']
        net.load_state_dict(state_dict, strict=True)
        net.eval()

        os.makedirs(args.output_dir, exist_ok=True)
        for idx in range(len(dataset)):
            name = dataset.img_fnames[idx]
            input = PerspectiveDataset.collate([dataset[idx]])
            with torch.no_grad():
                for k in input:
                    if isinstance(input[k], torch.Tensor):
                        input[k] = input[k].to(device)
                _, results_dict = net(input)
                (_, _, _), (_, pred_lwh, pred_cors), _ = postProcess(cfg, input, results_dict, 0)

            assert pred_cors is not None,  'pred_only postprocess produced no corners'
            e_hw = input['e_img'].shape[-1:-3:-1]  # (W, H)
            uv = pred_cors.cpu().numpy() / np.array(e_hw, np.float32)
            uv = [[float(x), float(y)] for x, y in uv]
            lwh = [float(v) for v in pred_lwh.cpu().numpy().reshape(-1)]

            result = {
                'z0': lwh[5] * 100.0,
                'z1': -CAMERA_H * 100.0,
                'uv': uv,
                'lwh': lwh,
            }
            json_path = os.path.join(args.output_dir,
                                     os.path.splitext(os.path.basename(name))[0] + '.json')
            with open(json_path, 'w') as f:
                json.dump(result, f, indent=2)
            print('RESULT_JSON: %s' % os.path.abspath(json_path))
    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)


if __name__ == '__main__':
    warnings.filterwarnings('ignore')
    main()