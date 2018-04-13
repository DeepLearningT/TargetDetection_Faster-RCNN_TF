# -*- coding:utf-8 -*-
#!/usr/bin/env python

# --------------------------------------------------------
# Fast R-CNN
# Copyright (c) 2015 Microsoft
# Licensed under The MIT License [see LICENSE for details]
# Written by Ross Girshick
# --------------------------------------------------------

"""Train a Fast R-CNN network on a region of interest database."""

import _init_paths
from fast_rcnn.train import get_training_roidb, train_net
from fast_rcnn.config import cfg,cfg_from_file, cfg_from_list, get_output_dir
from datasets.factory import get_imdb
from networks.factory import get_network
import argparse
import pprint
import numpy as np
import sys
import pdb

def parse_args():
    """
    Parse input arguments
    """
    parser = argparse.ArgumentParser(description='Train a Fast R-CNN network')
    parser.add_argument('--device', dest='device', help='device to use',
                        default='cpu', type=str)
    parser.add_argument('--device_id', dest='device_id', help='device id to use',
                        default=0, type=int)
    parser.add_argument('--solver', dest='solver',
                        help='solver prototxt',
                        default=None, type=str)
    parser.add_argument('--iters', dest='max_iters',
                        help='number of iterations to train',
                        default=70000, type=int)
    parser.add_argument('--weights', dest='pretrained_model',
                        help='initialize with pretrained model weights',
                        default=None, type=str)
    parser.add_argument('--cfg', dest='cfg_file',
                        help='optional config file',
                        default=None, type=str)
    parser.add_argument('--imdb', dest='imdb_name',
                        help='dataset to train on',
                        default='kitti_train', type=str)
    parser.add_argument('--rand', dest='randomize',
                        help='randomize (do not use a fixed seed)',
                        action='store_true')
    parser.add_argument('--network', dest='network_name',
                        help='name of the network',
                        default=None, type=str)
    parser.add_argument('--set', dest='set_cfgs',
                        help='set config keys', default=None,
                        nargs=argparse.REMAINDER)

    if len(sys.argv) == 1:
        parser.print_help()
        sys.exit(1)

    args = parser.parse_args()
    return args

"""
##############################
# run configuration 参数全路径
LD_LIBRARY_PATH=/usr/local/cuda-9.0/lib64




# 脚本
cd /data/home/deeplearn/tensorflow-workspace/TargetDetection_Faster-RCNN_TF
export PYTHON_ROOT=/opt/Python_gpu
export PATH="$PATH:/usr/local/cuda-9.0/bin" 
export LD_LIBRARY_PATH="/usr/local/cuda-9.0/lib64"
export CUDA_VISIBLE_DEVICES="0,1,2,3"
${PYTHON_ROOT}/bin/python tools/train_net.py \
--device 'gpu' \
--device_id 1 \
--iters 70000 \
--weights data/pretrain_model/VGG_imagenet.npy \
--cfg experiments/cfgs/faster_rcnn_end2end.yml \
--imdb voc_2007_trainval \
--network VGGnet_train 

"""


if __name__ == '__main__':
    args = parse_args()

    print('Called with args:')
    print(args)

    if args.cfg_file is not None:
        cfg_from_file(args.cfg_file)
    if args.set_cfgs is not None:
        cfg_from_list(args.set_cfgs)

    print('Using config:')
    pprint.pprint(cfg)

    if not args.randomize:
        # fix the random seeds (numpy and caffe) for reproducibility
        np.random.seed(cfg.RNG_SEED)
    # 加图片数据名称 和 处理标注框的handler类
    imdb = get_imdb(args.imdb_name)
    print 'Loaded dataset `{:s}` for training'.format(imdb.name)
    # 处理加载标注框数据roidb，roidb-然后图像进行翻转，roidb-准备训练数据
    roidb = get_training_roidb(imdb)

    output_dir = get_output_dir(imdb, None)
    print 'Output will be saved to `{:s}`'.format(output_dir)

    device_name = '/{}:{:d}'.format(args.device,args.device_id)
    print device_name

    network = get_network(args.network_name)
    print 'Use network `{:s}` in training'.format(args.network_name)

    train_net(network, imdb, roidb, output_dir,
              pretrained_model=args.pretrained_model,
              max_iters=args.max_iters)
