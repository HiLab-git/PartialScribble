# -*- coding: utf-8 -*-
'''`
Code for the following paper:
  https://www.sciencedirect.com/science/article/abs/pii/S1361841524001993?dgcid
    @article{han2024dmsps,
    title={DMSPS: Dynamically mixed soft pseudo-label supervision for scribble-supervised medical image segmentation},
    author={Han, Meng and Luo, Xiangde and Xie, Xiangjiang and Liao, Wenjun and Zhang, Shichuan and Song, Tao and Wang, Guotai and Zhang, Shaoting},
    journal={Medical Image Analysis},
    pages={103274},
    year={2024},
    publisher={Elsevier}
}

'''

from __future__ import print_function, division
import argparse
import logging
import os
import sys
from networks import mTDNetIdpSk_3D
from pymic.util.parse_config import *
from pymic.net_run.weak_sup import  WSLPSSEG



def main():
    if(len(sys.argv) < 2):
        print('Number of arguments should be at least 3. e.g.')
        print('   python run.py train config.cfg')
        exit()
    parser = argparse.ArgumentParser()
    parser.add_argument("stage", type = str, help="stage of train or test")
    parser.add_argument("cfg", type = str, help="configuration file")
    
    args = parser.parse_args()
    if(not os.path.isfile(args.cfg)):
        raise ValueError("The config file does not exist: " + args.cfg)
    config   = parse_config(args)
    config   = synchronize_config(config)

    log_dir  = config['training']['ckpt_dir']
    if(not os.path.exists(log_dir)):
        os.makedirs(log_dir)
    if sys.version.startswith("3.9"):
        logging.basicConfig(filename=log_dir+"/log_{0:}.txt".format(args.stage), level=logging.INFO,
                            format='%(message)s', force=True) # for python 3.9
    else:
        logging.basicConfig(filename=log_dir+"/log_{0:}.txt".format(args.stage), level=logging.INFO,
                            format='%(message)s') # for python 3.6
    logging.getLogger().addHandler(logging.StreamHandler(sys.stdout))
    logging_config(config)

    agent  = WSLPSSEG(config, args.stage)
    net_dict  = {"mTDNetIdpSk_3D": mTDNetIdpSk_3D}
    agent.set_net_dict(net_dict)
    agent.run()


if __name__ == "__main__":
    main()
