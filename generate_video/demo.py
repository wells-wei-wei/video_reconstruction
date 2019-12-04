import torch
import numpy as np
from tqdm import tqdm
import os
import glob

from models.imitator import Imitator
from options.test_options import TestOptions
from utils.util import mkdir
import pickle
from utils.video import make_video

from run_imitator import adaptive_personalize

pkl_path="../demo/transfer_data/human_body_info.pkl"
src_img_path=""
img_paths = sorted(glob.glob('../demo/transfer_data' + '/*.jpg'))
for img_path in img_paths:
    src_img_path=img_path

def load_smpl(pkl_path):
    with open(pkl_path, 'rb') as f:
        result = pickle.load(f)

    anim_len = result['anim_len']
    pose_array = result['smpl_array'].reshape(anim_len, -1)
    cam_array = result['cam_array']
    shape_array = np.ones((anim_len, 10))
    smpl_array = np.concatenate((cam_array, pose_array, shape_array), axis=1)

    return smpl_array

def generate_actor_result(test_opt, src_img_path):
    imitator = Imitator(test_opt)
    src_img_name = os.path.split(src_img_path)[-1][:-4]
    test_opt.src_path = src_img_path

    if test_opt.post_tune:
        adaptive_personalize(test_opt, imitator, visualizer=None)
    else:
        imitator.personalize(test_opt.src_path, visualizer=None)

    if test_opt.output_dir:
        pred_output_dir = os.path.join(test_opt.output_dir, 'frames')
        if os.path.exists(pred_output_dir):
            os.system("rm -r %s" % pred_output_dir)
        mkdir(pred_output_dir)
    else:
        pred_output_dir = None

    print(pred_output_dir)
    tgt_smpls = load_smpl(pkl_path)

    imitator.inference_by_smpls(tgt_smpls, cam_strategy='smooth', output_dir=pred_output_dir, visualizer=None)

    save_dir = test_opt.output_dir
    mkdir(save_dir)

    output_mp4_path = os.path.join(save_dir, 'demo.mp4')
    img_path_list = sorted(glob.glob('%s/*.jpg' % pred_output_dir))
    make_video(output_mp4_path, img_path_list, save_frames_dir=None, fps=30)
    

def clean(output_dir):
    for item in ['imgs', 'pairs', 'mixamo_preds', 'pairs_meta.pkl']:
        filepath = os.path.join(output_dir, item)
        if os.path.exists(filepath):
            os.system("rm -r %s" % filepath)

def main():
    # meta imitator
    test_opt = TestOptions().parse()
    test_opt.bg_ks = 25
    test_opt.front_warp = False
    test_opt.post_tune = True

    test_opt.output_dir = mkdir('../demo/results/')
    
    generate_actor_result(test_opt, src_img_path)

    # clean other files
    clean(test_opt.output_dir)

    print('Completed! All demo videos are save in {}'.format(test_opt.output_dir))


if __name__ == "__main__":
    main()
