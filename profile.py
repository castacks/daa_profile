import argparse
import os
import cv2
import yaml
from tqdm import tqdm
import numpy as np
import torch
import random
import time
import sys

base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(base_dir)

from model import SegDetector

device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

def load_config_data(fl: str) -> dict:
    with open(fl) as f:
        cfg: dict = yaml.load(f, Loader=yaml.FullLoader)
    return cfg

def seed_everything(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = False
    torch.backends.cudnn.benchmark = True

def run_inference(cfg, detector):
    fps = 0
    infer_fps = 0
    frame_ctr = 0
    prev_frame = {}
    frame_viz = {}
    padding = cfg['padding']

    img_names = os.listdir(os.path.join(cfg['input_dir'], 'cam_0'+'/images'))
    for i, img_name in tqdm(enumerate(img_names)):
        frame_ctr += 1
        Ttensor = 0
        Tinf = 0
        total_time = 0
        t0 = time.time()

        if i == 0:
            for num in range(cfg['num_cam']):
                img_dir = os.path.join(cfg['input_dir'], 'cam_'+str(num)+'/images')
                img_path = os.path.join(img_dir, img_name)
                img = cv2.imread(img_path, 0)
                frame = cv2.resize(img, (cfg['width'], cfg['height']))
                prev_frame['cam_'+str(num)] = frame

        t1 = time.time()
        X = np.zeros((cfg['num_cam'], 2, cfg['height'], cfg['width'] + padding * 2), dtype=np.uint8)
        for num in range(cfg['num_cam']):
            img_dir = os.path.join(cfg['input_dir'], 'cam_'+str(num)+'/images')
            img_path = os.path.join(img_dir, img_name)
            img = cv2.imread(img_path, 0)
            frame = cv2.resize(img, (cfg['width'], cfg['height']))
            frame_viz['cam_'+str(num)] = frame

            X[num, 0, :, padding:-padding] = prev_frame['cam_'+str(num)]
            X[num, 1, :, padding:-padding] = frame

            prev_frame['cam_'+str(num)] = frame

        X = torch.from_numpy(X).half().cuda() / 255.0
        Ttensor += time.time() - t1

        t2 = time.time()
        with torch.cuda.amp.autocast():
            pred_mask, pred_size, pred_offset, pred_distance, pred_tracking, pred_above_horizon = detector.model(X)

        torch.cuda.synchronize()
        Tinf += time.time() - t2
        
        total_time += time.time() - t0
        del pred_mask, pred_size, pred_offset, pred_distance, pred_tracking, pred_above_horizon
    
        print('Ttensor:', Ttensor)
        print('Tinf:', Tinf)
        print('Total:', total_time)
        fps += 1 / total_time
        infer_fps += 1 / (Tinf)
        avg_fps = fps / frame_ctr
        avg_infer_fps = infer_fps / frame_ctr
        print('Avg fps:', avg_fps)
        print('Avg infer fps:', avg_infer_fps)

def run_evaluation(experiment: str):
    print(f'Running exp {experiment}')
    # load eval config
    cfg = load_config_data(experiment)
    print('Loaded config:')
    print(cfg)

    cfg['trt_resolution'] = [cfg['height'], cfg['width']+2*cfg['padding']]

    detector = SegDetector(cfg=dict(
        use_tensorrt=cfg['use_tensorrt'],
        use_torch2trt=cfg['use_torch2trt'],
        use_dla=cfg['use_dla'],
        input_frames=cfg['input_frames'],
        trt_batch_size=cfg['trt_batch_size'],
        trt_resolution=cfg['trt_resolution'],
        models_dir=cfg['models_dir'],
        full_res_model_chkpt=cfg['full_res_model_chkpt'],
        fp16_mode=cfg['fp16_mode'],
        int8_mode=cfg['int8_mode']
    ))

    # helper fucntion to run the actual eval
    run_inference(cfg, detector)


if __name__ == "__main__":
    seed_everything(seed=49)
    parser = argparse.ArgumentParser()
    parser.add_argument("experiment", type=str, default="")

    args = parser.parse_args()

    run_evaluation(args.experiment)