import os
import argparse
import sys
import torch
import numpy as np
import cv2
import time
import random
import datetime
import math

import torch
import torch.nn as nn
from torch.utils import data
import torch.multiprocessing as mp
import subprocess as sp
import pickle
import pdb

from models import vinet
from utils.dataset import dataset
from models.utils import to_var


parser = argparse.ArgumentParser(description="Dee-Flow-Guided")
parser.add_argument("-b", type=int, default=1)
parser.add_argument("-e", type=int, default=0)
parser.add_argument("-n", type=str, default='youtube-vos') 
parser.add_argument("-m", type=str, default='fixed') 
args = parser.parse_args()


DATA_NAME = args.n
MASK_TYPE = args.m
DEFAULT_FPS = 15
# set random seed 
seed = 2020
torch.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
np.random.seed(seed)
random.seed(seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = True

class Object():
  pass
opt = Object()
opt.size = (432, 240)
opt.search_range = 4 # fixed as 4: search range for flow subnetworks
opt.pretrain_path = 'weights/save_agg_rec.pth'
opt.result_path = 'results/{}_{}'.format(DATA_NAME, MASK_TYPE)

def to_img(x):
  tmp = (x[0,:,0,:,:].cpu().data.numpy().transpose((1,2,0))+1)/2
  tmp = np.clip(tmp,0,1)*255.
  return tmp.astype(np.uint8)


# initialize model 
opt.model = 'vinet_final'
opt.batch_norm = False
opt.no_train = True
opt.test = True
opt.t_stride = 3
opt.loss_on_raw = False
opt.prev_warp = True


# set parameter to gpu or cpu
def set_device(args):
  if torch.cuda.is_available():
    if isinstance(args, list):
      return (item.cuda() for item in args)
    else:
      return args.cuda()
  return args

def get_clear_state_dict(old_state_dict):
  from collections import OrderedDict
  new_state_dict = OrderedDict()
  for k,v in old_state_dict.items():
    name = k 
    if k.startswith('module.'):
      name = k[7:]
    new_state_dict[name] = v
  return new_state_dict


def main_worker(gpu, ngpus_per_node):
  if ngpus_per_node > 1:
    torch.cuda.set_device(int(gpu))
  model = vinet.VINet_final(opt=opt)
  model = set_device(model)

  loaded, empty = 0,0
  pretrain = torch.load(opt.pretrain_path, map_location = lambda storage, loc: set_device(storage))
  pretrain['state_dict'] = get_clear_state_dict(pretrain['state_dict'])
  # load pretrained model parameters
  child_dict = model.state_dict()
  parent_list = pretrain['state_dict'].keys()
  parent_dict = {}
  for chi,_ in child_dict.items():
    if chi in parent_list:
      parent_dict[chi] = pretrain['state_dict'][chi]
      loaded += 1
    else:
      empty += 1
  print('GPU-{}: Loaded {} pretrained model from {}, remain {} empty'.format(
    gpu, loaded, opt.pretrain_path, empty))
  child_dict.update(parent_dict)
  model.load_state_dict(child_dict)
  model.eval()


  # dataset 
  DTset = dataset(DATA_NAME, MASK_TYPE, size=opt.size)
  step = math.ceil(len(DTset) / ngpus_per_node)
  DTset.set_subset(gpu*step, min(gpu*step+step, len(DTset)))
  Trainloader = data.DataLoader(DTset, batch_size=1, shuffle=False, num_workers=1)

  ts = opt.t_stride
  pre = 20

  with torch.no_grad():
    for seq, (inputs, masks, info) in enumerate(Trainloader):
      idx = torch.LongTensor([i for i in range(pre-1,-1,-1)])
      seq_name = info['name'][0]
      print('[{}] {}/{}: {} for {} frames ...'.format(datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"), 
        seq, len(Trainloader), seq_name, inputs.size()[2]))
      pre_inputs = inputs[:,:,:pre].index_select(2,idx)
      pre_masks = masks[:,:,:pre].index_select(2,idx)
      inputs = torch.cat((pre_inputs, inputs),2)
      masks = torch.cat((pre_masks, masks),2)

      bs = inputs.size(0)
      num_frames = inputs.size(2)
      save_path = os.path.join(opt.result_path, seq_name)
      
      # prepare datasets
      inputs = 2.*inputs - 1
      inverse_masks = 1-masks
      masked_inputs = inputs.clone()*inverse_masks
      masks = to_var(masks)
      masked_inputs = to_var(masked_inputs)
      inputs = to_var(inputs)
      comp_frames = []
      pred_frames = []
      orig_frames = []
      mask_frames = []

      lstm_state = None
      for t in range(num_frames):
        masked_inputs_ = []
        masks_ = []        

        if t < 2*ts:
          masked_inputs_.append(masked_inputs[0,:,abs(t-2*ts)])
          masked_inputs_.append(masked_inputs[0,:,abs(t-1*ts)])
          masked_inputs_.append(masked_inputs[0,:,t])
          masked_inputs_.append(masked_inputs[0,:,t+1*ts])
          masked_inputs_.append(masked_inputs[0,:,t+2*ts])
          masks_.append(masks[0,:,abs(t-2*ts)])
          masks_.append(masks[0,:,abs(t-1*ts)])
          masks_.append(masks[0,:,t])
          masks_.append(masks[0,:,t+1*ts])
          masks_.append(masks[0,:,t+2*ts])
        elif t > num_frames-2*ts-1:
          masked_inputs_.append(masked_inputs[0,:,t-2*ts])
          masked_inputs_.append(masked_inputs[0,:,t-1*ts])
          masked_inputs_.append(masked_inputs[0,:,t])
          masked_inputs_.append(masked_inputs[0,:,-1 -abs(num_frames-1-t - 1*ts)])
          masked_inputs_.append(masked_inputs[0,:,-1 -abs(num_frames-1-t - 2*ts)])
          masks_.append(masks[0,:,t-2*ts])
          masks_.append(masks[0,:,t-1*ts])
          masks_.append(masks[0,:,t])
          masks_.append(masks[0,:,-1 -abs(num_frames-1-t - 1*ts)])
          masks_.append(masks[0,:,-1 -abs(num_frames-1-t - 2*ts)])   
        else:
          masked_inputs_.append(masked_inputs[0,:,t-2*ts])
          masked_inputs_.append(masked_inputs[0,:,t-1*ts])
          masked_inputs_.append(masked_inputs[0,:,t])
          masked_inputs_.append(masked_inputs[0,:,t+1*ts])
          masked_inputs_.append(masked_inputs[0,:,t+2*ts])
          masks_.append(masks[0,:,t-2*ts])
          masks_.append(masks[0,:,t-1*ts])
          masks_.append(masks[0,:,t])
          masks_.append(masks[0,:,t+1*ts])
          masks_.append(masks[0,:,t+2*ts])            

        masked_inputs_ = torch.stack(masked_inputs_).permute(1,0,2,3).unsqueeze(0)
        masks_ = torch.stack(masks_).permute(1,0,2,3).unsqueeze(0)

        start = time.time()
        if t==0:
          prev_mask = masks_[:,:,2]
          prev_ones = to_var(torch.ones(prev_mask.size()))
          prev_feed = torch.cat([masked_inputs_[:,:,2,:,:], prev_ones, prev_ones*prev_mask], dim=1)
        else:
          prev_mask = to_var(torch.zeros(masks_[:,:,2].size()))
          prev_ones = to_var(torch.ones(prev_mask.size()))
          prev_feed = torch.cat([outputs.detach().squeeze(2), prev_ones, prev_ones*prev_mask], dim=1)

        masked_inputs_ = masked_inputs_.cuda()
        masks_ = masks_.cuda()
        prev_feed = prev_feed.cuda()

        pred, outputs, _, _, _, _ = model(masked_inputs_, masks_, lstm_state, prev_feed, t)
        end = time.time() - start
        if t >= pre:
          pred_frames.append(to_img(pred))
          comp_frames.append(to_img(outputs))
          mask_frames.append(to_img(masked_inputs[0:1,:,t:t+1]))
          orig_frames.append(to_img(inputs[0:1,:,t:t+1]))

      # write frames into videos
      os.makedirs(save_path, exist_ok=True)
      comp_writer = cv2.VideoWriter(os.path.join(save_path, 'comp.avi'),
        cv2.VideoWriter_fourcc(*"MJPG"), DEFAULT_FPS, opt.size)
      pred_writer = cv2.VideoWriter(os.path.join(save_path, 'pred.avi'),
        cv2.VideoWriter_fourcc(*"MJPG"), DEFAULT_FPS, opt.size)
      mask_writer = cv2.VideoWriter(os.path.join(save_path, 'mask.avi'),
        cv2.VideoWriter_fourcc(*"MJPG"), DEFAULT_FPS, opt.size)
      orig_writer = cv2.VideoWriter(os.path.join(save_path, 'orig.avi'),
        cv2.VideoWriter_fourcc(*"MJPG"), DEFAULT_FPS, opt.size)
      for f in range(len(comp_frames)):
        comp_writer.write(comp_frames[f])
        pred_writer.write(pred_frames[f])
        mask_writer.write(mask_frames[f])
        orig_writer.write(orig_frames[f])
      comp_writer.release()
      pred_writer.release()
      mask_writer.release()
      orig_writer.release()



if __name__ == '__main__':
  ngpus_per_node = torch.cuda.device_count()
  print('Using {} GPUs for testing {}_{}... '.format(ngpus_per_node, DATA_NAME, MASK_TYPE))
  processes = []
  mp.set_start_method('spawn', force=True)
  for rank in range(ngpus_per_node):
    p = mp.Process(target=main_worker, args=(rank, ngpus_per_node))
    p.start()
    processes.append(p)
  for p in processes:
    p.join()
  print('Finished testing for {}_{}'.format(DATA_NAME, MASK_TYPE))
    
