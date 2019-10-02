from __future__ import division
import torch
from torch.utils import data

# general libs
import cv2
import io
import matplotlib.pyplot as plt
from PIL import Image
import numpy as np
import math
import time
import tqdm
import os
import random
import argparse
import glob
import json

from scipy import ndimage, signal
import pdb
import zipfile
from utils.utils import get_video_masks_by_moving_random_stroke


class ZipReader(object):
  file_dict = dict()
  def __init__(self):
    super(ZipReader, self).__init__()

  @staticmethod
  def build_file_dict(path):
    file_dict = ZipReader.file_dict
    if path in file_dict:
      return file_dict[path]
    else:
      file_handle = zipfile.ZipFile(path, 'r')
      file_dict[path] = file_handle
      return file_dict[path]

  @staticmethod
  def imread(path, image_name):
    zfile = ZipReader.build_file_dict(path)
    data = zfile.read(image_name)
    im = Image.open(io.BytesIO(data))
    return im

class dataset(data.Dataset):
  def __init__(self, data_name, mask_type, size=(424, 240)):
    with open(os.path.join('../flist', data_name, 'test.json'), 'r') as f:
      self.video_dict = json.load(f)
    self.videos = list(self.video_dict.keys())
    with open(os.path.join('../flist', data_name, 'mask.json'), 'r') as f:
      self.mask_dict = json.load(f)
    self.masks = list(self.mask_dict.keys())
    self.size = self.w, self.h = size
    self.mask_type = mask_type
    self.data_name = data_name

  def __len__(self):
    return len(self.videos)
  
  def set_subset(self, start, end):
    self.videos = self.videos[start:end]


  def __getitem__(self, index):
    info = {}
    video = self.videos[index]
    info['name'] = video
    frame_names = self.video_dict[video]
    
    images = []
    masks = []
    
    for f, name in enumerate(frame_names):
      image_ = ZipReader.imread('../datazip/{}/JPEGImages/{}.zip'.format(self.data_name, video), name)
      image_ = cv2.resize(cv2.cvtColor(np.array(image_), cv2.COLOR_RGB2BGR), self.size, cv2.INTER_CUBIC)
      image_ = np.float32(image_)/255.0
      images.append(torch.from_numpy(image_))

      if self.mask_type != 'random_obj':
        mask_ = self._get_masks(index, video, f)
        mask_ = cv2.resize(mask_, self.size, cv2.INTER_NEAREST)
        masks.append(torch.from_numpy(mask_))
    if self.mask_type == 'random_obj':
      masks = [torch.from_numpy(np.array(m).astype(np.uint8)) for m in get_video_masks_by_moving_random_stroke(len(frame_names), imageWidth=self.w, imageHeight=self.h)]

    masks = torch.stack(masks)
    masks = ( masks == 1 ).type(torch.FloatTensor).unsqueeze(0)
    images = torch.stack(images).permute(3,0,1,2)
    return images, masks, info


  def _get_masks(self, index, video, i):
    h, w = self.size
    if self.mask_type == 'fixed':
      m = np.zeros(self.size, np.uint8)
      m[h//2-h//8:h//2+h//8, w//2-w//8:w//2+w//8] = 1
      return m
    elif self.mask_type == 'object':
      m_name = self.mask_dict[video][i]
      m = ZipReader.imread('../datazip/{}/Annotations/{}.zip'.format(self.data_name, video), m_name).convert('L')
      m = np.array(m)
      m = np.array(m>0).astype(np.uint8)
      m = cv2.dilate(m, cv2.getStructuringElement(cv2.MORPH_CROSS,(3,3)), iterations=4).astype(np.float32)
      return m
    else:
      raise NotImplementedError(f"Mask type {self.mask_type} not exists")

