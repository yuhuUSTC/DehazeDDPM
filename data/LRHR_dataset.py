from io import BytesIO
import lmdb
import torch
from PIL import Image, ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True
from torchvision.transforms import functional as trans_fn
from torch.utils.data import Dataset
import matplotlib.pyplot as plt
import random
import torchvision
import data.util as Util
from torchvision import transforms, utils
import os
import numpy as np
import cv2
from os import path as osp

class LRHRDataset(Dataset):
    def __init__(self, datarootlq, dataroothq, datatype, split='train', data_len=-1, img_sizeH=256, img_sizeW=256):
        self.datatype = datatype
        self.data_len = data_len
        self.split = split
        self.img_sizeH = img_sizeH
        self.img_sizeW = img_sizeW

        if datatype == 'img':
            # self.sr_path = Util.get_paths_from_images(datarootlq)
            # self.hr_path = Util.get_paths_from_images(dataroothq)
            self.paths = self.paired_paths_from_folder([datarootlq, dataroothq], ['lq', 'gt'])
            self.dataset_len = len(self.paths)
            if self.data_len <= 0:
                self.data_len = self.dataset_len
            else:
                self.data_len = min(self.data_len, self.dataset_len)
        else:
            raise NotImplementedError('data_type [{:s}] is not recognized.'.format(datatype))


    def __len__(self):
        return self.data_len


    def scandir(self, dir_path, suffix=None, recursive=False, full_path=False):
        if (suffix is not None) and not isinstance(suffix, (str, tuple)):
            raise TypeError('"suffix" must be a string or tuple of strings')
        root = dir_path
        def _scandir(dir_path, suffix, recursive):
            for entry in os.scandir(dir_path):
                if not entry.name.startswith('.') and entry.is_file():
                    if full_path:
                        return_path = entry.path
                    else:
                        return_path = osp.relpath(entry.path, root)

                    if suffix is None:
                        yield return_path
                    elif return_path.endswith(suffix):
                        yield return_path
                else:
                    if recursive:
                        yield from _scandir(entry.path, suffix=suffix, recursive=recursive)
                    else:
                        continue
        return _scandir(dir_path, suffix=suffix, recursive=recursive)


    def paired_paths_from_folder(self, folders, keys):
        input_folder, gt_folder = folders
        input_key, gt_key = keys
        input_paths = list(self.scandir(input_folder))
        gt_paths = list(self.scandir(gt_folder))
        paths = []

        for lq_path in input_paths:
            basename, ext = osp.splitext(osp.basename(lq_path))
            #basename = basename.split("_")[0]
            input_name = basename + ext
            gt_path = osp.join(gt_folder, input_name)
            input_path = osp.join(input_folder, lq_path)
            paths.append(dict([(f'{input_key}_path', input_path), (f'{gt_key}_path', gt_path)]))
        return paths

    def paired_random_crop(self, img_gts, img_lqs, Density, gt_patch_size, scale=1):
        if not isinstance(img_gts, list):
            img_gts = [img_gts]
        if not isinstance(img_lqs, list):
            img_lqs = [img_lqs]
        if not isinstance(Density, list):
            Density = [Density]

        # determine input type: Numpy array or Tensor
        input_type = 'Tensor' if torch.is_tensor(img_gts[0]) else 'Numpy'

        if input_type == 'Tensor':
            h_gt, w_gt = img_gts[0].size()[-2:]

        # randomly choose top and left coordinates for lq patch
        top = random.randint(0, h_gt - gt_patch_size[0])
        left = random.randint(0, w_gt - gt_patch_size[1])

        # crop lq patch
        if input_type == 'Tensor':
            img_lqs = [v[:, top:top + gt_patch_size[0], left:left + gt_patch_size[1]] for v in img_lqs]

        # crop corresponding gt patch
        if input_type == 'Tensor':
            img_gts = [v[:, top:top + gt_patch_size[0], left:left + gt_patch_size[1]] for v in img_gts]

        if input_type == 'Tensor':
            Density = [v[:, top:top + gt_patch_size[0], left:left + gt_patch_size[1]] for v in Density]

        if len(img_gts) == 1:
            img_gts = img_gts[0]
        if len(img_lqs) == 1:
            img_lqs = img_lqs[0]
        if len(Density) == 1:
            Density = Density[0]

        return img_gts, img_lqs, Density


    def __getitem__(self, index):
        img_HR = Image.open(self.paths[index]['gt_path']).convert("RGB")
        img_SR = Image.open(self.paths[index]['lq_path']).convert("RGB")
        #patch_size = (img_SR.size[1]//16*16, img_SR.size[0]//16*16)
        #if img_SR.size[0] > 1024 and img_SR.size[1] > 1024:
        #    patch_size = (1024, 1024)
        patch_size = (self.img_sizeH, self.img_sizeW)

        [img_SR, img_HR] = Util.transform_augment([img_SR, img_HR], split=self.split, img_size=patch_size, min_max=(-1, 1))
        return {'HR': img_HR, 'SR': img_SR, 'Index': index}
        
        