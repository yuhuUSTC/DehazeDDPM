import os
import torch
import torchvision
import random
import numpy as np
from torchvision import transforms, utils
IMG_EXTENSIONS = ['.jpg', '.JPG', '.jpeg', '.JPEG', '.png', '.PNG', '.ppm', '.PPM', '.bmp', '.BMP']


def is_image_file(filename):
    return any(filename.endswith(extension) for extension in IMG_EXTENSIONS)


def get_paths_from_images(path):
    assert os.path.isdir(path), '{:s} is not a valid directory'.format(path)
    images = []
    for dirpath, _, fnames in sorted(os.walk(path)):
        for fname in sorted(fnames):
            if is_image_file(fname):
                img_path = os.path.join(dirpath, fname)
                images.append(img_path)
    assert images, '{:s} has no valid image file'.format(path)
    return sorted(images)


def augment(img_list, hflip=True, rot=True, split='val'):
    # horizontal flip OR rotate
    hflip = hflip and (split == 'train' and random.random() < 0.5)
    vflip = rot and (split == 'train' and random.random() < 0.5)
    rot90 = rot and (split == 'train' and random.random() < 0.5)

    def _augment(img):
        if hflip:
            img = img[:, ::-1, :]
        if vflip:
            img = img[::-1, :, :]
        if rot90:
            img = img.transpose(1, 0, 2)
        return img

    return [_augment(img) for img in img_list]


def transform2numpy(img):
    img = np.array(img)
    img = img.astype(np.float32) / 255.
    if img.ndim == 2:
        img = np.expand_dims(img, axis=2)
    # some images have 4 channels
    if img.shape[2] > 3:
        img = img[:, :, :3]
    return img


def transform2tensor(img, min_max=(0, 1)):
    # HWC to CHW
    img = torch.from_numpy(np.ascontiguousarray(
        np.transpose(img, (2, 0, 1)))).float()
    # to range min_max
    img = img*(min_max[1] - min_max[0]) + min_max[0]
    return img


# implementation by numpy and torch
# def transform_augment(img_list, split='val', min_max=(0, 1)):
#     imgs = [transform2numpy(img) for img in img_list]
#     imgs = augment(imgs, split=split)
#     ret_img = [transform2tensor(img, min_max) for img in imgs]
#     return ret_img




def paired_random_crop(img_gts, img_lqs, gt_patch_size):
    if not isinstance(img_gts, list):
        img_gts = [img_gts]
    if not isinstance(img_lqs, list):
        img_lqs = [img_lqs]
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


    if len(img_gts) == 1:
        img_gts = img_gts[0]
    if len(img_lqs) == 1:
        img_lqs = img_lqs[0]

    return img_gts, img_lqs



#implementation by torchvision, detail in https://github.com/Janspiry/Image-Super-Resolution-via-Iterative-Refinement/issues/14 
totensor = torchvision.transforms.ToTensor()
hflip = torchvision.transforms.RandomHorizontalFlip()
resize = transforms.RandomResizedCrop(512,scale=(0.5,1.0))
def transform_augment(img_list, split='val', img_size=(512, 512), min_max=(0, 1)):
    Resize = transforms.Resize(img_size)
    img_list = [Resize(img) for img in img_list]
    imgs = [totensor(img) for img in img_list]
    if split == 'train':
        imgs = torch.stack(imgs, 0)
        imgs = hflip(imgs)
        imgs = torch.unbind(imgs, dim=0)
    ret_img = [img * (min_max[1] - min_max[0]) + min_max[0] for img in imgs]
    return ret_img

# totensor = torchvision.transforms.ToTensor()
# hflip = torchvision.transforms.RandomHorizontalFlip()
# resize = transforms.RandomResizedCrop(512,scale=(0.5,1.0))
# def transform_augment(img_list, split='val', img_size=(512, 512), min_max=(0, 1)):
#     imgs = [totensor(img) for img in img_list]
#     imgs[0], imgs[1] = paired_random_crop(imgs[0], imgs[1], img_size)
#     if split == 'train':
#         imgs = torch.stack(imgs, 0)
#         imgs = hflip(imgs)
#         imgs = torch.unbind(imgs, dim=0)
#     ret_img = [img * (min_max[1] - min_max[0]) + min_max[0] for img in imgs]
#     return ret_img
    
    

    
    
    
    