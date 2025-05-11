# Copyright (c) Meta Platforms, Inc. and affiliates.

# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.


import os
from torchvision import transforms
from torch.utils.data import Dataset

from PIL import Image
import io
import torch
from .dct import shuffle_patch
import random
from tqdm import tqdm

import io

try:
    from torchvision.transforms import InterpolationMode
    BICUBIC = InterpolationMode.BICUBIC
except ImportError:
    BICUBIC = Image.BICUBIC

from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True
import kornia.augmentation as K




class RandomMask(object):
    def __init__(self, ratio=0.5, patch_size=16, p=0.5):
        """
        Args:
            ratio (float or tuple of float): If float, the ratio of the image to be masked.
                                             If tuple of float, random sample ratio between the two values.
            patch_size (int): the size of the mask (d*d).
        """
        if isinstance(ratio, float):
            self.fixed_ratio = True
            self.ratio = (ratio, ratio)
        elif isinstance(ratio, tuple) and len(ratio) == 2 and all(isinstance(r, float) for r in ratio):
            self.fixed_ratio = False
            self.ratio = ratio
        else:
            raise ValueError("Ratio must be a float or a tuple of two floats.")

        self.patch_size = patch_size
        self.p = p

    def __call__(self, tensor):

        if random.random() > self.p: return tensor

        _, h, w = tensor.shape
        mask = torch.ones((h, w), dtype=torch.float32)

        if self.fixed_ratio:
            ratio = self.ratio[0]
        else:
            ratio = random.uniform(self.ratio[0], self.ratio[1])

        # Calculate the number of masks needed
        num_masks = int((h * w * ratio) / (self.patch_size ** 2))

        # Generate non-overlapping random positions
        selected_positions = set()
        while len(selected_positions) < num_masks:
            top = random.randint(0, (h // self.patch_size) - 1) * self.patch_size
            left = random.randint(0, (w // self.patch_size) - 1) * self.patch_size
            selected_positions.add((top, left))

        for (top, left) in selected_positions:
            mask[top:top+self.patch_size, left:left+self.patch_size] = 0

        return tensor * mask.expand_as(tensor)



def Get_Transforms(arg, size=256, mode='dwt'):
    Perturbations = K.container.ImageSequential(
    K.RandomGaussianBlur(kernel_size=(3, 3), sigma=(0.1, 3.0), p=0.1),
    K.RandomJPEG(jpeg_quality=(60, 90), p=0.1)
    )

    transform_train = [
        transforms.RandomCrop([size, size], pad_if_needed=True),
        transforms.RandomHorizontalFlip(p=0.5),
    ]
    transform_eval = [transforms.CenterCrop([size, size])]

    if mode == 'dwt':
        transform_train.extend([
            transforms.ColorJitter(brightness=0.5, contrast=0.5, saturation=0.5),
            transforms.ToTensor(),
            # transforms.RandomRotation(180),
            RandomMask(ratio=(0.00, 0.75), patch_size=16, p=0.5),
        ])
        transform_eval.extend([
            transforms.ToTensor(),
            ])
    else:
        transform_train.extend([
            transforms.ToTensor(),
            transforms.Lambda(lambda x: Perturbations(x)[0]),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
        transform_eval.extend([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

    return transforms.Compose(transform_train), transforms.Compose(transform_eval)

    
class TrainDataset(Dataset):
    def __init__(self, is_train, args):
        
        self.is_train = is_train
        root = args.data_path if is_train else args.eval_data_path
        dataid = 'UCAS'  # progan
        self.data_list = []
        TRANSFORM_DWT = Get_Transforms(args, 256, mode='dwt')
        self.transform_dwt = TRANSFORM_DWT[0] if is_train else TRANSFORM_DWT[1]
        TRANSFORM_CLIP = Get_Transforms(args, 256, mode='clip')
        self.transform_clip = TRANSFORM_CLIP[0] if is_train else TRANSFORM_CLIP[1]

        if dataid == 'UCAS' and is_train:
            for image_path in os.listdir(os.path.join(root, '0_real')):
                self.data_list.append({"image_path": os.path.join(root, '0_real', image_path), "label" : 0})
            for image_path in os.listdir(os.path.join(root, '1_fake')):
                self.data_list.append({"image_path": os.path.join(root, '1_fake', image_path), "label" : 1})
        elif dataid == 'UCAS' and not is_train:
            for image_path in os.listdir(os.path.join(root, '0_real')):
                self.data_list.append({"image_path": os.path.join(root, '0_real', image_path), "label" : 0})
            for image_path in os.listdir(os.path.join(root, '1_fake')):
                self.data_list.append({"image_path": os.path.join(root, '1_fake', image_path), "label" : 1})
        elif dataid == 'ADM' and is_train:
            for image_path in os.listdir(os.path.join(root, 'nature')):
                self.data_list.append({"image_path": os.path.join(root, 'nature', image_path), "label" : 0})
            for image_path in os.listdir(os.path.join(root, 'ai')):
                self.data_list.append({"image_path": os.path.join(root, 'ai', image_path), "label" : 1})
        
        elif dataid == 'progan' and is_train:
            train_classes = ['car', 'cat', 'chair', 'horse']
            for class_name in train_classes:
                file_path = os.path.join(root, class_name)
                print(file_path, '||', os.listdir(file_path))
                for image_path in os.listdir(os.path.join(file_path, '0_real')):
                    self.data_list.append({"image_path": os.path.join(file_path, '0_real', image_path), "label" : 0})
                for image_path in os.listdir(os.path.join(file_path, '1_fake')):
                    self.data_list.append({"image_path": os.path.join(file_path, '1_fake', image_path), "label" : 1})
        else:
            # print('------------------------------1111111111111111111111111111111-------------------------------------')
            for filename in os.listdir(root):
                file_path = os.path.join(root, filename)
                print(file_path, '||', os.listdir(file_path))
                if '0_real' not in os.listdir(file_path):
                    # print('111111111111111')
                    for folder_name in os.listdir(file_path):
                    
                        assert os.listdir(os.path.join(file_path, folder_name)) == ['0_real', '1_fake']

                        for image_path in os.listdir(os.path.join(file_path, folder_name, '0_real')):
                            self.data_list.append({"image_path": os.path.join(file_path, folder_name, '0_real', image_path), "label" : 0})
                    
                        for image_path in os.listdir(os.path.join(file_path, folder_name, '1_fake')):
                            self.data_list.append({"image_path": os.path.join(file_path, folder_name, '1_fake', image_path), "label" : 1})
                
                else:
                    for image_path in os.listdir(os.path.join(file_path, '0_real')):
                        self.data_list.append({"image_path": os.path.join(file_path, '0_real', image_path), "label" : 0})
                    for image_path in os.listdir(os.path.join(file_path, '1_fake')):
                        self.data_list.append({"image_path": os.path.join(file_path, '1_fake', image_path), "label" : 1})


    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, index):
        
        sample = self.data_list[index]
                
        image_path, targets = sample['image_path'], sample['label']
        image = Image.open(image_path).convert('RGB')
        image_dwt = self.transform_dwt(image)
        image_dwt = shuffle_patch(image_dwt, n=8)  # 打乱 patch
        image_clip = self.transform_clip(image)
        return torch.stack([image_dwt, image_clip], dim=0), torch.tensor(int(targets))

    

class TestDataset(Dataset):
    def __init__(self, is_train, args):
        
        root = args.data_path if is_train else args.eval_data_path

        self.data_list = []

        file_path = root
        TRANSFORM_DWT = Get_Transforms(args, 256, mode='dwt')
        self.transform_dwt = TRANSFORM_DWT[0] if is_train else TRANSFORM_DWT[1]
        TRANSFORM_CLIP = Get_Transforms(args, 256, mode='clip')
        self.transform_clip = TRANSFORM_CLIP[0] if is_train else TRANSFORM_CLIP[1]

        if '0_real' not in os.listdir(file_path):
            for folder_name in os.listdir(file_path):
    
                assert os.listdir(os.path.join(file_path, folder_name)) == ['0_real', '1_fake']
                
                for image_path in os.listdir(os.path.join(file_path, folder_name, '0_real')):
                    self.data_list.append({"image_path": os.path.join(file_path, folder_name, '0_real', image_path), "label" : 0})
                
                for image_path in os.listdir(os.path.join(file_path, folder_name, '1_fake')):
                    self.data_list.append({"image_path": os.path.join(file_path, folder_name, '1_fake', image_path), "label" : 1})
        
        else:
            for image_path in os.listdir(os.path.join(file_path, '0_real')):
                self.data_list.append({"image_path": os.path.join(file_path, '0_real', image_path), "label" : 0})
            for image_path in os.listdir(os.path.join(file_path, '1_fake')):
                self.data_list.append({"image_path": os.path.join(file_path, '1_fake', image_path), "label" : 1})


    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, index):
        
        sample = self.data_list[index]
                
        image_path, targets = sample['image_path'], sample['label']

        image = Image.open(image_path).convert('RGB')
        image_dwt = self.transform_dwt(image)
        image_dwt = shuffle_patch(image_dwt, n=8)  # 打乱 patch
        image_clip = self.transform_clip(image)
        
        return torch.stack([image_dwt, image_clip], dim=0), torch.tensor(int(targets))


class TestDataset_ucas(Dataset):
    def __init__(self, is_train, args):
        root = args.data_path if is_train else args.eval_data_path
        self.data_list = []

        file_path = root
        TRANSFORM_DWT = Get_Transforms(args, 256, mode='dwt')
        self.transform_dwt = TRANSFORM_DWT[0] if is_train else TRANSFORM_DWT[1]
        TRANSFORM_CLIP = Get_Transforms(args, 256, mode='clip')
        self.transform_clip = TRANSFORM_CLIP[0] if is_train else TRANSFORM_CLIP[1]
    
        # 新的纯图片结构，没有label
        for image_name in os.listdir(file_path):
            self.data_list.append({"image_path": os.path.join(file_path, image_name), "label": -1})  # label填-1占位

    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, index):
        sample = self.data_list[index]
        image_path, targets = sample['image_path'], sample['label']

        image = Image.open(image_path).convert('RGB')
        image_dwt = self.transform_dwt(image)
        # image_dwt = shuffle_patch(image_dwt, n=8)  # 打乱 patch
        image_clip = self.transform_clip(image)
        

        # 注意：这里返回了image_path，方便主程序记录图片名
        return torch.stack([image_dwt, image_clip], dim=0), targets, image_path