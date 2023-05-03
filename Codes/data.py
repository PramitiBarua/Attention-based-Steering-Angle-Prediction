import os
import random
import numpy as np
from PIL import Image

import torch
from torch.utils.data import Dataset, DataLoader, Sampler
from torchvision import transforms

from config import img_dir, csv_src, net

class TruckDataset(Dataset):
    """
    # (1) must cast from np to tensor: torch.from_numpy(x).permute(2, 0, 1) # D, H, W
    # (2) transformations: resize, totensor, normalize, augmentation?
    # (3) must return leftImg, centerImg, rightImg, leftAng, centerAng, rightAng, all of shape N x Img
    """
    def __init__(self, X, y):
        self.img_names = X
        self.angles = y
        self.model_name = net

    def __len__(self):
        return len(self.img_names)
    
    def __getitem__(self, index):

        print(self.img_names)
        name = self.img_names[index].split('/center')[1]
        front_name, left_name, right_name = os.path.join('/users/PCS0269/pbarua/Steering_Angle_Prediction/data/dataset2/', 'IMG/center' + name), os.path.join('/users/PCS0269/pbarua/Steering_Angle_Prediction/data/dataset2/', 'IMG/left' + name), os.path.join('/users/PCS0269/pbarua/Steering_Angle_Prediction/data/dataset2/', 'IMG/right' + name)
        #front_name, left_name, right_name = os.path.join('/users/PCS0269/pbarua/Steering_Angle_Prediction/data/dataset/dataset/', 'IMG/center' + name), os.path.join('/users/PCS0269/pbarua/Steering_Angle_Prediction/data/dataset/dataset/', 'IMG/left' + name), os.path.join('/users/PCS0269/pbarua/Steering_Angle_Prediction/data/dataset/dataset/', 'IMG/right' + name)
        front_img, left_img, right_img = np.array(Image.open(front_name)), np.array(Image.open(left_name)), np.array(Image.open(right_name))
        front_angle, left_angle, right_angle = self.angles[index], self.angles[index] + 0.4, self.angles[index] - 0.4

        front_img, front_angle = self.process(front_img, front_angle, self.model_name)
        left_img, left_angle = self.process(left_img, left_angle, self.model_name)
        right_img, right_angle = self.process(right_img, right_angle, self.model_name)

        return left_img, front_img, right_img, left_angle, front_angle, right_angle

    def process(self, img, angle, model_name):

        img = torch.from_numpy(img).permute(2, 0, 1) # D, H, W

        if model_name == "TruckResnet18":
            size = (224, 224)
        elif model_name == "TruckResnet34":
            size = (224, 224)
        elif model_name == "TruckResnet50":
            size = (224, 224)
        elif model_name == "TruckResnet101":
            size = (224, 224)
        elif model_name == "TruckResnet152":
            size = (224, 224)
        elif model_name == "ResNet":
            size = (224, 224)
        elif model_name == "ResNetLarge":
            size = (224, 224)
        elif model_name == "GoogLeNet":
            size = (224, 224)
        elif model_name == "GoogLeNetPlus":
            size = (224, 224)
        elif model_name == "CBAM":
            size = (224, 224)
        elif model_name == "attention":
            size = (224, 224)

        transform = transforms.Compose([
            transforms.Resize(size),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
            transforms.Lambda(lambda x: (x / 127.5) - 1),
        ])

        img = transform(img)

        # augmentation
        if np.random.rand() < 0.4:
            flip_transform = transforms.Compose([
                transforms.RandomHorizontalFlip(p=1)
            ])

            img = flip_transform(img)
            angle = angle * -1.0
        
        return img, angle


class TruckNNSampler(Sampler):
    def __init__(self, data_source, batch_size, seq_len):
        super(TruckNNSampler, self).__init__(data_source)
        self.data_source = data_source
        self.batch_size = batch_size
        self.seq_len = seq_len
        self.data_len = len(self.data_source)
    
    def __len__(self):
        idx_len = len(list(range(1, self.data_len, self.seq_len)))
        if idx_len // self.batch_size == 0:
            return idx_len // self.batch_size
        return 1 + idx_len // self.batch_size

    def __iter__(self):
        # get indices
        idx_list = list(range(1, self.data_len, self.seq_len))
        random.shuffle(idx_list)

        # get batches 
        cur_batch = []
        for i, idx in enumerate(idx_list):
            if idx + 1 < self.seq_len: # not enough. Pad with 0's.
                seq = [0] * (self.seq_len - idx - 1) + list(range(0, idx + 1))
            else:
                seq = list(range(idx - self.seq_len + 1, idx + 1))
        
            cur_batch.append(tuple(seq))
            if i + 1 == self.data_len or len(cur_batch) == self.batch_size:
                yield cur_batch
                cur_batch = []
