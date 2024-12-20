import os
import json
from os.path import join

import numpy as np
# import scipy
# from scipy import io
# import scipy.misc
from PIL import Image
# import pandas as pd
# import matplotlib.pyplot as plt

import torch
from torch.utils.data import Dataset
# from torchvision.datasets import VisionDataset
# from torchvision.datasets.folder import default_loader
# from torchvision.datasets.utils import download_url, list_dir, check_integrity, extract_archive, verify_str_arg
import torchvision.transforms.functional as TF
# import imageio
import cv2
import pdb


class Nutrition(Dataset):
    def __init__(self, image_path, txt_dir, transform=None):

        file = open(txt_dir, 'r')
        lines = file.readlines()
        self.images = []
        self.labels = []
        self.total_calories = []
        self.total_mass = []
        self.total_fat = []
        self.total_carb = []
        self.total_protein = []
        # pdb.set_trace()
        for line in lines:
            image = line.split()[0]  
            label = line.strip().split()[1]  #
            calories = line.strip().split()[2]
            mass = line.strip().split()[3]
            fat = line.strip().split()[4]
            carb = line.strip().split()[5]
            protein = line.strip().split()[6]

            self.images += [os.path.join(image_path, image)]  
            self.labels += [str(label)]
            self.total_calories += [np.array(float(calories))]
            self.total_mass += [np.array(float(mass))]
            self.total_fat += [np.array(float(fat))]
            self.total_carb += [np.array(float(carb))]
            self.total_protein += [np.array(float(protein))]
        # pdb.set_trace()
        # self.transform_rgb = transform[0]

        self.transform = transform

    def __getitem__(self, index):
        # img = cv2.imread(self.images[index])
        # try:
        #     # img = cv2.resize(img, (self.imsize, self.imsize))
        #     img = Image.fromarray(cv2.cvtColor(img,cv2.COLOR_BGR2RGB)) # 
        # except:
        #     print("图片有误：",self.images[index])
        img = Image.open(self.images[index]).convert('RGB')
        if self.transform is not None:
            try:
                # lmj  RGB-D图像尺寸不同,按照不同比例缩放
                if 'realsense_overhead' in self.images[index]:
                    # pdb.set_trace()
                    self.transform.transforms[0].size = (267, 356)
                    # print(self.transform)
                img = self.transform(img)
            except:
                # print('trans_img', img)
                print('trans_img有误')
        return img, self.labels[index], self.total_calories[index], self.total_mass[index], self.total_fat[index], \
               self.total_carb[index], self.total_protein[index]

    def __len__(self):
        return len(self.images)


# RGB-D
class Nutrition_RGBD(Dataset):
    def __init__(self, image_path, rgb_txt_dir, rgbd_txt_dir, transform=None):

        file_rgb = open(rgb_txt_dir, 'r')  # 
        file_rgbd = open(rgbd_txt_dir, 'r')  # 
        lines_rgb = file_rgb.readlines()
        lines_rgbd = file_rgbd.readlines()
        self.images = []
        self.labels = []
        self.total_calories = []
        self.total_mass = []
        self.total_fat = []
        self.total_carb = []
        self.total_protein = []
        self.images_rgbd = []
        # pdb.set_trace()
        for line in lines_rgb:
            image_rgb = line.split()[0]  
            label = line.strip().split()[1]  #
            calories = line.strip().split()[2]
            mass = line.strip().split()[3]
            fat = line.strip().split()[4]
            carb = line.strip().split()[5]
            protein = line.strip().split()[6]

            self.images += [os.path.join(image_path, image_rgb)]  # 
            self.labels += [str(label)]
            self.total_calories += [np.array(float(calories))]
            self.total_mass += [np.array(float(mass))]
            self.total_fat += [np.array(float(fat))]
            self.total_carb += [np.array(float(carb))]
            self.total_protein += [np.array(float(protein))]
        for line in lines_rgbd:  # 
            image_rgbd = line.split()[0]
            self.images_rgbd += [os.path.join(image_path, image_rgbd)]

            # pdb.set_trace()
        # self.transform_rgb = transform[0]

        self.transform = transform

    # RGB-D  20210805
    def my_loader(path, Type):
        with open(path, 'rb') as f:
            with Image.open(f) as img:
                if Type == 3:
                    img = img.convert('RGB')
                elif Type == 1:
                    img = img.convert('L')
                return img

    def __getitem__(self, index):
        img_rgb = cv2.imread(self.images[index])  # 
        img_rgbd = cv2.imread(self.images_rgbd[index])
        try:
            # img = cv2.resize(img, (self.imsize, self.imsize))
            img_rgb = Image.fromarray(cv2.cvtColor(img_rgb, cv2.COLOR_BGR2RGB))  # 
            img_rgbd = Image.fromarray(cv2.cvtColor(img_rgbd, cv2.COLOR_BGR2RGB))  # 
        except:
            print("图片有误：", self.images[index])
        # 4通道
        # rgb_path, d_path = self.images[index], self.images_rgbd[index]
        # rgb_img = np.array(self.my_loader(rgb_path, 3))
        # d_img = np.array(self.my_loader(d_path, 1) )
        # d_img = np.expand_dims(d_img, axis=2) #(480, 640, 1)
        # img = np.append(rgb_img, d_img, axis=2) # (480, 640, 4)

        if self.transform is not None:
            # img = self.transform(Image.fromarray(img))
            img_rgb = self.transform(img_rgb)
            img_rgbd = self.transform(img_rgbd)

        return img_rgb, self.labels[index], self.total_calories[index], self.total_mass[index], self.total_fat[index], \
               self.total_carb[index], self.total_protein[
                   index], img_rgbd  

    def __len__(self):
        return len(self.images)


############################################################
class SyncRandomRotateFlip:

    def __init__(self, p=0.5):
        self.p = p

    def __call__(self, rgb, depth):
        # 随机水平翻转
        if torch.rand(1) < self.p:
            rgb = TF.hflip(rgb)
            depth = TF.hflip(depth)

        # 随机垂直翻转（如果需要）
        if torch.rand(1) < self.p:
            rgb = TF.vflip(rgb)
            depth = TF.vflip(depth)
        return rgb, depth
###########################################
class Nutrition_RGBD_ingr(Dataset):
    def __init__(self, image_path, rgb_txt_dir, rgbd_txt_dir, ingr_txt_dir,enable_flip=False,transform=None):

        file_rgb = open(rgb_txt_dir, 'r')  # 
        file_rgbd = open(rgbd_txt_dir, 'r')  # 
        file_ingr = open(ingr_txt_dir, 'r')  # 
        lines_rgb = file_rgb.readlines()
        lines_rgbd = file_rgbd.readlines()
        lines_ingr = file_ingr.readlines()
        self.images = []
        self.labels = []
        self.total_calories = []
        self.total_mass = []
        self.total_fat = []
        self.total_carb = []
        self.total_protein = []
        self.images_rgbd = []
        # self.images_ingr = []
        self.ingr = []
        # pdb.set_trace()
        for line in lines_rgb:
            image_rgb = line.split()[0]  # 
            label = line.strip().split()[1]  # 
            calories = line.strip().split()[2]
            mass = line.strip().split()[3]
            fat = line.strip().split()[4]
            carb = line.strip().split()[5]
            protein = line.strip().split()[6]

            self.images += [os.path.join(image_path, image_rgb)]  #
            self.labels += [str(label)]
            self.total_calories += [np.array(float(calories))]
            self.total_mass += [np.array(float(mass))]
            self.total_fat += [np.array(float(fat))]
            self.total_carb += [np.array(float(carb))]
            self.total_protein += [np.array(float(protein))]
        for line in lines_rgbd:  # 
            image_rgbd = line.split()[0]
            self.images_rgbd += [os.path.join(image_path, image_rgbd)]

        for line in lines_ingr:  # 
            # image_ingr = line.split()[0]
            # dish_ingr = line.strip().split()[2].split(',')
            dish_ingr = line.strip().split()[2]
            # print(dish_ingr)
            # print(type(dish_ingr)):str
            dish_ingr = dish_ingr.replace(',', '')
            dish_list = [int(label) for label in dish_ingr]
            # print(dish_list)
            # dish_ingr = np.array(dish_ingr)
            # self.ingr += [np.array(dish_ingr).astype(float)]  # 
            self.ingr.append(dish_list)
            # print(self.ingr)
            # print(type(self.ingr)):list
        # self.ingr = np.array(self.ingr).astype(float)
        self.ingr = np.array(self.ingr,dtype=np.float32)#
        # print(self.ingr)

        # self.ingr = np.array(self.ingr)
        # print(type(self.ingr))、


        # print(self.ingr)
            # dish_ingr=line.strip().split()[2].split(',')
            # ingr_label = [int(x) for x in dish_ingr] # 
            # self.ingr += [ingr_label]


            # pdb.set_trace()
        # self.transform_rgb = transform[0]

        self.transform = transform
        self.flip = enable_flip
        self.rf = SyncRandomRotateFlip(p=0.5)

    # RGB-D  20210805
    def my_loader(path, Type):
        with open(path, 'rb') as f:
            with Image.open(f) as img:
                if Type == 3:
                    img = img.convert('RGB')
                elif Type == 1:
                    img = img.convert('L')
                return img

    def __getitem__(self, index):
        img_rgb = cv2.imread(self.images[index])  # 
        img_rgbd = cv2.imread(self.images_rgbd[index])
        try:
            # img = cv2.resize(img, (self.imsize, self.imsize))
            img_rgb = Image.fromarray(cv2.cvtColor(img_rgb, cv2.COLOR_BGR2RGB))  # 
            img_rgbd = Image.fromarray(cv2.cvtColor(img_rgbd, cv2.COLOR_BGR2RGB))  #
        except:
            print("图片有误：", self.images[index])
        # 4通道
        # rgb_path, d_path = self.images[index], self.images_rgbd[index]
        # rgb_img = np.array(self.my_loader(rgb_path, 3))
        # d_img = np.array(self.my_loader(d_path, 1) )
        # d_img = np.expand_dims(d_img, axis=2) #(480, 640, 1)
        # img = np.append(rgb_img, d_img, axis=2) # (480, 640, 4)

        if self.transform is not None:
            # img = self.transform(Image.fromarray(img))
            img_rgb = self.transform(img_rgb)
            img_rgbd = self.transform(img_rgbd)

        if self.flip:
            img_rgb,img_rgbd = self.rf(img_rgb,img_rgbd)

        # print( self.labels[index], self.total_calories[index], self.total_mass[index], self.total_fat[index], \
        #     self.total_carb[index], self.total_protein[index],  self.ingr[index])
        # print(type(self.ingr[index]))
        # print( self.ingr[index].shape)
        # print(self.labels[index],self.ingr[index])
        # return img_rgb, self.labels[index], self.total_calories[index], self.total_mass[index], self.total_fat[index], \
        #        self.total_carb[index], self.total_protein[index], img_rgbd, torch.FloatTensor(self.ingr[index])# 
        return img_rgb, self.labels[index], self.total_calories[index], self.total_mass[index], self.total_fat[index], \
               self.total_carb[index], self.total_protein[index], img_rgbd, self.ingr[index]


    def __len__(self):
        return len(self.images)




