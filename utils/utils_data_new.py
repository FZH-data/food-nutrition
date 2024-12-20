import logging
from PIL import Image
import os
import torch
from torchvision import transforms
from torch.utils.data import DataLoader, RandomSampler, DistributedSampler, SequentialSampler
from torchvision.transforms.transforms import CenterCrop
from mydataset_new import Nutrition_RGBD, Nutrition,Nutrition_RGBD_ingr
import pdb
import random

def get_DataLoader(args):

    if args.dataset == 'nutrition_rgb':

        train_transform = transforms.Compose([
                                    # transforms.RandomRotation(degrees=(0, 180)),
                                    transforms.Resize((270, 480), Image.BILINEAR),
                                    # transforms.RandomHorizontalFlip(),
                                    transforms.CenterCrop((256,256)), #256
                                    # transforms.ColorJitter(hue=0.05),
                                    transforms.ToTensor(),
                                    transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])])
        test_transform = transforms.Compose([
                                    transforms.Resize((270, 480), Image.BILINEAR),
                                    transforms.CenterCrop((256, 256)), #256
                                    transforms.ToTensor(),
                                    transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])])

        if 'T2t_vit' in args.model or 'vit' in  args.model: # imagesize = 224* 224
            pdb.set_trace()
            print(args.model)
            train_transform = transforms.Compose([
                                    transforms.Resize((270, 480)),
                                    transforms.CenterCrop((256,256)),
                                    transforms.Resize((224, 224)),
                                    transforms.ToTensor(),
                                    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
            test_transform = transforms.Compose([
                                    transforms.Resize((270, 480)),
                                    transforms.CenterCrop((256, 256)),
                                    transforms.Resize((224, 224)),
                                    transforms.ToTensor(),
                                    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])

        nutrition_rgb_ims_root = os.path.join(args.data_root, 'imagery')
        if args.rgbd_after_check:
            # pdb.set_trace()
            nutrition_train_txt = os.path.join(args.data_root, 'imagery','rgb_train_processed_tianhao.txt')
            nutrition_test_txt = os.path.join(args.data_root, 'imagery','rgb_test_processed_tianhao.txt')
        else:
            nutrition_train_txt = os.path.join(args.data_root, 'imagery','rgb_train_processed.txt')
            nutrition_test_txt = os.path.join(args.data_root, 'imagery','rgb_test_processed.txt')
        trainset = Nutrition(image_path = nutrition_rgb_ims_root, txt_dir = nutrition_train_txt, transform = train_transform)
        testset = Nutrition(image_path = nutrition_rgb_ims_root, txt_dir = nutrition_test_txt, transform = test_transform)

    elif args.dataset == 'nutrition_rgbd':
        image_sizes = ((256, 352), (320, 448))
        train_transform = transforms.Compose([
                                    # transforms.RandomRotation(degrees=(0, 180)),
                                    # transforms.Resize(image_sizes[random.randint(0,1)]),

                                    # transforms.Resize((336, 448)),
                                    transforms.Resize((384, 384)),#
                                    # transforms.RandomHorizontalFlip(),############
                                    # transforms.CenterCrop((384, 384)),
                                    # transforms.ColorJitter(hue=0.05),
                                    transforms.ToTensor(),
                                    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
                                    # transforms.Normalize([0.5, 0.5, 0.5, 0.5], [0.5, 0.5, 0.5, 0.5])
                                    ])
        test_transform = transforms.Compose([
                                    # transforms.Resize(image_sizes[random.randint(0,1)]),

                                    # transforms.Resize((336, 448)),
                                    transforms.Resize((384, 384)),#
                                    # transforms.CenterCrop((256, 256)),
                                    transforms.ToTensor(),
                                    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
                                    # transforms.Normalize([0.5, 0.5, 0.5, 0.5], [0.5, 0.5, 0.5, 0.5])
                                    ])

        nutrition_rgbd_ims_root = os.path.join(args.data_root, 'imagery')
        nutrition_train_txt = os.path.join(args.data_root, 'imagery','rgbd_train_processed.txt')
        nutrition_test_txt = os.path.join(args.data_root, 'imagery','rgbd_test_processed.txt') # depth_color.png
        nutrition_train_rgbd_txt = os.path.join(args.data_root, 'imagery','rgb_in_overhead_train_processed.txt')
        nutrition_test_rgbd_txt = os.path.join(args.data_root, 'imagery','rgb_in_overhead_test_processed.txt') # rbg.png
        nutrition_train_dish_ingredient = os.path.join(args.data_root, 'imagery',
                                                       'train_rgb_dish_ingredient1.txt')  
        nutrition_test_dish_ingredient = os.path.join(args.data_root, 'imagery', 'test_rgb_dish_ingredient_new_1.txt')
        trainset = Nutrition_RGBD_ingr(nutrition_rgbd_ims_root, nutrition_train_rgbd_txt, nutrition_train_txt,nutrition_train_dish_ingredient,enable_flip=True,transform=train_transform)
        testset = Nutrition_RGBD_ingr(nutrition_rgbd_ims_root, nutrition_test_rgbd_txt, nutrition_test_txt,nutrition_test_dish_ingredient,enable_flip=False,transform=test_transform)

    train_loader = DataLoader(trainset,
                              batch_size=args.b,
                              shuffle=True,
                              num_workers=0,#多进程
                              pin_memory=True
                              )
    test_loader = DataLoader(testset,
                             batch_size=args.b,
                             shuffle=False,
                             num_workers=0,
                             pin_memory=True
                             )

    return train_loader, test_loader



