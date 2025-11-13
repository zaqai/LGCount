import json
import numpy as np
import random
from torchvision import transforms
import torch
import cv2
import torchvision.transforms.functional as TF
import scipy.ndimage as ndimage
from PIL import Image
from torch.utils.data import Dataset
import os
import imgaug as ia
import imgaug.augmenters as iaa
import pickle
from imgaug.augmentables import Keypoint, KeypointsOnImage
# import utils.debug_utils
MAX_HW = 384
IM_NORM_MEAN = [0.485, 0.456, 0.406]
IM_NORM_STD = [0.229, 0.224, 0.225]



class ZSC8K(Dataset):
    def __init__(self, 
                 split:str, 
                 subset_scale:float=1.0, 
                 resize_val:bool=True, 
                 additional_prompt:bool=True):
        """
        Parameters
        ----------
        split : str, 'train', 'val' or 'test'
        subset_scale : float, scale of the subset of the dataset to use
        resize_val : bool, whether to random crop validation images to 384x384
        """
        assert split in ['train', 'val', 'test' , 'val_coco', 'test_coco']

        #!HARDCODED Dec 25: 
        self.data_dir = "data/zsc-8k"
        self.dataset_type = 'FSC_147'

        self.resize_val = resize_val
        self.im_dir = os.path.join(self.data_dir,'images')
        self.gt_dir = os.path.join(self.data_dir, 'gt_density')
        self.anno_file = os.path.join(self.data_dir,'anno','ZSC-8k.json')
        self.split_file = os.path.join(self.data_dir,'anno','ZSC_splits.json')
        self.split = split
        
        self.class_dict={}
        self.point_dict={}
        self.split_dict={}
        
        self.read_classes()
        self.read_points()
        self.read_splits()
        print(f"train: {len(self.split_dict['train'])}\nval: {len(self.split_dict['val'])}\ntest: {len(self.split_dict['test'])}")

        self.idx_running_set = self.split_dict[split]




        self.transform = None
        if self.split == 'train' or (self.split == 'val' and self.resize_val):
            use_aug = self.split == 'train'
            self.transform = transforms.Compose([ResizeTrainImage(MAX_HW, self, aug=use_aug)])
        random.shuffle(self.idx_running_set)

    def read_classes(self):
        with open(self.anno_file) as f:
            anno = json.load(f)
            for key, value in anno.items():
                self.class_dict[key] = value['class']
    def read_points(self):
        with open(self.anno_file) as f:
            anno = json.load(f)
            for key, value in anno.items():
                self.point_dict[key] = value['points']
    def read_splits(self):
        with open(self.split_file) as f:
            splits = json.load(f)
            self.split_dict = splits
        
            
    def __len__(self):
        return len(self.idx_running_set)

    def __getitem__(self, idx):
        im_id = self.idx_running_set[idx]
        anno = self.point_dict[im_id]
        text = self.class_dict[im_id]

        coarse_interval_integer=[0, 25, 50, 75, 120, 300, 550]
        fine_interval_integer=[
                              [0, 5, 10, 15, 20, 25],
                              [25, 30, 35, 40, 45, 50],
                              [50, 55, 60, 65, 70, 75],
                              [75, 80, 85, 90, 100, 120],
                              [120, 140, 170, 200, 250, 300],
                              [300, 350, 400, 450, 500, 550],
                              ]
        
        coarse_text_list=[]
        fine_text_list=[]
        if self.split == 'train' or (self.split == 'val' and self.resize_val):


            dots = np.array(anno)

            image = Image.open('{}/{}'.format(self.im_dir, im_id))
            image.load()
            density_path = self.gt_dir + '/' + im_id.split(".")[0] + ".npy"
            density = np.load(density_path).astype('float32')   
            m_flag = 0
            

            sample = {'image':image, 'gt_density':density, 'dots':dots, 'id':im_id,'m_flag':m_flag}

            sample = self.transform(sample)
            GT_count = torch.sum(sample['gt_density']/60).item()
            coarse_GT=np.searchsorted(coarse_interval_integer, GT_count, side='left')-1
            if coarse_GT>5:
                coarse_GT=5
            fine_GT=np.searchsorted(fine_interval_integer[coarse_GT], GT_count, side='left')-1
            if fine_GT>4:
                fine_GT=4
            for i in range(len(coarse_interval_integer)-1):
                coarse_text_list.append(f'There are between {coarse_interval_integer[i]} and {coarse_interval_integer[i+1]} {text}')
                fine_text_list_temp=[]
                for j in range(len(fine_interval_integer)-1):
                    fine_text_list_temp.append(f'There are between {fine_interval_integer[i][j]} and {fine_interval_integer[i][j+1]} {text}')
                fine_text_list.append(fine_text_list_temp)
            return sample['image'].float(), sample['gt_density'], text, coarse_text_list, fine_text_list, coarse_GT, fine_GT, im_id
            # return sample['image'].float(), sample['gt_density'], sample['boxes'], sample['m_flag'], text
        elif self.split == "test" or self.split == "test_coco" or self.split == "val_coco" or (self.split == "val" and not self.resize_val):
            dots = np.array(anno)
            image = Image.open('{}/{}'.format(self.im_dir, im_id))
            text = self.class_dict[im_id]
            image.load()

            if image.size[1]>image.size[0]:
                image = image.rotate(-90, expand=True)
                new_dots=[]
                for (x, y) in dots:
                    x_ = image.size[0] -y
                    y_ = x
                    new_dots.append([x_, y_])
                dots = np.array(new_dots)
            image = transforms.Resize(384)(image)
            W, H = image.size
            new_H = 16*int(H/16)
            new_W = 16*int(W/16)
            image = transforms.Resize((new_H, new_W))(image)
            # image = transforms.Resize((new_H, new_W))(image)
            Normalize = transforms.Compose([transforms.ToTensor()])
            image = Normalize(image)
            for i in range(len(coarse_interval_integer)-1):
                coarse_text_list.append(f'There are between {coarse_interval_integer[i]} and {coarse_interval_integer[i+1]} {text}')
                fine_text_list_temp=[]
                for j in range(len(fine_interval_integer)-1):
                    fine_text_list_temp.append(f'There are between {fine_interval_integer[i][j]} and {fine_interval_integer[i][j+1]} {text}')
                fine_text_list.append(fine_text_list_temp)
            sample = {'image':image,'dots':dots}
            return sample['image'].float(), dots, text, coarse_text_list, fine_text_list, im_id


class ResizeTrainImage(object):
    """
    Resize the image so that:
        1. Image is equal to 384 * 384
        2. The new height and new width are divisible by 16
        3. The aspect ratio is possibly preserved
    Density map is cropped to have the same size(and position) with the cropped image
    Exemplar boxes may be outside the cropped area.
    Augmentation including Gaussian noise, Color jitter, Gaussian blur, Random affine, Random horizontal flip and Mosaic (or Random Crop if no Mosaic) is used.
    """

    def __init__(self, MAX_HW=384, dataset:ZSC8K=None, aug = True):
        self.max_hw = MAX_HW
        self.dataset = dataset
        self.use_out_mosaic = False
        self.use_augmentation = aug
    def __call__(self, sample):
        image, density, dots, im_id, m_flag = sample['image'], sample['gt_density'], sample['dots'], sample['id'], sample['m_flag']
        # make sure the width is larger than height
        if image.size[1]>image.size[0]:
            image = image.rotate(-90, expand=True)
            density = np.rot90(density, k=-1)
            new_dots=[]
            for (x, y) in dots:
                x_ = image.size[0] -y
                y_ = x
                new_dots.append([x_, y_])
            dots = np.array(new_dots)
        original_width, original_height = image.size
        image = transforms.Resize(384)(image)
        resize_width, resize_height = image.size
        assert resize_height == 384 and resize_width >= 384, "height must be equal to 384 and width must be larger than or equal to 384"
        height_scale = resize_height / original_height
        adjusted_annotations = [
            (int(x * height_scale), int(y * height_scale)) for x, y in dots
            ]
        dots = np.array(adjusted_annotations)

        density = cv2.resize(density, image.size)
        # density = transforms.Resize(384)(density)
        W, H = image.size

        new_H = 16*int(H/16)
        new_W = 16*int(W/16)
        scale_factor = float(new_W)/ W
        resized_image = transforms.Resize((new_H, new_W))(image)
        resized_density = cv2.resize(density, (new_W, new_H))   
        
        # Augmentation probability
        aug_p = random.random()
        aug_flag = 0
        mosaic_flag = 0
        if self.use_augmentation and aug_p < 0.5: # 0.4
            aug_flag = 1
            if aug_p < 0.3: # 0.25
                aug_flag = 0
                mosaic_flag = 1

        # Gaussian noise
        resized_image = TTensor(resized_image)
        if aug_flag == 1:
            noise = np.random.normal(0, 0.1, resized_image.size())
            noise = torch.from_numpy(noise)
            re_image = resized_image + noise
            re_image = torch.clamp(re_image, 0, 1)

        # Color jitter and Gaussian blur
        if aug_flag == 1:
            re_image = Augmentation(re_image)

        # Random affine
        if aug_flag == 1:
            re1_image = re_image.transpose(0,1).transpose(1,2).numpy()
            keypoints = []
            for i in range(dots.shape[0]):
                keypoints.append(Keypoint(x=min(new_W-1,int(dots[i][0]*scale_factor)), y=min(new_H-1,int(dots[i][1]))))
            kps = KeypointsOnImage(keypoints, re1_image.shape)

            seq = iaa.Sequential([
                iaa.Affine(
                    rotate=(-15,15),
                    scale=(0.8, 1.2),
                    shear=(-10,10),
                    translate_percent={"x": (-0.2,0.2), "y": (-0.2,0.2)},
                    mode=ia.ALL, 
                )
            ])
            re1_image, kps_aug = seq(image=re1_image, keypoints=kps)

            # Produce dot annotation map
            resized_density = np.zeros((resized_density.shape[0], resized_density.shape[1]),dtype='float32')
            for i in range(len(kps.keypoints)):
                if(int(kps_aug.keypoints[i].y)<= new_H-1 and int(kps_aug.keypoints[i].x)<=new_W-1) and not kps_aug.keypoints[i].is_out_of_image(re1_image):
                    resized_density[int(kps_aug.keypoints[i].y)][int(kps_aug.keypoints[i].x)]=1
            resized_density = torch.from_numpy(resized_density)

            re_image = TTensor(re1_image)

        # Random horizontal flip
        if aug_flag == 1:
            flip_p = random.random()
            if flip_p > 0.5:
                re_image = TF.hflip(re_image)
                resized_density = TF.hflip(resized_density)
        
        # Random vertical flip
        if aug_flag == 1:
            flip_p = random.random()
            if flip_p > 0.5:
                re_image = TF.vflip(re_image)
                resized_density = TF.vflip(resized_density)

        
        # Random 384*384 crop in a new_W*384 image and 384*new_W density map
        
        if mosaic_flag == 0:
            if aug_flag == 0:
                re_image = resized_image
                resized_density = np.zeros((resized_density.shape[0], resized_density.shape[1]),dtype='float32')
                for i in range(dots.shape[0]):
                    resized_density[min(new_H-1,int(dots[i][1]))][min(new_W-1,int(dots[i][0]*scale_factor))]=1
                resized_density = torch.from_numpy(resized_density)

            start = random.randint(0, new_W-1-383)
            reresized_image = TF.crop(re_image, 0, start, 384, 384)
            reresized_density = resized_density[:, start:start+384]
        # Random self mosaic
        else:
            image_array = []
            map_array = []
            blending_l = random.randint(10, 20)
            resize_l = 192 + 2 * blending_l
            if dots.shape[0] >= 70 or not self.use_out_mosaic: #! Dec 29: ??
                for i in range(4):
                    length =  random.randint(150, 384)
                    start_W = random.randint(0, new_W-length)
                    start_H = random.randint(0, new_H-length)
                    reresized_image1 = TF.crop(resized_image, start_H, start_W, length, length)
                    reresized_image1 = transforms.Resize((resize_l, resize_l))(reresized_image1)
                    reresized_image = Augmentation(reresized_image1)
                    reresized_density1 = np.zeros((resize_l,resize_l),dtype='float32')
                    for i in range(dots.shape[0]):
                        if min(new_H-1,int(dots[i][1])) >= start_H and min(new_H-1,int(dots[i][1])) < start_H + length and min(new_W-1,int(dots[i][0]*scale_factor)) >= start_W and min(new_W-1,int(dots[i][0]*scale_factor)) < start_W + length:
                            reresized_density1[min(resize_l-1,int((min(new_H-1,int(dots[i][1]))-start_H)*resize_l/length))][min(resize_l-1,int((min(new_W-1,int(dots[i][0]*scale_factor))-start_W)*resize_l/length))]=1
                    reresized_density1 = torch.from_numpy(reresized_density1)
                    image_array.append(reresized_image1)
                    map_array.append(reresized_density1)
            elif self.use_out_mosaic: #! Dec 29: else mosaic with other classes?
                m_flag = 1
                prob = random.random()
                if prob > 0.25:
                    gt_pos = random.randint(0,3)
                else:
                    gt_pos = random.randint(0,4) # 5% 0 objects
                for i in range(4):
                    if i == gt_pos:
                        Tim_id = im_id
                        r_image = resized_image
                        Tdots = dots
                        new_TH = new_H
                        new_TW = new_W
                        Tscale_factor = scale_factor
                    else:
                        Tim_id = self.dataset.idx_running_set[random.randint(0, len(self.dataset.idx_running_set)-1)]
                        Tdots = np.array(self.dataset.annotations[Tim_id]['points'])
                        '''while(abs(Tdots.shape[0]-dots.shape[0]<=10)):
                            Tim_id = train_set[random.randint(0, len(train_set)-1)]
                            Tdots = np.array(annotations[Tim_id]['points'])'''
                        Timage = Image.open('{}/{}'.format(self.dataset.im_dir, Tim_id))
                        Timage.load()
                        new_TH = 16*int(Timage.size[1]/16)
                        new_TW = 16*int(Timage.size[0]/16)
                        Tscale_factor = float(new_TW)/ Timage.size[0]
                        r_image = TTensor(transforms.Resize((new_TH, new_TW))(Timage))

                    length =  random.randint(250, 384)
                    start_W = random.randint(0, new_TW-length)
                    start_H = random.randint(0, new_TH-length)
                    r_image1 = TF.crop(r_image, start_H, start_W, length, length)
                    r_image1 = transforms.Resize((resize_l, resize_l))(r_image1)
                    r_density1 = np.zeros((resize_l,resize_l),dtype='float32')
                    if self.dataset.class_dict[im_id] == self.dataset.class_dict[Tim_id]:
                        for i in range(Tdots.shape[0]):
                            if min(new_TH-1,int(Tdots[i][1])) >= start_H and min(new_TH-1,int(Tdots[i][1])) < start_H + length and min(new_TW-1,int(Tdots[i][0]*Tscale_factor)) >= start_W and min(new_TW-1,int(Tdots[i][0]*Tscale_factor)) < start_W + length:
                                r_density1[min(resize_l-1,int((min(new_TH-1,int(Tdots[i][1]))-start_H)*resize_l/length))][min(resize_l-1,int((min(new_TW-1,int(Tdots[i][0]*Tscale_factor))-start_W)*resize_l/length))]=1
                    r_density1 = torch.from_numpy(r_density1)
                    image_array.append(r_image1)
                    map_array.append(r_density1)


            reresized_image5 = torch.cat((image_array[0][:,blending_l:resize_l-blending_l],image_array[1][:,blending_l:resize_l-blending_l]),1)
            reresized_density5 = torch.cat((map_array[0][blending_l:resize_l-blending_l],map_array[1][blending_l:resize_l-blending_l]),0)
            for i in range(blending_l):
                    reresized_image5[:,192+i] = image_array[0][:,resize_l-1-blending_l+i] * (blending_l-i)/(2*blending_l) + reresized_image5[:,192+i] * (i+blending_l)/(2*blending_l)
                    reresized_image5[:,191-i] = image_array[1][:,blending_l-i] * (blending_l-i)/(2*blending_l) + reresized_image5[:,191-i] * (i+blending_l)/(2*blending_l)
            reresized_image5 = torch.clamp(reresized_image5, 0, 1)

            reresized_image6 = torch.cat((image_array[2][:,blending_l:resize_l-blending_l],image_array[3][:,blending_l:resize_l-blending_l]),1)
            reresized_density6 = torch.cat((map_array[2][blending_l:resize_l-blending_l],map_array[3][blending_l:resize_l-blending_l]),0)
            for i in range(blending_l):
                    reresized_image6[:,192+i] = image_array[2][:,resize_l-1-blending_l+i] * (blending_l-i)/(2*blending_l) + reresized_image6[:,192+i] * (i+blending_l)/(2*blending_l)
                    reresized_image6[:,191-i] = image_array[3][:,blending_l-i] * (blending_l-i)/(2*blending_l) + reresized_image6[:,191-i] * (i+blending_l)/(2*blending_l)
            reresized_image6 = torch.clamp(reresized_image6, 0, 1)

            reresized_image = torch.cat((reresized_image5[:,:,blending_l:resize_l-blending_l],reresized_image6[:,:,blending_l:resize_l-blending_l]),2)
            reresized_density = torch.cat((reresized_density5[:,blending_l:resize_l-blending_l],reresized_density6[:,blending_l:resize_l-blending_l]),1)
            for i in range(blending_l):
                    reresized_image[:,:,192+i] = reresized_image5[:,:,resize_l-1-blending_l+i] * (blending_l-i)/(2*blending_l) + reresized_image[:,:,192+i] * (i+blending_l)/(2*blending_l)
                    reresized_image[:,:,191-i] = reresized_image6[:,:,blending_l-i] * (blending_l-i)/(2*blending_l) + reresized_image[:,:,191-i] * (i+blending_l)/(2*blending_l)
            reresized_image = torch.clamp(reresized_image, 0, 1)
        
        # Gaussian distribution density map
        reresized_density = ndimage.gaussian_filter(reresized_density.numpy(), sigma=(1, 1), order=0)

        # Density map scale up
        reresized_density = reresized_density * 60
        reresized_density = torch.from_numpy(reresized_density)
            


        
        # boxes shape [3,3,64,64], image shape [3,384,384], density shape[384,384]       
        sample = {'image':reresized_image, 'gt_density':reresized_density, 'm_flag': m_flag}

        return sample

PreTrainNormalize = transforms.Compose([   
        transforms.RandomResizedCrop(384, scale=(0.2, 1.0), interpolation=3), 
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        #transforms.Normalize(mean=IM_NORM_MEAN, std=IM_NORM_STD)
        ])

TTensor = transforms.Compose([   
        transforms.ToTensor(),
        ])

Augmentation = transforms.Compose([   
        transforms.ColorJitter(brightness=0.3, contrast=0.15, saturation=0.2, hue=0.2),
        transforms.GaussianBlur(kernel_size=(7,9))
        ])

Normalize = transforms.Compose([   
        transforms.ToTensor(),
        transforms.Normalize(mean=IM_NORM_MEAN, std=IM_NORM_STD)
        ])
