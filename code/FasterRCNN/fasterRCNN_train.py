'''
This file holds the code to fine tune a pretrained FasterRCNN model.
'''
import os
import random
import numpy as np
import pandas as pd
import warnings
warnings.filterwarnings('ignore')
import cv2
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import torch
import torchvision
from torchvision import transforms as torchtrans  
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
import transforms as T
import utils
import xml.etree.ElementTree as ET
from tqdm import tqdm
import random
import shutil
from skimage.morphology import square
from skimage.morphology import closing
import albumentations as A
from albumentations.pytorch.transforms import ToTensorV2
from coco_eval import CocoEvaluator
from coco_utils import get_coco_api_from_dataset
from engine import train_one_epoch, evaluate
import time

class SpaceInvadersDataset(torch.utils.data.Dataset):

    def __init__(self, files_dir, width, height, transforms=None):
        self.transforms = transforms
        self.files_dir = files_dir
        self.height = height
        self.width = width
        
        # sorting the images for consistency
        # To get images, the extension of the filename is checked to be png
        self.imgs = [image for image in sorted(os.listdir(files_dir))
                        if image[-4:]=='.png']
        
        
        # classes: 0 index is reserved for background
        self.classes = ['none', 'player','score', 'alien', 'shield', 'satellite', 'bullet', 'lives']

    def __getitem__(self, idx):

        img_name = self.imgs[idx]
        image_path = os.path.join(self.files_dir, img_name)
        print(image_path)

        # reading the images and converting them to correct size and color    
        img = cv2.imread(image_path)
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB).astype(np.float32)
        img_res = cv2.resize(img_rgb, (self.width, self.height), cv2.INTER_AREA)
        # diving by 255
        img_res /= 255.0
        
        # annotation file
        annot_filename = img_name[:-4] + '.xml'
        annot_file_path = os.path.join(self.files_dir, annot_filename)
        print(annot_file_path)
        
        boxes = []
        labels = []
        tree = ET.parse(annot_file_path)
        root = tree.getroot()
        
        # cv2 image gives size as height x width
        wt = img.shape[1]
        ht = img.shape[0]
        
        # box coordinates for xml files are extracted and corrected for image size given
        for member in root.findall('object'):
            labels.append(self.classes.index(member.find('name').text))
            
            # bounding box
            xmin = int(member.find('bndbox').find('xmin').text)
            xmax = int(member.find('bndbox').find('xmax').text)
            
            ymin = int(member.find('bndbox').find('ymin').text)
            ymax = int(member.find('bndbox').find('ymax').text)
            
            
            xmin_corr = (xmin/wt)*self.width
            xmax_corr = (xmax/wt)*self.width
            ymin_corr = (ymin/ht)*self.height
            ymax_corr = (ymax/ht)*self.height
            
            boxes.append([xmin_corr, ymin_corr, xmax_corr, ymax_corr])
        
        # convert boxes into a torch.Tensor
        boxes = torch.as_tensor(boxes, dtype=torch.float32)
        
        # getting the areas of the boxes
        area = (boxes[:, 3] - boxes[:, 1]) * (boxes[:, 2] - boxes[:, 0])

        # suppose all instances are not crowd
        iscrowd = torch.zeros((boxes.shape[0],), dtype=torch.int64)
        
        labels = torch.as_tensor(labels, dtype=torch.int64)


        target = {}
        target["boxes"] = boxes
        target["labels"] = labels
        target["area"] = area
        target["iscrowd"] = iscrowd
        #image_id = torch.tensor([idx])
        #target["image_id"] = image_id
        #target["image_id"] = torch.tensor([idx]).to(device) if isinstance(idx, int) else idx
        target["image_id"] = idx


        if self.transforms:
            
            sample = self.transforms(image = img_res,
                                     bboxes = target['boxes'],
                                     labels = labels)
            
            img_res = sample['image']
            target['boxes'] = torch.Tensor(sample['bboxes'])
            
            
            
        return img_res, target

    def __len__(self):
        return len(self.imgs)
    
    # Function to visualize bounding boxes in the image

    def plot_img_bbox(self, img, target):
        # plot the image and bboxes
        # Bounding boxes are defined as follows: x-min y-min width height
        fig, a = plt.subplots(1,1)
        fig.set_size_inches(5,5)
        a.imshow(img)
        for box in (target['boxes']):
            x, y, width, height  = box[0], box[1], box[2]-box[0], box[3]-box[1]
            rect = patches.Rectangle((x, y),
                                    width, height,
                                    linewidth = 2,
                                    edgecolor = 'r',
                                    facecolor = 'none')

            # Draw the bounding box on top of the image
            a.add_patch(rect)
        plt.show()


   
# To load the model    
def get_object_detection_model(num_classes, model_path = None):

    model = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=True)
    
    # get number of input features for the classifier
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    # replace the pre-trained head with a new one
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
    
    # Load saved weights if a model_path is provided
    if model_path:
        loaded_model_state = torch.load(model_path)
        model.load_state_dict(loaded_model_state)

    return model
    
# Augumentations
def get_transform(train):
    if train:
        return A.Compose([
                            A.HorizontalFlip(0.5),
                    # ToTensorV2 converts image to pytorch tensor without div by 255
                            ToTensorV2(p=1.0) 
                        ], bbox_params={'format': 'pascal_voc', 'label_fields': ['labels']})
    else:
        return A.Compose([
                            ToTensorV2(p=1.0)
                        ], bbox_params={'format': 'pascal_voc', 'label_fields': ['labels']})

def collate_fn(batch):
    return tuple(zip(*batch))

def train_model(model, optimizer, lr_scheduler, data_loader, device, num_epochs):
    model.train()

    for epoch in range(num_epochs):
        pbar = tqdm(data_loader, desc=f'Epoch {epoch}')
        total_loss = []
        for images, targets in pbar:
            images = [image.to(device) for image in images]
            targets = [{k: v.to(device) for k, v in t.items() if not 'image_name' in k} for t in targets]

            loss_dict = model(images, targets)
            losses = sum(loss for loss in loss_dict.values())
            total_loss.append(losses)
            pbar.set_description(f"Epoch {epoch} - loss {losses:.3f}")

            optimizer.zero_grad()
            losses.backward()
            optimizer.step()

        avg_loss = torch.stack(total_loss).mean().cpu().detach().numpy()
        lr_scheduler.step()
        print(f'Epoch {epoch} - average loss: {avg_loss:.3f} - new learning rate: {optimizer.param_groups[0]["lr"]}')

    torch.save(model.state_dict(), 'fasterRCNN-SPI-server.pth')

def apply_nms(orig_prediction, iou_thresh=0.2):
    
    # torchvision returns the indices of the bboxes to keep
    keep = torchvision.ops.nms(orig_prediction['boxes'], orig_prediction['scores'], iou_thresh)
    
    final_prediction = orig_prediction
    final_prediction['boxes'] = final_prediction['boxes'][keep]
    final_prediction['scores'] = final_prediction['scores'][keep]
    final_prediction['labels'] = final_prediction['labels'][keep]
    
    return final_prediction
    
    

if __name__ == '__main__':
    
    # provide the input directory path for train and test data
    files_dir = r'/scratch/users/sundararaj/msc2023_jayakumar/Dataset/SpaceInvaders/train'
    test_dir = r'/scratch/users/sundararaj/msc2023_jayakumar/Dataset/SpaceInvaders/test'
    
    # use our dataset and defined transformations
    dataset = SpaceInvadersDataset(files_dir, width=160, height=210, transforms= get_transform(train=True))
    dataset_test = SpaceInvadersDataset(files_dir, width=160, height=210, transforms=get_transform(train=False))

    # split the dataset in train and test set
    torch.manual_seed(1)
    indices = torch.randperm(len(dataset)).tolist()

    # train test split
    test_split = 0.2
    tsize = int(len(dataset)*test_split)
    dataset = torch.utils.data.Subset(dataset, indices[:-tsize])
    dataset_test = torch.utils.data.Subset(dataset_test, indices[-tsize:])

    # define training and validation data loaders
    data_loader = torch.utils.data.DataLoader(
        dataset, batch_size=8, shuffle=True, num_workers=0,
        collate_fn=collate_fn)

    data_loader_test = torch.utils.data.DataLoader(
        dataset_test, batch_size=8, shuffle=False, num_workers=0,
        collate_fn=collate_fn)
    
    # to train on gpu if selected.
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

    # get the model using our helper function
    num_classes = 8
    model_path = r'/scratch/users/sundararaj/msc2023_jayakumar/fasterRCNN/fasterRCNN-SPI-10.pth'
    model = get_object_detection_model(num_classes, model_path)
    model.to(device)

    # construct an optimizer
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.SGD(params, lr=0.005,
                                momentum=0.9, weight_decay=0.0005)

    # and a learning rate scheduler which decreases the learning rate by
    # 10x every 3 epochs
    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer,
                                                step_size=3,
                                                gamma=0.1)
    
    # Training for mentioned epochs
    num_epochs = 10
    
    print('reached upto train function')
    train_model(model, optimizer, lr_scheduler, data_loader, device, num_epochs)
    # evaluate(model, data_loader, device)