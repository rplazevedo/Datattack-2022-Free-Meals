# -*- coding: utf-8 -*-
"""
Created on Sun May 15 09:29:04 2022

@author: Rui Pedro Lopes de Azevedo
"""

import os
import torch
import torch.utils.data
import torchvision
from PIL import Image
from pycocotools.coco import COCO
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
import numpy as np
import random
import pandas as pd
from torchvision.io import read_image
from torchvision.transforms.functional import convert_image_dtype

from google.colab import drive
drive.mount('/content/drive')

class myOwnDataset(torch.utils.data.Dataset):
    def __init__(self, root, annotation, transforms=None):
        self.root = root
        self.transforms = transforms
        self.coco = COCO(annotation)
        self.ids = list(sorted(self.coco.imgs.keys()))

    def __getitem__(self, index):
        # Own coco file
        coco = self.coco
        # Image ID
        img_id = self.ids[index]
        # List: get annotation id from coco
        ann_ids = coco.getAnnIds(imgIds=img_id)
        # Dictionary: target coco_annotation file for an image
        coco_annotation = coco.loadAnns(ann_ids)
        # path for input image
        path = coco.loadImgs(img_id)[0]["file_name"]
        # open the input image
        img = Image.open(os.path.join(self.root, path))
        # number of objects in the image
        num_objs = len(coco_annotation)

        # Bounding boxes for objects
        # In coco format, bbox = [xmin, ymin, width, height]
        # In pytorch, the input should be [xmin, ymin, xmax, ymax]
        boxes = []
        for i in range(num_objs):
            xmin = coco_annotation[i]["bbox"][0]
            ymin = coco_annotation[i]["bbox"][1]
            xmax = xmin + coco_annotation[i]["bbox"][2]
            ymax = ymin + coco_annotation[i]["bbox"][3]
            boxes.append([xmin, ymin, xmax, ymax])
        boxes = torch.as_tensor(boxes, dtype=torch.float32)
        # Labels (In my case, I only one class: target class or background)
        # labels = torch.ones((num_objs,), dtype=torch.int64)
        labels = []
        for i in range(num_objs):
            label = coco_annotation[i]["category_id"]
            labels.append(label)
        labels = torch.as_tensor(labels, dtype=torch.int64)
        # Tensorise img_id
        img_id = torch.tensor([img_id])
        # Size of bbox (Rectangular)
        areas = []
        for i in range(num_objs):
            areas.append(coco_annotation[i]["area"])
        areas = torch.as_tensor(areas, dtype=torch.float32)
        # Iscrowd
        iscrowd = torch.zeros((num_objs,), dtype=torch.int64)

        # Annotation is in dictionary format
        my_annotation = {}
        my_annotation["boxes"] = boxes
        my_annotation["labels"] = labels
        my_annotation["image_id"] = img_id
        my_annotation["area"] = areas
        my_annotation["iscrowd"] = iscrowd

        if self.transforms is not None:
            img = self.transforms(img)

        return img, my_annotation

    def __len__(self):
        return len(self.ids)


# In my case, just added ToTensor
def get_transform():
    custom_transforms = []
    custom_transforms.append(torchvision.transforms.ToTensor())
    return torchvision.transforms.Compose(custom_transforms)


# collate_fn needs for batch
def collate_fn(batch):
    return tuple(zip(*batch))


def get_model_instance_segmentation(num_classes):
    # load an instance segmentation model pre-trained pre-trained on COCO
    model = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=True, progress=True)
    # get number of input features for the classifier
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    # replace the pre-trained head with a new one
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)

    return model

def printDistances(distances, token1Length, token2Length):
    for t1 in range(token1Length + 1):
        for t2 in range(token2Length + 1):
            print(int(distances[t1][t2]), end=" ")
        print()

def levenshteinDistanceDP(token1, token2):
    distances = np.zeros((len(token1) + 1, len(token2) + 1))

    for t1 in range(len(token1) + 1):
        distances[t1][0] = t1

    for t2 in range(len(token2) + 1):
        distances[0][t2] = t2
        
    a = 0
    b = 0
    c = 0
    
    for t1 in range(1, len(token1) + 1):
        for t2 in range(1, len(token2) + 1):
            if (token1[t1-1] == token2[t2-1]):
                distances[t1][t2] = distances[t1 - 1][t2 - 1]
            else:
                a = distances[t1][t2 - 1]
                b = distances[t1 - 1][t2]
                c = distances[t1 - 1][t2 - 1]
                
                if (a <= b and a <= c):
                    distances[t1][t2] = a + 1
                elif (b <= a and b <= c):
                    distances[t1][t2] = b + 1
                else:
                    distances[t1][t2] = c + 1

    #printDistances(distances, len(token1), len(token2))
    return distances[len(token1)][len(token2)]

def out_to_str(result, thresh=0.7):
    """
    Sort and order the predicted digits into one or two serial numbers.
    'result' is the dictionary containing the prediction dictionary with keys
             'boxes', 'labels' and 'scores'
    'thres' is the score value for rejecting false detections
    """

    def split(arr, cond):
      return [arr[cond], arr[~cond]]
    if 'scores' in result.keys():
        indices = result['scores'] > thresh
        boxes = np.array((result['boxes'][indices]).tolist())
        labels = np.array((result['labels'][indices]).tolist())-1
    else:
        boxes = np.array((result['boxes']).tolist())
        labels = np.array((result['labels']).tolist())-1

    if labels.shape[0] == 0:
      return "",""
    if len(boxes.shape) == 1:
      return str(labels[0]), ""

    y_sort = boxes[:, 1].argsort()
    box_sorted_asc = boxes[y_sort]
    labels_sorted_asc = labels[y_sort]
    avg_y = np.array([0.5*(box[1]+box[3]) for box in box_sorted_asc])

    is_first_row = avg_y<box_sorted_asc[0,3]
    fst_row, sec_row = split(labels_sorted_asc, is_first_row)
    fst_box, sec_box = split(box_sorted_asc, is_first_row)

    x_sort_first = fst_box[:, 0].argsort()
    first_label = fst_row[x_sort_first]
    row_1_str = "".join(map(str,first_label))

    x_sort_sec = sec_box[:, 0].argsort()
    sec_label = sec_row[x_sort_sec]
    row_2_str = "".join(map(str,sec_label))
    
    return row_1_str, row_2_str

def export_predictions(images_dir: str, team_name: str, *args) -> None:
    """
    Go through all images, get the predicted strings and export them to .csv

    Args:
        images_dir (str): the path to the images' folder
        team_name (str): the name of the team to include in the exported .csv file
    """
    
    predictions = []
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

    for image in os.listdir(images_dir):
        #print(images_dir+'/'+image)
        img_read = read_image(images_dir+'/'+image)
        img_read = img_read.to(device)
        batch = torch.stack([img_read])
        batch = convert_image_dtype(batch, dtype=torch.float)
        
        outputs = model(batch)
        # get predictions
        predicted_string_1, predicted_string_2 = out_to_str(outputs[0])
        
        # store predictions
        predictions.append({"image_name": image, 
                            "string_1_prediction": predicted_string_1, 
                            "string_2_prediction": predicted_string_2})

    # export predictions
    results_df = pd.DataFrame(predictions)
    results_df.to_csv(f"submissions/{team_name}.csv", index=False, header=True)

# path to your own data and coco file
train_data_dir = "/content/drive/Shareddrives/Datattack (Free Meals)/train/data"
train_coco = "/content/drive/Shareddrives/Datattack (Free Meals)/train/data/labels_train.json"

# Batch size
train_batch_size = 1

# Params for dataloader
train_shuffle_dl = True
num_workers_dl = 2

# Params for training

# Two classes; Only target class or background
num_classes = 11
num_epochs = 20

lr = 0.0001
momentum = 0.9
weight_decay = 0.005

def train():
    print("Torch version:", torch.__version__)

    # create own Dataset
    my_dataset = myOwnDataset(
        root=train_data_dir, annotation=train_coco, transforms=get_transform()
    )
    
    # split the data
    val_perc = 0.2
    dataset_size = len(my_dataset)
    val_size = int(val_perc * dataset_size)
    train_size = dataset_size - val_size
    train_data, val_data = torch.utils.data.random_split(my_dataset, [train_size, val_size])
    
    # own DataLoader
    train_loader = torch.utils.data.DataLoader(
        train_data,
        batch_size=train_batch_size,
        shuffle=train_shuffle_dl,
        num_workers=num_workers_dl,
        collate_fn=collate_fn,
    )
    
    val_loader = torch.utils.data.DataLoader(
        val_data,
        batch_size=train_batch_size,
        shuffle=train_shuffle_dl,
        num_workers=num_workers_dl,
        collate_fn=collate_fn,
    )
    
    all_data_loader = torch.utils.data.DataLoader(
        my_dataset,
        batch_size=train_batch_size,
        shuffle=train_shuffle_dl,
        num_workers=num_workers_dl,
        collate_fn=collate_fn,
    )
    
    
    # select device (whether GPU or CPU)
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    print(device)
    
    # DataLoader is iterable over Dataset
    for imgs, annotations in train_loader:
        imgs = list(img.to(device) for img in imgs)
        annotations = [{k: v.to(device) for k, v in t.items()} for t in annotations]
        #print(annotations)
    
    for imgs, annotations in val_loader:
        imgs = list(img.to(device) for img in imgs)
        annotations = [{k: v.to(device) for k, v in t.items()} for t in annotations]
        #print(annotations)
    
    
    model = get_model_instance_segmentation(num_classes)
    
    # move model to the right device
    model.to(device)
    
    # parameters
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.SGD(
        params, lr=lr, momentum=momentum, weight_decay=weight_decay
    )
    
    len_trainloader = len(train_loader)
    len_valloader = len(val_loader)
    
    model_name="resNet50_FPN_full"
    best_dist = 1000
    print(device)
    # Training
    for epoch in range(num_epochs):
        print(f"Epoch: {epoch}/{num_epochs}")
    
        print("Training Phase")
    
        model.train()
        
        for imgs, annotations in all_data_loader:
            imgs = list(img.to(device) for img in imgs)
            annotations = [{k: v.to(device) for k, v in t.items()} for t in annotations]
            loss_dict = model(imgs, annotations)
            losses = sum(loss for loss in loss_dict.values())
    
            optimizer.zero_grad()
            losses.backward()
            optimizer.step()
        
        print("Validation Phase")
    
        dist_list = []
    
        model.eval()
        preds = list()
        # Deactivate gradients
        with torch.no_grad():
    
          i=0
          for imgs, annotations in val_loader:
            i += 1
            imgs = list(img.to(device) for img in imgs)
            annotations = [{k: v.to(device) for k, v in t.items()} for t in annotations]
            preds = model(imgs)
    
            for pred in preds:
              row1_str, row2_str = out_to_str(pred)
            for annotation in annotations:
              gt_row1, gt_row2 = out_to_str(annotation)
            
            distance1 = levenshteinDistanceDP(row1_str,gt_row1)
            distance2 = levenshteinDistanceDP(row2_str,gt_row2)
            t_distance = distance1 + distance2
    
            dist_list.append(t_distance)
    
    
          run_dist = np.array(dist_list).mean()
          print(f"Iteration: {i}/{len_valloader}, Loss: {losses}, Distance Avg = {run_dist}") 
        # Save checkpoint
        if np.array(dist_list).mean()<best_dist:
          best_dist = run_dist
          weights_dir = os.path.join("/content/drive/Shareddrives/Datattack (Free Meals)/train/","trained_models", "weights")
          model_path = os.path.join(weights_dir, f"{model_name}_continental.pt")
          torch.save(model.state_dict(), model_path)
          print(f"Successfully saved at: {model_path}")
          
def main():
    # Results and Weights
    weights_dir = os.path.join("/content/drive/Shareddrives/Datattack (Free Meals)/train/","trained_models", "weights")
    if not os.path.isdir(weights_dir):
        os.makedirs(weights_dir)
    print(weights_dir)
    model_name="resNet50_FPN"
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

    model = get_model_instance_segmentation(num_classes)
    model_path = os.path.join(weights_dir, f"{model_name}_continental.pt")
    model.load_state_dict(torch.load(model_path, map_location = device))
    model.eval()

    TEAM_NAME = "FreeMeals"
    TEST_DIR = "dataset/test"

    export_predictions(TEAM_NAME, TEST_DIR)
    
if __name__ == '__main__':
    main()