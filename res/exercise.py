# 数据集自我练习
from torch.utils.data import Dataset
import cv2
from PIL import Image
import os

class MyData(Dataset):
    def __init__(self, root_dir, label_dir):
        self.root_dir = root_dir
        self.label_dir = label_dir
        self.path = os.path.join(root_dir,label_dir)
        self.images = os.listdir(self.path)

    def __getitem__(self, idx):
        img_name = self.images[idx]
        img_idx_path = os.path.join(self.root_dir,self.label_dir,img_name)
        # img = Image.open(img_idx_path)
        img = cv2.imread(img_idx_path)
        label = self.label_dir
        return img,label

    def __len__(self):
        return len(self.images)

root_dir = "../dataset/train"
ants_label_dir = "ants_image"

ants = MyData(root_dir,ants_label_dir)

img= ants[0][0]
cv2.imshow('show',img)
cv2.waitKey(0)



