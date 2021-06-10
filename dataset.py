import os

import torch
import torchvision.transforms.functional as TF
from PIL import Image

class Font(torch.utils.data.Dataset):
    def __init__(self, data_path):
        self.datas = []
        self.labels = []
        self.data_path = data_path
        self.preprocess()

    def preprocess(self):
        # pick = [72 + x for x in range(1)]
        pick = [87 + x for x in range(5)]
        for class_folder in os.listdir(self.data_path):
            if class_folder[:5] != 'Class':
                continue
            label = int(class_folder.split('_')[1])
            if label not in pick:
                continue
            folder_path = os.path.join(self.data_path, class_folder)
            for text_image_file in os.listdir(folder_path):
                self.datas.append(os.path.join(folder_path, text_image_file))
                self.labels.append(label)

    def __len__(self):
        return len(self.datas)

    def __getitem__(self, index):
        img = Image.open(self.datas[index]).convert('L')
        img = TF.to_tensor(img)
        label = self.labels[index]
        return img, label
