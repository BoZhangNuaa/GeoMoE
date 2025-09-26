import os
from PIL import Image
import torch
from torchvision import transforms
import numpy as np


class AllDataset(torch.utils.data.Dataset):
    def __init__(self, root, train=True, transform=None, split=None, tag=None):

        train_files = []
        train_targets = []
        for root, dirs, files in os.walk(root):
            for filename in dirs:
                file_dir = os.path.join(root, filename)
                for dirpath, dirnames, filenames in os.walk(file_dir):
                    for filename in filenames:
                        image_path = os.path.join(file_dir,  filename)
                        train_files.append(image_path)
                        train_targets.append(int(1))

        self.files = train_files
        self.targets = train_targets
        self.transform = transform
        self.norm = transforms.Normalize(
            mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        print('Creating All dataset with {} examples'.format(len(self.targets)))

    def __len__(self):
        return len(self.targets)

    def __getitem__(self, i):
        img_path = self.files[i]
        try:
            img = Image.open(img_path).convert('RGB')
            if self.transform != None:
                img = self.transform(img)
                if img.shape[0] == 1:
                    img = img.float()
                    img = np.stack((img,) * 3, axis=0)
                    img = torch.tensor(img).squeeze()
                elif img.shape[0] != 1 and img.shape[0] != 3:
                    return self.__getitem__(i+1)
        except:
            return self.__getitem__(i+1)

        # return img, self.targets[i]
        return img


def get_OpticalRS(root_dir, trainsform=None):
    return AllDataset(root_dir, trainsform)


OpticalRS = get_OpticalRS
