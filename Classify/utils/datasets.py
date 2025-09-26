import json
import torch
import PIL.Image as Image


class NWPUDataset(torch.utils.data.Dataset):
    def __init__(self, root_dir, transform, split, tr=None):
        super().__init__()
        self.transform = transform
        self.root_dir = root_dir
        if tr is not None:
            tr = f"{root_dir}/{tr}"
        else:
            tr = f"{root_dir}/anno.json"
        with open(tr, 'r') as f:
            datainfo = json.load(f)
            self.num_classes = len(datainfo['categories'])
            datainfo = datainfo[split]

        self.img_list = datainfo

    def __len__(self):
        return len(self.img_list)

    def __getitem__(self, idx):
        img_path = f"{self.root_dir}/{self.img_list[idx]['image']}"
        label = self.img_list[idx]['label']
        image = Image.open(img_path).convert('RGB')

        return self.transform(image), label


def nwpu(root_dir, transform, split, tr=None):
    dataset = NWPUDataset(root_dir, transform, split, tr=tr)
    return dataset


NWPU = nwpu
AID = nwpu
