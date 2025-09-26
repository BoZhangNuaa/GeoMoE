from timm.data import create_transform
from torchvision import transforms
from torchvision.transforms import InterpolationMode
from timm.data.constants import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD

def create_transform_classify(input_size, aa='rand-m9-mstd0.5-inc1', reprob=0.25, remode='pixel', recount=1, mean=IMAGENET_DEFAULT_MEAN, std=IMAGENET_DEFAULT_STD, split="train"):
    if split == "train":
        # this should always dispatch to transforms_imagenet_train
        transform = create_transform(
            input_size=input_size,
            is_training=True,
            auto_augment=aa,
            interpolation='bicubic',
            re_prob=reprob,
            re_mode=remode,
            re_count=recount,
            mean=mean,
            std=std,
        )
    else:
        t = []
        if input_size <= 224:
            crop_pct = 224 / 256
        else:
            crop_pct = 1.0
        size = int(input_size / crop_pct)
        t.append(
            # to maintain same ratio w.r.t. 224 images
            transforms.Resize(size, interpolation=InterpolationMode.BICUBIC),
        )
        t.append(transforms.CenterCrop(input_size))
        t.append(transforms.ToTensor())
        t.append(transforms.Normalize(mean, std))
        transform = transforms.Compose(t)
    return transform

NWPU = create_transform_classify
AID = create_transform_classify