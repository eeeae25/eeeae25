from torch.utils.data import Dataset
from data_aug1 import paired_patch_augmentation
import os
import random
from PIL import Image
import torchvision.transforms as T
import torch

class InpaintingAugDataset(Dataset):
    def __init__(self, image_dir, mask_dir, patch_size=64):
        self.image_paths = sorted([os.path.join(image_dir, f) for f in os.listdir(image_dir)])
        self.mask_paths = sorted([os.path.join(mask_dir, f) for f in os.listdir(mask_dir)])
        self.patch_size = patch_size
        self.transform = T.Compose([
            T.Resize((256, 256)),
            T.ToTensor()
        ])

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        indices = random.sample(range(len(self.image_paths)), 4)
        imgs, masks = [], []
        for i in indices:
            img = self.transform(Image.open(self.image_paths[i]).convert('RGB'))
            mask = self.transform(Image.open(self.mask_paths[i]).convert('L')).unsqueeze(0)
            imgs.append(img)
            masks.append(mask)

        aug_img, aug_mask = paired_patch_augmentation(imgs, masks, self.patch_size)
        input_tensor = torch.cat([aug_img * (1 - aug_mask), aug_mask], dim=0)
        return input_tensor, aug_img
