import os
import glob

from torch.utils.data import Dataset
from PIL import Image
from typing import List


class CarDataset(Dataset):
    file_format = '{}.jpg'
    maskfile_format = '{}_mask.gif'
    imgs_dir = '../input/train'
    masks_dir = '../input/train_masks'

    @staticmethod
    def create(augment_transform:callable, mask_transform: callable) -> Dataset:
        ids = [os.path.splitext(file)[0] for file in os.listdir(CarDataset.imgs_dir)]
        return CarDataset(ids, augment_transform, mask_transform)


    def __init__(self, ids: List[str], augment_transform:callable, mask_transform: callable):
        self.ids = ids
        self.augment_transform = augment_transform
        self.mask_transform = mask_transform

    def __getitem__(self, index):
        id = self.ids[index]

        img = Image.open(os.path.join(self.imgs_dir, self.file_format.format(id)))
        mask = Image.open(os.path.join(self.masks_dir, self.maskfile_format.format(id)))

        return self.augment_transform(img), self.mask_transform(mask)

if __name__ == '__main__':
    from torchvision.transforms import Compose

    test_dataset = CarDataset.create(Compose([]), Compose([]))
    test_sample = test_dataset[10]
    print(test_sample[0].size, test_sample[1].size)
        

    