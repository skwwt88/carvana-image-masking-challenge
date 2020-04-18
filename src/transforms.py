from torchvision.transforms import Compose, Resize, ToTensor

augment_transform = Compose([
    Resize((959, 640)),
    ToTensor()
])

mask_transform = Compose([
    Resize((959, 640)),
    ToTensor()
])

if __name__ == '__main__':
    from PIL import Image
    import os
    img = Image.open('../input/train/0cdf5b5d0ce1_01.jpg')
    img_augmented = augment_transform(img)
    print(img_augmented.shape)

    mask = Image.open('../input/train_masks/0cdf5b5d0ce1_01_mask.gif')
    mask_transformed = mask_transform(mask)
    print(mask_transformed.shape)