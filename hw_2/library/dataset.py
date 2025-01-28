import os


from PIL import Image
from torch.utils.data import Dataset



class TrainDataset(Dataset):
    def __init__(self, image_folder, transform=None):
        self.image_folder = image_folder
        self.image_paths = [os.path.join(image_folder, f) for f in os.listdir(image_folder)]
        self.transform = transform

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        image = Image.open(img_path)
        if self.transform:
            image = self.transform(image)
        return image


class TestDataset(Dataset):
    def __init__(self, image_folder, annotation_file, transform=None):
        self.image_folder = image_folder
        self.transform = transform
        self.labels = {}
        with open(annotation_file, 'r') as f:
            for line in f.readlines():
                filename, label = line.strip().split()
                self.labels[filename] = int(label)
        self.image_names = list(self.labels.keys())

    def __len__(self):
        return len(self.image_names)

    def __getitem__(self, idx):
        img_name = self.image_names[idx]
        img_path = os.path.join(self.image_folder, img_name)
        image = Image.open(img_path)
        if self.transform:
            image = self.transform(image)
        return image, self.labels[img_name], img_name


class ImageDataset(Dataset):
    def __init__(self, image_folder, transform=None):
        self.images = [os.path.join(image_folder, img) for img in os.listdir(image_folder)]
        self.transform = transform

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img_path = self.images[idx]
        image = Image.open(img_path)
        if self.transform:
            image = self.transform(image)
        return image
