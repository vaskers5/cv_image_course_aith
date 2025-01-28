import torch
from torchvision import transforms
import numpy as np


def generate_patch_pairs(image, patch_ratio=0.2, gap_ratio=0.1, resize_size=224):
    _, h, w = image.shape
    patch_size = int(min(h, w) * patch_ratio)
    gap = int(min(h, w) * gap_ratio)

    patch_locations = [(1, 1), (1, 2), (1, 3), (2, 1), (2, 3), (3, 1), (3, 2), (3, 3)]

    offset_x = np.random.randint(0, h - (3 * patch_size + 2 * gap))
    offset_y = np.random.randint(0, w - (3 * patch_size + 2 * gap))

    center_x = offset_x + patch_size + gap
    center_y = offset_y + patch_size + gap
    patch1 = image[
        :, center_x : center_x + patch_size, center_y : center_y + patch_size
    ]

    label = np.random.randint(0, 8)
    dx, dy = patch_locations[label]
    patch2_x = offset_x + (dx - 1) * (patch_size + gap)
    patch2_y = offset_y + (dy - 1) * (patch_size + gap)
    patch2 = image[
        :, patch2_x : patch2_x + patch_size, patch2_y : patch2_y + patch_size
    ]

    resize_transform = transforms.Resize((resize_size, resize_size))
    patch1 = resize_transform(patch1.unsqueeze(0)).squeeze(0)
    patch2 = resize_transform(patch2.unsqueeze(0)).squeeze(0)

    return patch1, patch2, label


class SSLDataset(torch.utils.data.Dataset):
    def __init__(self, dataset, patch_ratio=0.2, gap_ratio=0.1):
        self.dataset = dataset
        self.patch_ratio = patch_ratio
        self.gap_ratio = gap_ratio

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        image, _ = self.dataset[idx]
        patch1, patch2, label = generate_patch_pairs(
            image, self.patch_ratio, self.gap_ratio
        )
        return patch1, patch2, label
