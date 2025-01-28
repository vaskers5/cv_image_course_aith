import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import vgg16


class PerceptualLoss(nn.Module):
    def __init__(self, device):
        super().__init__()
        self.device = device
        self.vgg = vgg16(pretrained=True).features[:16].eval().to(device)
        for param in self.vgg.parameters():
            param.requires_grad = False
        
        # Нормализация для VGG
        self.mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1).to(device)
        self.std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1).to(device)
            
    def forward(self, inputs, reconstructions):
        # Конвертация из [-1,1] в VGG-формат
        inputs_vgg = (inputs + 1) / 2  # Диапазон [0,1]
        inputs_vgg = (inputs_vgg - self.mean) / self.std
        
        recons_vgg = (reconstructions + 1) / 2
        recons_vgg = (recons_vgg - self.mean) / self.std
        
        vgg_inputs = self.vgg(inputs_vgg)
        vgg_recons = self.vgg(recons_vgg)
        return F.l1_loss(vgg_recons, vgg_inputs)