import torch
import torch.nn as nn
from transformers import ViTModel


# Класс модели
class VisualClassifier(nn.Module):
    def __init__(self, vision_model_name, num_classes, hidden_dim=1024):
        super(VisualClassifier, self).__init__()

        # Предобученная модель ViT
        self.vision_model = ViTModel.from_pretrained(vision_model_name)
        vision_config = self.vision_model.config

        # Классификатор (линейные слои поверх эмбеддингов)
        self.classifier = nn.Sequential(
            nn.Linear(vision_config.hidden_size, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, num_classes)
        )

    def forward(self, images):
        # Извлечение эмбеддингов из ViT
        with torch.no_grad():  # Заморозка параметров ViT
            embeddings = self.vision_model(pixel_values=images).pooler_output

        # Классификация на основе эмбеддингов
        logits = self.classifier(embeddings)
        return logits
