import torch
import torch.nn as nn
from transformers import ViTModel


class AlexNetwork(nn.Module):
    def __init__(self):
        super(AlexNetwork, self).__init__()
        self.cnn = nn.Sequential(
            nn.Conv2d(3, 96, kernel_size=11, stride=4),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.LocalResponseNorm(96),
            nn.Conv2d(96, 384, kernel_size=5, stride=2, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.LocalResponseNorm(384),
            nn.Conv2d(384, 384, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(384),
            nn.Conv2d(384, 384, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(384),
            nn.Conv2d(384, 256, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(256),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
        )
        self.fc6 = nn.Sequential(
            nn.Linear(256 * 3 * 3, 4096),  # Corrected input size
            nn.ReLU(inplace=True),
            nn.BatchNorm1d(4096),
        )
        self.fc = nn.Sequential(
            nn.Linear(2 * 4096, 4096),
            nn.ReLU(inplace=True),
            nn.Linear(4096, 4096),
            nn.ReLU(inplace=True),
            nn.Linear(4096, 8),
        )

    def forward_once(self, x):
        output = self.cnn(x)
        output = output.view(output.size()[0], -1)
        output = self.fc6(output)
        return output

    def forward(self, input1, input2):
        output1 = self.forward_once(input1)
        output2 = self.forward_once(input2)
        output = torch.cat((output1, output2), 1)
        output = self.fc(output)
        return output


class SSLVisualClassifier(nn.Module):
    def __init__(self, vision_model_name, num_classes=8):
        super().__init__()
        self.vision_model = ViTModel.from_pretrained(vision_model_name)
        self.classifier = nn.Linear(
            self.vision_model.config.hidden_size * 2, num_classes
        )

    def forward(self, x1, x2):
        feat1 = self.vision_model(pixel_values=x1).pooler_output
        feat2 = self.vision_model(pixel_values=x2).pooler_output
        combined = torch.cat([feat1, feat2], dim=1)
        return self.classifier(combined)


class VisualClassifier(nn.Module):
    def __init__(
        self, model_name, num_classes, model_configs, hidden_dim=1024, use_ssl=False
    ):
        super().__init__()
        self.model_name = model_name
        config = model_configs[model_name]

        if model_name == "ViT":
            self.vision_model = ViTModel.from_pretrained(
                config["ssl_args"]["vision_model_name"]
            )
            if use_ssl:
                self.vision_model.load_state_dict(torch.load(config["weights_path"]))
            for param in self.vision_model.parameters():
                param.requires_grad = False
            feature_size = self.vision_model.config.hidden_size
        elif model_name == "AlexNet":
            self.cnn = AlexNetwork().cnn
            self.fc6 = AlexNetwork().fc6
            if use_ssl:
                alex_weights = torch.load(config["weights_path"])
                self.cnn.load_state_dict(
                    {
                        k.replace("cnn.", ""): v
                        for k, v in alex_weights.items()
                        if k.startswith("cnn")
                    }
                )
                self.fc6.load_state_dict(
                    {
                        k.replace("fc6.", ""): v
                        for k, v in alex_weights.items()
                        if k.startswith("fc6")
                    }
                )
            for param in self.cnn.parameters():
                param.requires_grad = False
            for param in self.fc6.parameters():
                param.requires_grad = False
            feature_size = config["feature_size"]

        self.classifier = nn.Sequential(
            nn.Linear(feature_size, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, num_classes),
        )

    def forward(self, images):
        if self.model_name == "ViT":
            with torch.no_grad():
                features = self.vision_model(pixel_values=images).pooler_output
        elif self.model_name == "AlexNet":
            features = self.cnn(images)
            features = features.view(features.size(0), -1)
            features = self.fc6(features)
        return self.classifier(features)


class ContextPredictionViT(nn.Module):
    def __init__(self, vision_model_name, num_positions=8):
        super().__init__()
        self.vit = ViTModel.from_pretrained(vision_model_name)
        self.num_positions = num_positions

        self.patch_encoder = nn.Sequential(
            nn.Linear(self.vit.config.hidden_size, 1024),
            nn.ReLU(),
            nn.Linear(1024, 512),
        )

        self.position_classifier = nn.Sequential(
            nn.Linear(1024, 512), nn.ReLU(), nn.Linear(512, num_positions)
        )

    def forward(self, patch1, patch2):
        emb1 = self._encode_patch(patch1)
        emb2 = self._encode_patch(patch2)

        combined = torch.cat([emb1, emb2], dim=1)
        return self.position_classifier(combined)

    def _encode_patch(self, x):
        outputs = self.vit(pixel_values=x)
        return self.patch_encoder(outputs.pooler_output)
