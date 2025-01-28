import numpy as np
import torch
import torch.nn as nn


class Evaluator:
    def __init__(self, model, device):
        self.model = model
        self.device = device
        self.reconstruction_criterion = nn.MSELoss(reduction='none')

    def compute_errors(self, data_loader):
        self.model.eval()
        errors = []
        with torch.no_grad():
            for batch in data_loader:
                if isinstance(batch, (list, tuple)):
                    inputs = batch[0].to(self.device)
                else:
                    inputs = batch.to(self.device)
                outputs, _ = self.model(inputs)
                mse_per_pixel = self.reconstruction_criterion(outputs, inputs)
                mse_per_image = torch.mean(mse_per_pixel, dim=[1, 2, 3])
                errors.extend(mse_per_image.cpu().numpy().tolist())
        return np.array(errors)

    def determine_threshold(self, normal_errors, percentile=95):
        return np.percentile(normal_errors, percentile)

    def evaluate(self, test_loader, threshold):
        self.model.eval()
        predictions, true_labels = [], []
        with torch.no_grad():
            for images, labels, _ in test_loader:
                images = images.to(self.device)
                outputs, _ = self.model(images)
                mse_per_pixel = self.reconstruction_criterion(outputs, images)
                mse_per_image = torch.mean(mse_per_pixel, dim=[1, 2, 3])
                batch_preds = (mse_per_image.cpu().numpy() > threshold).astype(int)
                predictions.extend(batch_preds)
                true_labels.extend(labels.numpy())
        true_labels = np.array(true_labels)
        predictions = np.array(predictions)
        tp = np.sum((predictions == 1) & (true_labels == 1))
        tn = np.sum((predictions == 0) & (true_labels == 0))
        fp = np.sum((predictions == 1) & (true_labels == 0))
        fn = np.sum((predictions == 0) & (true_labels == 1))
        tpr = tp / (tp + fn) if (tp + fn) > 0 else 0
        tnr = tn / (tn + fp) if (tn + fp) > 0 else 0
        return tpr, tnr
