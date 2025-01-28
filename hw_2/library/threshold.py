import numpy as np

from library.evaluator import Evaluator


class ThresholdOptimizer:
    def __init__(self, model, normal_loader, anomaly_loader, device):
        self.model = model
        self.normal_loader = normal_loader
        self.anomaly_loader = anomaly_loader
        self.device = device
        self.evaluator = Evaluator(model, device)
        
    def find_optimal_threshold(self):
        normal_errors = self.evaluator.compute_errors(self.normal_loader)
        anomaly_errors = self.evaluator.compute_errors(self.anomaly_loader)
        all_errors = np.concatenate([normal_errors, anomaly_errors])
        all_labels = np.concatenate([np.zeros_like(normal_errors), np.ones_like(anomaly_errors)])
        best_threshold = 0
        best_j = -1
        for threshold in np.percentile(all_errors, np.linspace(80, 99, 50)):
            preds = (all_errors > threshold).astype(int)
            tp = np.sum((preds == 1) & (all_labels == 1))
            tn = np.sum((preds == 0) & (all_labels == 0))
            fp = np.sum((preds == 1) & (all_labels == 0))
            fn = np.sum((preds == 0) & (all_labels == 1))
            tpr = tp / (tp + fn) if (tp + fn) > 0 else 0
            tnr = tn / (tn + fp) if (tn + fp) > 0 else 0
            j = tpr + tnr - 1
            if j > best_j:
                best_j = j
                best_threshold = threshold
        print(tpr, tnr)
        return best_threshold