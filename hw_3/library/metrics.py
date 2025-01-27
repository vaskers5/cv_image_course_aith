from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score


def compute_metrics(predictions, targets, num_classes):
    predictions = predictions.cpu().numpy()
    targets = targets.cpu().numpy()

    metrics = {
        "accuracy": accuracy_score(targets, predictions),
        "precision_weighted": precision_score(
            targets, predictions, average="weighted", zero_division=0
        ),
        "recall_weighted": recall_score(
            targets, predictions, average="weighted", zero_division=0
        ),
        "f1_score_weighted": f1_score(
            targets, predictions, average="weighted", zero_division=0
        ),
    }

    # Метрики по каждому классу
    per_class_precision = precision_score(
        targets, predictions, average=None, zero_division=0, labels=range(num_classes)
    )
    per_class_recall = recall_score(
        targets, predictions, average=None, zero_division=0, labels=range(num_classes)
    )
    per_class_f1 = f1_score(
        targets, predictions, average=None, zero_division=0, labels=range(num_classes)
    )

    for class_idx in range(num_classes):
        metrics[f"precision_class_{class_idx}"] = per_class_precision[class_idx]
        metrics[f"recall_class_{class_idx}"] = per_class_recall[class_idx]
        metrics[f"f1_score_class_{class_idx}"] = per_class_f1[class_idx]

    return metrics
