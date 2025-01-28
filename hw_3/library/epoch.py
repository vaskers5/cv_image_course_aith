import torch
from tqdm.auto import tqdm
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

from library.metrics import compute_metrics


def train_epoch(model, loader, optimizer, criterion, device):
    model.train()
    total_loss = 0
    for images, labels in tqdm(loader, leave=False):
        images, labels = images.to(device), labels.to(device)

        logits = model(images)
        loss = criterion(logits, labels)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
    return total_loss / len(loader)


def evaluate_model(model, loader, device, num_classes):
    model.eval()
    all_predictions, all_targets = [], []
    with torch.no_grad():
        for images, labels in tqdm(loader, leave=False):
            images, labels = images.to(device), labels.to(device)

            logits = model(images)
            predictions = torch.argmax(logits, dim=1)

            all_predictions.append(predictions)
            all_targets.append(labels)

    all_predictions = torch.cat(all_predictions)
    all_targets = torch.cat(all_targets)
    return compute_metrics(all_predictions, all_targets, num_classes)


def train_ssl_epoch(model, loader, optimizer, criterion, device, run, epoch):
    model.train()
    total_loss = 0
    all_preds, all_labels = [], []

    for patch1, patch2, labels in tqdm(loader, leave=False):
        patch1, patch2, labels = patch1.to(device), patch2.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(patch1, patch2)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        preds = torch.argmax(outputs, dim=1)
        all_preds.append(preds.cpu())
        all_labels.append(labels.cpu())

    all_preds = torch.cat(all_preds).numpy()
    all_labels = torch.cat(all_labels).numpy()

    metrics = {
        "loss": total_loss / len(loader),
        "accuracy": accuracy_score(all_labels, all_preds),
        "precision": precision_score(all_labels, all_preds, average="weighted"),
        "recall": recall_score(all_labels, all_preds, average="weighted"),
        "f1": f1_score(all_labels, all_preds, average="weighted"),
    }

    for name, value in metrics.items():
        run.track(value, name=name, epoch=epoch, context={"phase": "ssl_train"})

    return metrics
