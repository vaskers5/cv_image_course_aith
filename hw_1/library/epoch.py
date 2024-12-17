import torch
from tqdm.auto import tqdm

from library.metrics import compute_metrics

# Обучение
def train_epoch(model, loader, optimizer, criterion, device):
    model.train()
    total_loss = 0
    for images, labels in tqdm(loader):
        images, labels = images.to(device), labels.to(device)

        logits = model(images)
        loss = criterion(logits, labels)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
    return total_loss / len(loader)


# Тестирование
def evaluate_model(model, loader, device, num_classes):
    model.eval()
    all_predictions, all_targets = [], []
    with torch.no_grad():
        for images, labels in tqdm(loader):
            images, labels = images.to(device), labels.to(device)

            logits = model(images)
            predictions = torch.argmax(logits, dim=1)

            all_predictions.append(predictions)
            all_targets.append(labels)

    all_predictions = torch.cat(all_predictions)
    all_targets = torch.cat(all_targets)
    return compute_metrics(all_predictions, all_targets, num_classes)
