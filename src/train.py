import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
import yaml
import os
from dataset import SyntheticMorphologyDataset
from model import SimpleMorphCNN

# Load config
def load_config(config_path):
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)

def main():
    config = load_config('src/phase1_config.yaml')
    device = torch.device(config['train']['device'] if torch.cuda.is_available() else 'cpu')

    # Dataset
    dataset = SyntheticMorphologyDataset(
        data_dir=config['dataset']['data_dir'],
        metadata_csv=config['dataset']['metadata_csv'],
        augment=config['augment']['use_augment']
    )
    n_total = len(dataset)
    n_train = int(n_total * config['dataset']['train_split'])
    n_val = int(n_total * config['dataset']['val_split'])
    n_test = n_total - n_train - n_val
    train_set, val_set, test_set = random_split(dataset, [n_train, n_val, n_test])
    train_loader = DataLoader(train_set, batch_size=config['dataset']['batch_size'], shuffle=True, num_workers=config['dataset']['num_workers'])
    val_loader = DataLoader(val_set, batch_size=config['dataset']['batch_size'], shuffle=False, num_workers=config['dataset']['num_workers'])

    # Model
    model = SimpleMorphCNN(num_classes=config['model']['num_classes'], in_channels=config['model']['in_channels'])
    model = model.to(device)

    # Loss, optimizer, scheduler
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=config['train']['learning_rate'], weight_decay=config['train']['weight_decay'])
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=config['train']['step_size'], gamma=config['train']['gamma'])

    best_val_acc = 0.0
    for epoch in range(config['train']['epochs']):
        model.train()
        running_loss, correct, total = 0.0, 0, 0
        for imgs, labels in train_loader:
            imgs, labels = imgs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(imgs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * imgs.size(0)
            _, preds = torch.max(outputs, 1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)
        train_loss = running_loss / total
        train_acc = correct / total

        # Validation
        model.eval()
        val_loss, val_correct, val_total = 0.0, 0, 0
        with torch.no_grad():
            for imgs, labels in val_loader:
                imgs, labels = imgs.to(device), labels.to(device)
                outputs = model(imgs)
                loss = criterion(outputs, labels)
                val_loss += loss.item() * imgs.size(0)
                _, preds = torch.max(outputs, 1)
                val_correct += (preds == labels).sum().item()
                val_total += labels.size(0)
        val_loss = val_loss / val_total
        val_acc = val_correct / val_total

        print(f"Epoch {epoch+1}/{config['train']['epochs']} | Train Loss: {train_loss:.4f} Acc: {train_acc:.4f} | Val Loss: {val_loss:.4f} Acc: {val_acc:.4f}")

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), 'best_model.pt')
            print("Saved new best model.")
        scheduler.step()

if __name__ == "__main__":
    main() 