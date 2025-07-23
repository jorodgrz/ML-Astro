import torch
from torch.utils.data import DataLoader, random_split
from dataset import SyntheticMorphologyDataset

def load_dataset(config):
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
    test_loader = DataLoader(test_set, batch_size=config['dataset']['batch_size'], shuffle=False, num_workers=config['dataset']['num_workers'])
    return train_loader, val_loader, test_loader 