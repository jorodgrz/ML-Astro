import torch
import torch.nn.functional as F
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.manifold import TSNE
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import argparse
import os
import yaml
from model import SimpleMorphCNN
from data_loader import load_dataset

MORPH_CLASSES = ['spiral', 'shell', 'outflow', 'irregular', 'no_gas']

def load_config(config_path):
    """
    Load experiment configuration from a YAML file.
    Args:
        config_path (str): Path to the YAML config file.
    Returns:
        dict: Configuration dictionary.
    """
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)

def evaluate(model, dataloader, class_names, device):
    """
    Evaluate the model on the test set, print metrics, and plot/save the confusion matrix.
    Args:
        model (nn.Module): Trained model.
        dataloader (DataLoader): Test set DataLoader.
        class_names (list): List of class names.
        device (torch.device): Device to run evaluation on.
    Returns:
        np.ndarray: Latent features from the model.
        np.ndarray: True labels.
    """
    model.eval()
    all_preds, all_labels, all_feats = [], [], []
    with torch.no_grad():
        for images, labels in dataloader:
            images, labels = images.to(device), labels.to(device)
            outputs, features = model(images, return_features=True)
            preds = torch.argmax(outputs, dim=1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            all_feats.extend(features.cpu().numpy())
    # Print classification report
    print("\n📊 Classification Report:")
    print(classification_report(all_labels, all_preds, target_names=class_names, digits=4))
    # Confusion matrix
    cm = confusion_matrix(all_labels, all_preds)
    os.makedirs("results", exist_ok=True)
    plt.figure(figsize=(7,6))
    sns.heatmap(cm, annot=True, fmt="d", xticklabels=class_names, yticklabels=class_names, cmap="Blues", cbar=True, annot_kws={"size":16})
    plt.title("Confusion Matrix", fontsize=18)
    plt.xlabel("Predicted Label", fontsize=16)
    plt.ylabel("True Label", fontsize=16)
    plt.xticks(fontsize=14)
    plt.yticks(fontsize=14)
    plt.tight_layout()
    plt.savefig("results/confusion_matrix.svg", format="svg")  # Vector format for publication
    plt.savefig("results/confusion_matrix.png", dpi=300)
    plt.show()
    return np.array(all_feats), np.array(all_labels)

def plot_tsne(features, labels, class_names):
    """
    Compute and plot t-SNE projection of latent features, colored by class.
    Args:
        features (np.ndarray): Latent features from the model.
        labels (np.ndarray): True class labels.
        class_names (list): List of class names.
    """
    tsne = TSNE(n_components=2, perplexity=30, random_state=42)
    X_embedded = tsne.fit_transform(features)
    plt.figure(figsize=(9, 7))
    palette = sns.color_palette("deep", n_colors=len(class_names))
    sns.scatterplot(x=X_embedded[:, 0], y=X_embedded[:, 1], hue=[class_names[l] for l in labels], palette=palette, s=60, edgecolor="k", alpha=0.85)
    plt.title("t-SNE Projection of Latent Features", fontsize=18)
    plt.xlabel("t-SNE 1", fontsize=16)
    plt.ylabel("t-SNE 2", fontsize=16)
    plt.legend(title="Class", fontsize=13, title_fontsize=14, loc="best")
    plt.tight_layout()
    plt.savefig("results/tsne_projection.svg", format="svg")  # Vector format for publication
    plt.savefig("results/tsne_projection.png", dpi=300)
    plt.show()

def main():
    """
    Main function to run evaluation: loads model, test set, computes metrics, and plots results.
    """
    parser = argparse.ArgumentParser(description="Evaluate trained CNN on synthetic JWST NIRCam morphologies.")
    parser.add_argument("--weights", type=str, default="best_model.pt", help="Path to model weights")
    parser.add_argument("--config", type=str, default="src/phase1_config.yaml", help="Path to config file")
    args = parser.parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"🖥️  Using device: {device}")
    config = load_config(args.config)
    model = SimpleMorphCNN(num_classes=len(MORPH_CLASSES), in_channels=config['model']['in_channels'])
    model.load_state_dict(torch.load(args.weights, map_location=device))
    model.to(device)
    _, _, test_loader = load_dataset(config)
    features, labels = evaluate(model, test_loader, MORPH_CLASSES, device)
    plot_tsne(features, labels, MORPH_CLASSES)

if __name__ == "__main__":
    main() 