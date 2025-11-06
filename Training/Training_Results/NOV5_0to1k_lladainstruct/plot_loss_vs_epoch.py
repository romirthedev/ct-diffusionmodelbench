import os
import json
import argparse

import matplotlib.pyplot as plt


def load_metrics(json_path: str):
    with open(json_path, 'r') as f:
        data = json.load(f)
    # Expect a list of dicts with keys: epoch, loss
    epochs = []
    losses = []
    for item in data:
        if 'epoch' in item and 'loss' in item:
            epochs.append(float(item['epoch']))
            losses.append(float(item['loss']))
    if not epochs:
        raise ValueError("No epoch/loss entries found in JSON.")
    return epochs, losses


def plot_loss_vs_epoch(epochs, losses, out_path: str, title: str = None):
    plt.figure(figsize=(10, 6))
    plt.plot(epochs, losses, marker='o', linewidth=1.5, markersize=3, color='#1f77b4')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title(title or 'Loss vs Epoch')
    plt.grid(True, linestyle='--', alpha=0.4)
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    print(f"Saved plot: {out_path}")


def main():
    parser = argparse.ArgumentParser(description='Plot Loss vs Epoch from a training metrics JSON.')
    default_json = os.path.join(os.path.dirname(__file__), 'basemodel_0to1k.json')
    parser.add_argument('--json', type=str, default=default_json, help='Path to metrics JSON file')
    parser.add_argument('--out', type=str, default=None, help='Output image path (PNG). Defaults next to JSON.')
    parser.add_argument('--title', type=str, default='Loss vs Epoch (NOV5_0to1k_lladainstruct)', help='Plot title')
    args = parser.parse_args()

    epochs, losses = load_metrics(args.json)

    out_path = args.out
    if out_path is None:
        base_dir = os.path.dirname(args.json)
        out_path = os.path.join(base_dir, 'loss_vs_epoch.png')

    plot_loss_vs_epoch(epochs, losses, out_path, title=args.title)


if __name__ == '__main__':
    main()