import json
import matplotlib.pyplot as plt
import os

def plot_training_loss(json_file_path, output_dir, start_epoch=0):
    with open(json_file_path, 'r') as f:
        logs = json.load(f)

    epochs = []
    losses = []

    for log in logs:
        if "loss" in log and "epoch" in log and "eval_loss" not in log:
            if log["epoch"] >= start_epoch:
                epochs.append(log["epoch"])
                losses.append(log["loss"])

    if not epochs or not losses:
        print(f"No training loss data found from epoch {start_epoch} onwards in the JSON file.")
        return

    plt.figure(figsize=(10, 6))
    plt.plot(epochs, losses, label='Training Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title(f'Training Loss Over Epochs (from Epoch {start_epoch})')
    plt.grid(True)
    plt.legend()

    output_path = os.path.join(output_dir, f"epoch_vs_loss_from_{str(start_epoch).replace('.', '')}epoch.png")
    plt.savefig(output_path)
    plt.close()
    print(f"Plot saved to {output_path}")

if __name__ == "__main__":
    # Define the path to your training_metrics.json file
    json_file = "/Users/romirpatel/ct-diffusionmodelbench/Training/Training Results/Oct2_3epoch_5e-5LR_1000sample_1024maxseqlen/training_metrics.json"
    output_directory = "/Users/romirpatel/ct-diffusionmodelbench/"
    
    # Plot for all epochs
    plot_training_loss(json_file, output_directory, start_epoch=0)
    
    # Plot from epoch 0.5 onwards
    plot_training_loss(json_file, output_directory, start_epoch=0.5)