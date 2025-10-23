import json
import matplotlib.pyplot as plt
import numpy as np
import os

def plot_eval_loss():
    # Path to the training metrics JSON file
    json_path = "/Users/romirpatel/ct-diffusionmodelbench/Training/Training_Results/October/training_metrics.json"
    
    # Check if file exists
    if not os.path.exists(json_path):
        print(f"Error: File {json_path} not found!")
        return
    
    # Load the JSON data
    with open(json_path, 'r') as f:
        data = json.load(f)
    
    # Extract step and eval_loss values
    steps = []
    eval_losses = []
    
    for entry in data:
        if 'step' in entry and 'loss' in entry:
            steps.append(entry['step'])
            eval_losses.append(entry['loss'])
    
    # Create the plot
    plt.figure(figsize=(12, 8))
    plt.plot(steps, eval_losses, 'bo', markersize=4, alpha=0.7)

    # Set fixed y-axis range as requested
    plt.ylim(0.0000, 0.001)
    plt.xlabel('Training Step', fontsize=12)
    plt.ylabel('Eval Loss', fontsize=12)
    plt.title('Evaluation Loss vs Training Step', fontsize=14, fontweight='bold')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    # Save the plot
    output_path = "eval_loss_plot.png"
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Plot saved as {output_path}")
    
    # Show the plot
    plt.show()
    
    # Print some statistics
    print(f"\nStatistics:")
    print(f"Total data points: {len(steps)}")
    print(f"Step range: {min(steps)} to {max(steps)}")
    print(f"Eval loss range: {min(eval_losses)} to {max(eval_losses)}")
    print(f"Final eval loss: {eval_losses[-1]}")
    print(f"All eval_loss values are zero: {all(loss == 0 for loss in eval_losses)}")

if __name__ == "__main__":
    plot_eval_loss()