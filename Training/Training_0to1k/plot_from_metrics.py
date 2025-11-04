import os
import json
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

def plot_training_metrics(logs, output_dir):
    train_step_logs = [log for log in logs if "loss" in log and "step" in log]
    train_final_logs = [log for log in logs if "train_loss" in log and "step" in log]
    eval_logs = [log for log in logs if "eval_loss" in log and "step" in log]

    if not (train_step_logs or train_final_logs or eval_logs):
        print("No training or eval logs found; nothing to plot.")
        return

    two_panels = bool((train_step_logs or train_final_logs) and eval_logs)
    plt.figure(figsize=(12, 5))

    if train_step_logs or train_final_logs:
        ax1 = plt.subplot(1, 2, 1) if two_panels else plt.gca()
        if train_step_logs:
            steps = [log["step"] for log in train_step_logs]
            losses = [log["loss"] for log in train_step_logs]
            ax1.plot(steps, losses, 'b-', marker='o', markersize=3, label='Training Loss')
        if train_final_logs:
            f_steps = [log["step"] for log in train_final_logs]
            f_losses = [log["train_loss"] for log in train_final_logs]
            ax1.plot(f_steps, f_losses, 'bx', markersize=6, label='Final Train Loss')
        ax1.set_xlabel('Steps')
        ax1.set_ylabel('Loss')
        ax1.set_title('Training Loss (LLaDA-MoE)')
        ax1.grid(True)
        ax1.legend()

    if eval_logs:
        ax2 = plt.subplot(1, 2, 2) if two_panels else plt.gca()
        eval_steps = [log["step"] for log in eval_logs]
        eval_values = [log["eval_loss"] for log in eval_logs]
        ax2.plot(eval_steps, eval_values, 'r-', marker='s', markersize=3, label='Validation Loss')
        ax2.set_xlabel('Steps')
        ax2.set_ylabel('Loss')
        ax2.set_title('Validation Loss (LLaDA-MoE)')
        ax2.grid(True)
        ax2.legend()

    plt.tight_layout()
    out_path = os.path.join(output_dir, "training_plots.png")
    plt.savefig(out_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved plot to {out_path}")

if __name__ == "__main__":
    output_dir = os.environ.get("FAST_OUTPUT_DIR", "./llada-moe-numina-finetuned-optimized")
    metrics_file = os.path.join(output_dir, "training_metrics.json")
    if not os.path.exists(metrics_file):
        raise SystemExit(f"Metrics file not found: {metrics_file}")
    with open(metrics_file, "r") as f:
        logs = json.load(f)
    plot_training_metrics(logs, output_dir)
