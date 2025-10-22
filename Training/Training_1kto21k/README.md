# Continued Training: LLaDA 1k to 21k

This directory contains the setup for continued training of the already finetuned LLaDA model on rows 1000-21000 of the NuminaMath-LEAN dataset.

## Key Changes from Original Training

### Model Configuration
- **Base Model**: Uses the already finetuned model from `../llada-numina-finetuned`
- **Training Type**: Continued fine-tuning (not training from scratch)
- **Dataset Range**: Rows 1000-21000 (20,000 additional samples)
- **Output Directory**: `./llada-numina-continued-1kto21k`

### Training Parameters
- **Learning Rate**: Reduced to 3e-5 (from 5e-5) for continued training
- **Warmup Steps**: Increased to 200 (from 100) for larger dataset
- **Save/Eval Steps**: Increased to 100 (from 50) for longer training
- **Logging Steps**: Increased to 20 (from 10)
- **Runtime**: Extended to 6 hours (from 2 hours) in SLURM script

### Dataset Configuration
- **Start Row**: 1000 (skips first 1000 samples used in initial training)
- **Max Samples**: 20000 (trains on 20k additional samples)
- **Total Range**: Rows 1000-20999

## Files

- `train.py`: Updated training script that loads the finetuned model and trains on rows 1k-21k
- `run_training.sh`: Updated SLURM script with extended runtime and new job name
- `README.md`: This documentation file

## Usage

1. Ensure the finetuned model exists at `../llada-numina-finetuned`
2. Submit the training job:
   ```bash
   sbatch run_training.sh
   ```

## Output

The continued training will produce:
- **Model**: Saved to `./llada-numina-continued-1kto21k`
- **Metrics**: Training metrics in `training_metrics.json`
- **Plots**: Training loss plots in `training_plots.png`
- **Config**: Training configuration in `training_config.json`

## Verification

The training script includes checks to:
- Verify the finetuned model exists before starting
- Log the exact dataset range being used
- Track total parameters being trained
- Save comprehensive training configuration for reproducibility