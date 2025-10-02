For this, we will be using the GSAI-ML model Llada-8B-Instruct

This is th emodel which we are fine-training. The results for the model initially are at a strict 0% success rate with only hints of Lean3, with no Lean4. 

## Accessing the Trained Model on TACC Vista

After a successful training run, the model, training plots, and metrics are saved in the following directory on TACC Vista:

`/scratch/10936/romirpatel/ct-diffusionmodelbench/Training/llada-numina-finetuned`

To transfer the entire `llada-numina-finetuned` directory to your local machine, open a new terminal on your local computer and run:

```bash
scp -r romirpatel@vista.tacc.utexas.edu:/scratch/10936/romirpatel/ct-diffusionmodelbench/Training/llada-numina-finetuned /Users/romirpatel/ct-diffusionmodelbench/Training/
```

To specifically transfer the `training_plots.png` file, use:

```bash
scp romirpatel@vista.tacc.utexas.edu:/scratch/10936/romirpatel/ct-diffusionmodelbench/Training/llada-numina-finetuned/training_plots.png /Users/romirpatel/ct-diffusionmodelbench/
```

Remember to replace `/Users/romirpatel/ct-diffusionmodelbench/Training/` or `/Users/romirpatel/ct-diffusionmodelbench/` with your desired local destination path if it's different.