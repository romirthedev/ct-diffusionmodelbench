1. Transfer Your Local Repo to TACC Scratch
On your local terminal (Mac):

```bash
scp -r /Users/romirpatel/ct-diffusionmodelbench romirpatel@vista.tacc.utexas.edu:/scratch/10936/romirpatel/
```

2. SSH to Vista
```bash
ssh romirpatel@vista.tacc.utexas.edu
```

3. Free Up Home Directory Quota (if needed)
Check quota:

```bash
myquota
```
If home quota is full, clean cache:

```bash
rm -rf ~/.cache/huggingface ~/.cache/torch
```
Move big files from home to scratch as needed.

4. Launch Interactive GPU Session
```bash
idev -p gh-dev -N 1 -n 1 -t 02:00:00
```

5. Load Required Modules
Once inside the compute node session:

```bash
module load gcc
module load cuda
module load python3
```

6. Navigate to Your Project on Scratch
```bash
cd /scratch/10936/romirpatel/ct-diffusionmodelbench
```

7. Setup and Activate Virtual Environment
If you haven't created the virtual environment:

```bash
python3 -m venv venv
source venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
```
If you already created one:

```bash
source venv/bin/activate
```

8. Set Hugging Face Cache Directory
Set these environment variables to avoid filling home directory:

```bash
export HF_HOME=$SCRATCH/huggingface_cache
export HF_HUB_CACHE=$HF_HOME
export TRANSFORMERS_CACHE=$HF_HOME
mkdir -p $HF_HOME
```

9. In order to run inference, run the following command:

```bash
python Inference/chat_finetuned.py
```