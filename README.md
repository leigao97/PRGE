
# P-RGE

### Step 1: Create Conda Environment and Install Packages
```bash
conda create -n PRGE python=3.10
conda activate PRGE
pip install -r requirements.txt 
```

### Step 2: Run expriments 
Detailed hyperparameter configurations can be found in the `sweeps` folder. 

An example use of sweep for fine-tuning TinyLlama-1.1B with P-RGE on the glue dataset:
```bash
~> wandb sweep sweeps/P-RGE/glue-tinyllama.yaml
wandb: Creating sweep from: sweeps/P-RGE/glue-tinyllama.yaml
wandb: Created sweep with ID: <ID>
wandb: View sweep at: https://wandb.ai/<unique ID>
wandb: Run sweep agent with: wandb agent <unique ID>
~> wandb agent <unique ID>
```

### Step 3: Check `android` Folder for On-device Expriments 