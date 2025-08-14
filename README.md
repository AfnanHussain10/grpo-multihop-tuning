# 🔍 HotpotQA GRPO Fine-Tuning with OpenPipe ART

This repository contains my experiments on **multi-hop reasoning** using the **HotpotQA** dataset and the **OpenPipe ART** reinforcement learning framework.  
The goal is to fine-tune a Qwen-based agent with **GRPO (Group Relative Policy Optimization)** so it can produce **explicit reasoning traces** and accurate answers for complex questions.

## 📊 Overview

- **Dataset**: [HotpotQA](https://hotpotqa.github.io/) multi-hop QA dataset from the [Agent-R1](https://github.com/0russwest0/Agent-R1) repo  
- **Framework**: [OpenPipe ART](https://github.com/OpenPipe/ART) for reinforcement learning with local model training  
- **Model**: Qwen-based language model fine-tuned to output `<think>` reasoning steps and `<answer>` final answers  
- **Results**: See experiments in the `results/` folder for training curves and accuracy metrics  

---

## 📂 Setup

### 1️⃣ Download & Preprocess HotpotQA Data

```bash
# Clone Agent-R1 repo (contains data preprocessing scripts)
git clone https://github.com/0russwest0/Agent-R1.git
cd Agent-R1

# Create the conda environment
conda create -n verl python==3.10
conda activate verl

# Install verl dependencies
git submodule update --init --recursive
cd verl
pip3 install -e .

# Install vLLM for inference backend
pip3 install vllm

# Install flash-attn for faster attention
pip3 install flash-attn --no-build-isolation
cd ..

# Create data directory
mkdir -p data/hotpotqa

# Run preprocessing
python examples/data_preprocess/hotpotqa.py --local_dir ./data/hotpotqa
```
### 2️⃣ Install & Setup OpenPipe ART
```bash
# Install uv (faster Python package manager)
pip install uv

# Create virtual environment
uv venv
source .venv/bin/activate

# Install OpenPipe ART
uv pip install openpipe-art==0.3.11.post3 "gql<4" --prerelease allow --no-cache-dir
```
## 🚀 Training

Once the dataset and ART environment are ready, you can run one of the GRPO fine-tuning scripts:
```bash
python train_grpo_experiment1.py
```

Each experiment is configured with different:
- Learning rates
- Batch sizes
- Reward shaping parameters
- Temperature & sampling strategies
  
The trained models are saved inside:
```bash
.art/hotpotqa-multihop/models/
```
## 💬 Inference Demo

Here’s a minimal example for running inference on a fine-tuned model:
```bash
from transformers import AutoTokenizer, AutoModelForCausalLM, PeftModel
import torch

BASE_MODEL = "Qwen/Qwen2.5-3B-Instruct"
ADAPTER_PATH = ".art/hotpotqa-multihop/models/hotpotqa-agent/<latest_checkpoint>"

# Load base model
tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)
base_model = AutoModelForCausalLM.from_pretrained(BASE_MODEL, device_map="auto")

# Load adapter
model = PeftModel.from_pretrained(base_model, ADAPTER_PATH)
model = model.merge_and_unload()
model.eval()

# Run inference
prompt = "What city is the birthplace of the person who wrote Pride and Prejudice?"
inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
with torch.no_grad():
    outputs = model.generate(**inputs, max_new_tokens=512, temperature=0.7, top_p=0.9)
print(tokenizer.decode(outputs[0], skip_special_tokens=True))
```

## 📜 License

This project is for research purposes only.
Datasets and model weights should be used in accordance with their respective licenses.
