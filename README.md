# ViAMR: Fine-tuning LLMs for Abstract Meaning Representation in Vietnamese

🇻🇳 **Vietnamese AMR Parser for VLSP 2025 Competition**

## 📍 Overview

This project implements a Vietnamese Abstract Meaning Representation (AMR) parser developed for the VLSP 2025 competition. The system converts Vietnamese sentences into their semantic AMR representations using state-of-the-art language models with supervised fine-tuning (SFT) and reinforcement learning approaches (GRPO).

## 🎯 Features

* **Vietnamese AMR Parsing**: Convert Vietnamese sentences to PENMAN-format AMR graphs
* **Multiple Training Approaches**:
  * Supervised Fine-Tuning (SFT)
  * Group Relative Policy Optimization (GRPO) with reinforcement learning
* **Advanced Post-processing**: Comprehensive AMR validation and correction
* **Evaluation Metrics**: Automated scoring and evaluation system
* **DeepSpeed Integration**: Efficient training with ZeRO optimization

## 🏗️ Architecture

```
ViAMR/
├── viamr/                     # Python package
│   ├── data_processing.py     # Parse raw AMR files into DataFrames
│   ├── dataset.py             # Build HuggingFace datasets (SFT/GRPO)
│   ├── prompts.py             # System prompts
│   ├── postprocessing.py      # AMR/PENMAN sanitization pipeline
│   ├── rewards.py             # SMATCH + combined reward for GRPO
│   ├── inference.py           # QwenReasoner + batch inference CLI
│   ├── scoring.py             # SMATCH evaluation CLI
│   ├── split_data.py          # Train/test split CLI
│   └── training/
│       ├── _common.py         # Shared model/tokenizer/LoRA helpers
│       ├── sft.py             # Supervised fine-tuning CLI
│       └── grpo.py            # GRPO training CLI
├── config/
│   └── ds_zero2.json          # DeepSpeed ZeRO stage 2 config
└── scripts/                   # Bash wrappers for the CLIs
    ├── train_sft.sh
    ├── train_grpo.sh
    ├── infer.sh
    ├── get_score.sh
    └── main.sh
```

## 🚀 Setup and Usage

### 1. Installation

```bash
# From the repo root
pip install -r requirements.txt
```

All CLIs are exposed as modules of the `viamr` package, so run them from the
repo root (or from anywhere with `PYTHONPATH` pointing at the repo):

```bash
python -m viamr.training.sft   ...
python -m viamr.training.grpo  ...
python -m viamr.inference      ...
python -m viamr.scoring        ...
python -m viamr.split_data     ...
```

### 2. Data Preparation

```bash
python -m viamr.split_data \
    --inputs data/train_amr_1.txt data/train_amr_2.txt \
    --train_out data/train.txt \
    --test_out  data/test.txt \
    --test_ratio 0.15
```

### 3. Training Models

#### Supervised Fine-Tuning (SFT)

```bash
# Train with supervised fine-tuning
bash scripts/train_sft.sh
```

#### GRPO Reinforcement Learning

```bash
# Train with Group Relative Policy Optimization
bash scripts/train_grpo.sh
```

### 4. Inference

```bash
# Run AMR parsing inference
bash scripts/infer.sh

# Or run the main pipeline
bash scripts/main.sh
```

### 5. Evaluation

```bash
# Evaluate model performance
bash scripts/get_score.sh
```

## 📊 Key Components

### AMR Parser (`viamr/inference.py`)

The main parsing component is the `QwenReasoner` class:

```python
from viamr.inference import QwenReasoner

reasoner = QwenReasoner(model_name="outputs/Qwen-1.7B-SFT-2")
thinking, amr = reasoner.inference("câu tiếng việt", is_extract_amr=True, is_thinking=True)
```

### Post-processing (`viamr/postprocessing.py`)

AMR/PENMAN sanitization functions:

* `penman_safe_minimal` — canonical sanitization pipeline
* `has_duplicate_nodes` — check for duplicate variable names
* `balance_parens` — fix parentheses balance
* `fix_amr_vars` — correct variable declarations
* `normalize_roles_spacing`, `join_concepts_underscores`, `strip_orphan_slashes`

### Prompting System (`viamr/prompts.py`)

Structured prompts with Vietnamese-specific instructions:

```python
SYSTEM_PROMPT = '''
Bạn là một mô hình ngôn ngữ lớn chuyên về phân tích cú pháp ngữ nghĩa cho tiếng Việt. 
Nhiệm vụ của bạn là chuyển đổi một câu tiếng Việt đầu vào thành biểu diễn AMR hoàn chỉnh.
'''
```

## 🛠️ Configuration

### Training Configuration

* **DeepSpeed**: `config/ds_zero2.json` - ZeRO stage 2 optimization
* **Model Support**: Qwen2.5, LLaMA3, and other transformer models
* **RL Training**: GRPO algorithm with custom reward functions

### Key Parameters

* **Max Sequence Length**: 2048 tokens
* **Training Approaches**: SFT + GRPO reinforcement learning
* **Output Format**: PENMAN notation AMR graphs
* **Language**: Vietnamese with underthesea tokenization

## 📈 Model Training

### Supervised Fine-Tuning

`viamr/training/sft.py` trains the model on Vietnamese sentence-AMR pairs with standard cross-entropy loss.

### Reinforcement Learning (GRPO)

`viamr/training/grpo.py` drives GRPO training:
* Custom reward functions from `viamr/rewards.py`
* Group Relative Policy Optimization
* AMR quality-based rewards

## 🔍 Evaluation

The evaluation CLI (`viamr/scoring.py`) provides:
* AMR graph accuracy metrics
* Semantic similarity scoring
* Structure validation checks
* Performance benchmarking

## 📝 Usage Example

```python
from viamr.inference import QwenReasoner
from viamr.postprocessing import penman_safe_minimal

reasoner = QwenReasoner(model_name="outputs/Qwen-1.7B-SFT-2")

sentence = "Tôi đang học tiếng Việt."
_, amr_result = reasoner.inference(sentence, is_extract_amr=True)

cleaned_amr = penman_safe_minimal(amr_result)
print(cleaned_amr)
```

## 🤝 Contributing

This project is developed for the VLSP 2025 competition. The system focuses on Vietnamese language processing and AMR semantic representation.

## 📚 References

* Vietnamese Language Processing
* Abstract Meaning Representation (AMR)
* PENMAN Notation
* Group Relative Policy Optimization (GRPO)
