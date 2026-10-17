# CoPT: Context-based Prompt Tuning for Pretrained Language Model via Test-time Adaptation

This repository provides the implementation of **CoPT**, proposed in *CoPT: Context-based Prompt Tuning for Pretrained Language Model via Test-time Adaptation*, published in **Expert Systems with Applications**.

CoPT conditions soft prompts on batch-derived contextual information through a low-rank parameterization. The pretrained T5 backbone is frozen during training, while the prompt parameters are learned. At inference time, the context is recomputed from the incoming batch without gradient updates.

**Paper:** https://doi.org/10.1016/j.eswa.2026.134712

## Installation

The training code uses CUDA and requires an NVIDIA GPU with a compatible driver. The dependency versions in `requirements.txt` were selected from our Python 3.10.20 environment, which used PyTorch 2.1.2 with CUDA 12.1.

Clone the repository and enter its directory:

```bash
git clone https://github.com/bloomer-in-making/CoPT.git
cd CoPT
```

Create an environment and install the CUDA build of PyTorch before installing the remaining dependencies:

```bash
conda create -n copt python=3.10.20 -y
conda activate copt

python -m pip install torch==2.1.2 --index-url https://download.pytorch.org/whl/cu121
python -m pip install -r requirements.txt
python -m pip check
```

Model weights, tokenizers, and datasets are loaded through Hugging Face and require network access on the first run unless already cached. The reduced dependency list is derived from our existing environment; a clean installation of this list has not yet been independently validated.

## Run CoPT

### 1. Check the bottleneck configuration

CoPT is already implemented in `method/copt.py` and registered as `copt` in `method/__init__.py`. No new prompt module is needed.

In the current implementation, CoPT reads its bottleneck setting from `cfg.py`:

```python
CFG.DPT = CN()
CFG.DPT.HIDDEN_SIZE = 10
```

Despite the configuration name, this setting is also used by CoPT. With a value of 10, CoPT combines nine learned rows with one batch-derived context row. The final column of `prompt_first` provides the learned scale for the context contribution.

**Changing `--bottle_neck` alone does not change the CoPT module's bottleneck size.** To change it, update `CFG.DPT.HIDDEN_SIZE`; keep the command-line value consistent.

### 2. Train and evaluate

Run CoPT on COPA with T5-base:

```bash
mkdir -p results

CUDA_VISIBLE_DEVICES=0 python main.py \
    --datasets copa \
    --model_name t5-base \
    --method copt \
    --enc_prompt_tokens 100 \
    --bottle_neck 10 \
    --train_batch_size 16 \
    --eval_batch_size 128 \
    --n_epochs 100 \
    --lr 0.3 \
    --seed 2024 \
    --save_dir ./results \
    --save_name t5-base_copa_copt_seed2024 \
    --save_ckpt_dir ./saved_models/t5-base_copa_copt_seed2024 \
    --model_output_path ./saved_outputs/t5-base_copa_copt_seed2024
```

Validation runs automatically during training.

- **Tasks:** `boolq`, `cb`, `copa`, `rte`, `wic`, `wsc`.
- **Models:** `t5-small`, `t5-base`, `t5-large`.
- **Multiple runs:** change `--seed` and use distinct output names and directories. Compute the mean and sample standard deviation across runs separately.

## Code structure

- `method/copt.py`: CoPT prompt parameterization and batch-context computation.
- `models/t5.py`: Integration of soft prompts with T5.
- `main.py`: Training and validation.
- `cfg.py`: Prompt-module configuration.
- `options.py`: Command-line arguments.
- `utils/`: Dataset loading, preprocessing, and evaluation utilities.

Other prompt modules are registered under `vanilla`, `residual`, and `dpt`. The examples above focus on CoPT.

## Acknowledgements

This codebase builds on the following repositories:

- [Decomposed Prompt Tuning (DPT)](https://github.com/XYaoooo/DPT)
- [Residual Prompt Tuning](https://github.com/arazd/ResidualPrompts)

We thank the authors for making their implementations publicly available.
