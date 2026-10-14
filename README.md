# CoPT: Context-based Prompt Tuning for Pretrained Language Model via Test-time Adaptation

This repository provides the implementation of **CoPT**, proposed in *CoPT: Context-based Prompt Tuning for Pretrained Language Model via Test-time Adaptation*, published in **Expert Systems with Applications**.

CoPT incorporates batch-derived contextual information into soft prompts through a low-rank parameterization.

**Paper:** https://doi.org/10.1016/j.eswa.2026.134712

## Usage

The following instructions describe how to add a soft prompt module and run a basic prompt-tuning experiment.

### 1. Configure your prompt module

Add the configuration for your method to `cfg.py`. For example:

```python
# Residual prompt configuration
CFG.RESIDUAL = CN()
CFG.RESIDUAL.HIDDEN_SIZE = 400
CFG.RESIDUAL.DROP_RATE = 0.05
CFG.RESIDUAL.INIT_METHOD = 'random'

# Configuration for a new prompt module
CFG.MY_PROMPT = CN()
CFG.MY_PROMPT.HIDDEN_SIZE = 10
```

### 2. Implement your prompt module

The following example defines a basic trainable soft prompt. Register your module in the project's method registry before using it.

```python
class Vanilla(nn.Module):
    """A basic trainable soft prompt."""

    def __init__(self, cfg, prefix_len=10, emb_size=768):
        super().__init__()

        self.prefix_len = prefix_len
        self.emb_size = emb_size
        self.prompt = nn.Parameter(
            torch.empty(self.prefix_len, self.emb_size)
        )
        self.init_prompt()

    def init_prompt(self):
        with torch.no_grad():
            self.prompt.uniform_(-0.5, 0.5)

    def generate(self):
        return self.prompt
```

### 3. Run a training example

The following command runs the **vanilla prompt-tuning baseline** on COPA with T5-base:

```bash
CUDA_VISIBLE_DEVICES=0 python main.py \
    --datasets='copa' \
    --model_name=t5-base \
    --method vanilla \
    --enc_prompt_tokens 10 \
    -ts 16 \
    -e 10 \
    --save_name Vanilla_test
```

This is a basic usage example. Its settings differ from those used in the paper.

## Acknowledgements

This codebase builds on the following repositories:

- https://github.com/XYaoooo/DPT
- https://github.com/arazd/ResidualPrompts
