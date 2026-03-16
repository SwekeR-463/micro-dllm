# micro-dllm

This repo trains a micro-dLLM(~12.23M) based on [mercury's training and inference](https://arxiv.org/abs/2506.17298) approach with BPE tokens and generates text with iterative denoising.


[![Diffusion Trace](./artifacts/media/diffusion_trace.gif)](./artifacts/media/diffusion_trace.mp4)


## What This Implements

- Forward corruption with `[MASK]` tokens over random timesteps `t in [1..T]`
- Time-conditioned Transformer denoiser (`timestep_emb`)
- Full-sequence denoising objective (predict clean `x0` from noisy `x_t`)
- Reverse denoising at inference (`t = T -> 1`, plus final `t=0` pass)
- Confidence-based remasking for iterative refinement
- MP4 trace output for denoising steps

## Architecture Details

- Tokenization: BPE tokenizer from `data/stories.txt` + `[MASK]` token
- Context length: `block_size = 256` tokens
- Diffusion steps: `T = 100`
- Layers: `n_layer = 6`
- Attention heads: `n_head = 6`
- Embedding dimension: `n_embd = 384`
- Head dimension: `head_dim = 64`
- Parameters: `10,706,304` (~`10.71M`, with current `vocab_size = 66`)
- Attention type: bidirectional (`is_causal=False`)
- Positional scheme: RoPE (precomputed rotary cos/sin buffers)
- Normalization: RMSNorm via `F.rms_norm`
- MLP: expansion `4 * n_embd` with `relu(x)^2` nonlinearity
- Timestep conditioning: learned embedding `Embedding(T + 1, n_embd)`

Training setup:

- Batch size: `64`
- Max iterations: `5000`
- Optimizer: `AdamW`
- Learning rate: `3e-4`
- Forward noising: cosine survival schedule  
  `a_t = cos((t / T) * pi / 2)`
- Objective: predict clean sequence `x0` from noisy `xt` (cross-entropy on masked positions)

Inference setup:

- Reverse denoising loop: `t = T -> 1`
- Per-step decoding: multinomial sampling (`temperature > 0`) or greedy (`temperature = 0`)
- Iterative refinement: low-confidence generated tokens are re-masked each step
- Final explicit denoise at `t=0` with greedy selection
- Visualization outputs: MP4 timeline of decoding steps


## Project Files

- Root:
  - `train.py`: model + training
  - `inference.py`: checkpoint loading + generation/trace export
  - `README.md`, `learning.md`, `requirements.txt`
- `scripts/`: dataset/tokenizer preparation scripts
- `data/`: local corpus files (`stories.txt`, etc.)
- `utils/`: shared utility modules
- `artifacts/`: checkpoints, tokenizer JSON, plots, and media outputs

## Setup

Requirements:

- Python 3
- `torch`
- `Pillow` (for frame rendering)
- `ffmpeg` (for MP4 export)
- `tokenizers` (for BPE tokenization)

## Training

```bash
python3 scripts/data.py --num-stories 10000 --output data/stories.txt
python3 scripts/train_tokenizer.py --input data/stories.txt --output artifacts/tokenizer/tokenizer.json
# single GPU / CPU
python3 train.py
```

Checkpoints are saved to `artifacts/models/` during training and at the end.

You can run `python3 train.py` directly for single-GPU training (`WORLD_SIZE=1` path).

### Kaggle 2x T4 (Longer Run Stable Preset)

```bash
CUDA_VISIBLE_DEVICES=0,1 \
BATCH_SIZE=16 \
GRAD_ACCUM_STEPS=4 \
MAX_TOKENS=300000 \
EVAL_ITERS=10 \
LOSS_CURVE_EVERY=50 \
torchrun --standalone --nproc_per_node=2 train.py
```

This keeps effective batch size at `64` with lower peak memory.
If OOM restarts continue, try `BATCH_SIZE=8` and/or `BLOCK_SIZE=192` (or `128`).

At the end of training, `train.py` also prints a final validation metrics block with:

- `Perplexity` (derived from masked validation cross-entropy)
- `Masked reconstruction accuracy` (accuracy on corrupted positions only)
- `Entropy per timestep` (masked-token predictive entropy across diffusion timesteps)
- `Reverse-step token change rate` (fraction of generated tokens that change between reverse steps)
- `Distinct-2 diversity` (unique generated bigrams / total generated bigrams, prompt excluded)

## Inference + Visualizer

```bash
python3 inference.py \
  --checkpoint artifacts/models/model_stories_10k_256_muon.pt \
  --prompt "Once upon a time" \
  --gen-len 256 \
  --temperature 0.0 \
  --viz-video artifacts/media/diffusion_trace_adamw.mp4 \
  --trace-every 1 \
  --gif-frame-ms 180
```

## TinyStories Dataset Prep

Use `scripts/data.py` to stream TinyStories from Hugging Face and build a small local subset for laptop training:

```bash
python3 scripts/data.py \
  --dataset roneneldan/TinyStories \
  --split train \
  --num-stories 40000 \
  --seed 1337 \
  --output data/stories.txt
```

Notes:

- `--num-stories 100` or `--num-stories 200` is a good range for quick local runs.
- Sampling uses streaming + shuffle buffer, so it does not download the full dataset at once.
- `train.py` and `inference.py` use `data/stories.txt` and `artifacts/tokenizer/tokenizer.json`.


## Practical Next Improvements/Experiments

- [ ] Add resume training from checkpoint (`--resume model.pt`)
- [x] Train on 100-200 Tiny Stories
- [x] Train on 2k+ Stories
- [ ] Do loss curve ablations with gpt2 config for arm vs dif on 100-200 tiny stories
- [ ] Muon Ablations
- [ ] Train on SynTH/fineweb-edu
- [ ] Try speed running for a 200M+ param model
- [ ] Adding training with block based masking
- [ ] Using uniform diffusion instead of masked diffusion
