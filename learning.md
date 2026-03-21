## Inspiration

- **Start with clean text, add noise by masking**
  - `get_batch()` in `train.py` builds `x0` and masked `xt`
- **Random timestep controls noise amount**
  - random `t` sampled per example in `get_batch()`
- **Masking schedule (more masks as `t` increases)**
  - `survival_prob(t)` controls keep/unmask probability
- **Denoising training**
  - model gets `xt, t`, predicts clean `x0` (cross-entropy objective)
- **Inference via iterative unmask/refine**
  - reverse loop in `inference.py`
- **Bidirectional context**
  - non-causal attention (`is_causal=False`)

## Core Learnings

- Noising is done by **masking tokens** (characters here), not adding Gaussian noise to embeddings.
- Training samples random timesteps `t`, builds noisy `x_t`, and learns to recover clean `x_0`.
- Time conditioning (`timestep_emb`) is essential so one model can denoise at different noise levels.
- Bidirectional attention helps denoising because each masked token can use both left and right context.
- Inference is reverse-time denoising (`T -> 1 -> 0`) with iterative refinement via confidence-based re-masking.
- Tokenization is character-level, so generation and the visualization decode **character by character**.
- Context length is fixed by `block_size`; here that means 256-character windows during train/infer.
- Quality is strongly tied to the masking schedule and sampler settings (`temperature`, trace step density, remask policy).

## Loss and Token Generation Logic

Loss used in training:

- Objective follows MDLM-style masked denoising:
  1. Sample timestep `t`.
  2. Mask tokens according to the diffusion schedule.
  3. Replace masked tokens with `[MASK]`.
  4. Predict clean `x0` from noisy `xt`.
  5. Compute categorical CE only on masked positions (`mask == True`).

### Findings: Current Loss vs MDLM Paper

- Earlier implementation differed from the paper in two ways:
  - It used full-sequence CE, so visible tokens dominated the loss via easy copy behavior.
  - It allowed a rare fallback to full-sequence CE when no tokens were masked in a batch.
- Current fix in `train.py`:
  - Enforces at least one masked token per sample in batch construction.
  - Computes CE strictly on masked tokens only (no full-sequence fallback path).
- Practical effect:
  - Reported loss better reflects denoising difficulty on corrupted positions.
  - Fast early loss drops from easy visible-token copying are reduced.

How each character token is generated:

1. Initialize sequence with prompt chars + `_` masks for remaining positions.
2. For each reverse diffusion step `t = T -> 1`, run model on current sequence and timestep embedding.
3. Convert logits to probabilities via softmax (temperature-scaled when `temperature > 0`).
4. Choose token ids:
   - stochastic multinomial sampling when `temperature > 0`
   - greedy argmax when `temperature = 0`
5. Keep prompt positions fixed; update generated positions.
6. Re-mask low-confidence generated positions so later steps can refine them.
7. Run final explicit `t=0` denoise pass and take argmax for the final output.

#### Why video decoding looks character-by-character:

- Vocabulary is character-based, so each predicted token id maps to one character.

## Mercury Alignment Notes

Is this repo following the Mercury training approach exactly?

- Short answer: no, not exactly. It follows the same core diffusion-dLLM recipe, but as a simplified micro implementation.

What matches:

- Random timestep masking/noising of clean text (`x0 -> xt`).
- Time-conditioned Transformer denoiser via `timestep_emb`.
- Masked-position denoising objective: predict clean `x0` from noisy `xt`, optimize CE on masked tokens.
- Reverse iterative denoising at inference (`t = T -> 1`) with an explicit final `t=0` pass.

What is simplified vs Mercury-scale training:

- Small training setup (`~10.7M` params, `block_size=256`, `max_iters=5000`), not trillion-token scale.
- Character-level local dataset (`data.txt`), not large mixed web + curated proprietary corpora.
- No RLHF/DPO alignment stage in this codebase.
- Loss is plain cross-entropy over masked positions only; there is no explicit `gamma(t)` weighting term.
- Forward corruption is implemented directly from `x0` at sampled `t` (masking), rather than exposing a full formal Markov chain object in code.

## Run 2: Change Done

- Updated training objective from full-sequence CE to **masked-position CE only** (`mask == True`) in `train.py`.
- Reason: full-sequence loss was dropping too fast(4.3->1.8->0.6->0.7....) because many visible tokens are easy copy targets.
- Added periodic masked-loss evaluation for both train and validation splits:
  - `eval step {iter} | train masked loss ... | val masked loss ...`
- Updated batch masking to enforce at least one masked token per sample, so masked-only CE is always valid without fallback.

### Why the fast loss drop was expected (before objective change)

- This drop was mostly expected in the previous setup, and part of it was "too easy" loss.
- `step 0 loss 4.3315` is almost exactly the random baseline `ln(vocab_size)`: `ln(76) ~= 4.33`, so initialization and early training behavior were normal.
- In batch corruption (`train.py`, `get_batch`), `t` is sampled uniformly from `1..100`, and the average masked fraction is about `0.368`, so roughly `63%` of tokens stay visible.
- Earlier, loss was computed on all positions (`Model.forward` cross-entropy over full sequence), so the model could quickly learn visible-token copying, which made loss fall fast.
- The dataset is small and repetitive (`~88k` characters of simple stories), so char-level denoising picks up easy patterns quickly.

### Effect of the new objective

- Masked-only CE does not inherently increase randomness.
- It makes the metric more honest for denoising quality by removing easy visible-token positions from the loss.
- Generation randomness still mainly comes from inference choices (`temperature`, multinomial sampling, and remask policy).

## Experiments with Diffusion Steps (T)

Setup and observed train loss:

- `T=30` -> train loss ~`2.0`
- `T=50` -> train loss ~`2.0`
- `T=100` -> train loss ~`1.8`

Condensed takeaway:

- Increasing `T` samples a wider range of noise levels in the objective `E_t E_{x_t}[-log p_theta(x0 | x_t, t)]`.
- Higher `T` includes more heavily masked cases, which can make optimization smoother and reduce average training loss.
- Lower training loss here mainly reflects objective smoothing across noise regimes, not guaranteed better generation quality.
- With fixed training budget, higher `T` means fewer updates per timestep and a longer reverse chain, which can reduce sampling stability.
- Practical tradeoff: higher `T` improves objective smoothness; lower/moderate `T` can improve reverse-process robustness and speed.

one-liner:
- Increasing diffusion steps (`T`) can lower training loss by smoothing denoising across a broader noise spectrum, while hurting reverse-process stability under limited timestep coverage.

## Fundamental Q&A (Follow-up)

### Q1: Was the earlier rapid loss drop mainly because of char-level tokenization?

- Partly, but not mainly.
- Char-level tokenization can lower loss faster because many characters are highly predictable (spaces, common letters, frequent local patterns).
- The bigger reason for the rapid drop was the earlier full-sequence loss on all positions, where many visible tokens were easy copy targets.
- Small/repetitive training data also accelerated this effect.

### Q2: If we switch to word-level tokens and keep old full-sequence loss, will it be better?

- Not automatically.
- Even with word-level masking, full-sequence loss still includes many visible-token positions, so easy-copy domination remains.
- Word-level vocabularies are larger/sparser and can be harder on small datasets; subword tokenization is usually more stable than pure word-level tokenization.
- Better direction: keep masked-position-focused objective (or strong masked weighting), then tune tokenizer choice and masking policy.

### Q3: Does masked-only loss increase randomness, or will model still learn over 5k steps?

- Masked-only loss does not make the model inherently more random.
- It removes easy tokens from the objective, so loss can look higher/harder at first.
- It usually improves what matters: denoising masked tokens correctly.
- Randomness in output is mainly controlled by inference settings (`temperature`, multinomial sampling, remask policy), not by this objective change.
- Over 5k steps, model can learn the denoising behavior, but on small datasets it may plateau early and overfit style patterns.
- Track `val masked loss` and sample quality, not only train loss.

### Q4: Why was loss reducing so fast before the objective change?

- Short answer: that drop was mostly expected in the previous setup, and part of it was "too easy" loss.
- `step 0 loss 4.3315` matched random baseline closely: `ln(vocab_size) = ln(76) ~= 4.33`.
- In corruption (`get_batch`), `t` was uniform in `1..100`, with average masked fraction about `0.368`, so around `63%` tokens were visible.
- Loss was computed over all positions, so visible-token copying reduced loss quickly.
- Dataset size/style also mattered: about `~88k` characters of repetitive simple stories.
- Better metric for true denoising progress: masked-position loss (`mask == True`) plus masked validation loss tracking.

### Q5: Why is `temperature=1.0` random but `temperature=0.0` cleaner and more structured? Is that inherent to diffusion LLMs?

- `temperature` directly controls sampling entropy.
- At `temperature=1.0`, decoding samples from the full token distribution, so lower-probability tokens are picked more often.
- In diffusion decoding, this sampling happens across many reverse denoising steps, so randomness can compound over iterations.
- At `temperature=0.0` (greedy/argmax behavior), each step picks the highest-probability token, so outputs become cleaner and more structured.
- This behavior is expected and is not a bug.
- For diffusion LLMs, iterative denoising can make sampling noise accumulation more visible than one-shot decoding, especially in small models.
- With less training data, `temperature=1.0` often looks noisy rather than creative, because the model's token probabilities are not sharp/reliable enough to support diverse sampling.
- But it is not true that diffusion LLMs are inherently noisy: with stronger models, better data, and controlled decoding (`low temperature`, `top-k/top-p`, improved remask policy), outputs can be very clean.

### Q6: How do diffusion steps affect generation? Do steps need to match context length?

- Diffusion steps (`T`) control how many iterative denoising refinements happen during generation.
- Higher `T` usually gives more opportunities to correct mistakes, but increases latency/cost and can accumulate sampling noise if decoding is too stochastic.
- Lower `T` is faster, but may under-refine and leave more errors.
- `T` does **not** need to equal context length (`block_size`); they are different axes:
  - `block_size` = number of tokens/characters in the sequence window.
  - `T` = number of denoising iterations applied to that window.
- Practical rule: choose `T` by quality-vs-speed tradeoff (and training setup), not by sequence length.
- `T` should be consistent between training assumptions (noise schedule) and inference procedure; large mismatch can hurt quality.

### Q6 Refinements

- `training_steps / T` is a good intuition for timestep coverage, but real coverage also depends on batch size and timestep sampling/weighting; some timesteps can still be undertrained.
- "Entropy amplification" risk is strongest under stochastic decoding; with greedy or constrained decoding, larger `T` can still improve quality, but usually with diminishing returns and higher inference cost.

## Kaggle Longer Runs Support

When running with `torchrun` on Kaggle (2x T4), processes can get `SIGKILL` if host/GPU memory spikes.  
To stabilize longer runs, `train.py` now supports memory-control env vars and mixed precision.

### What was added in `train.py`

- `BATCH_SIZE` (global across GPUs)
- `GRAD_ACCUM_STEPS` (keeps effective batch while lowering peak memory)
- `MAX_TOKENS` (optional cap on tokenized corpus size)
- `EVAL_ITERS`, `LOSS_CURVE_EVERY`, `MAX_ITERS`, `BLOCK_SIZE`, `LR`, etc. via env overrides
- CUDA AMP (`AMP=1` default on CUDA) + `GradScaler`
- Gradient accumulation with DDP `no_sync()` for intermediate micro-steps

### Recommended 2x T4 command (stable for long runs)

```bash
CUDA_VISIBLE_DEVICES=0,1 \
BATCH_SIZE=16 \
GRAD_ACCUM_STEPS=4 \
MAX_TOKENS=300000 \
EVAL_ITERS=10 \
LOSS_CURVE_EVERY=50 \
torchrun --standalone --nproc_per_node=2 train.py
```

Notes:
- Effective batch here is `64` (`16 * 4`) with much lower peak memory.
- If restarts continue: set `BATCH_SIZE=8` and/or `BLOCK_SIZE=192` (or `128`).
- If dataset is very large, reduce `MAX_TOKENS` further.

### Why these hyperparameter choices

- `BATCH_SIZE=16` (global on 2 GPUs):
  - Per-GPU microbatch becomes `8`, which is much safer for T4 memory than larger per-rank batches.
  - Keeps throughput reasonable while avoiding frequent OOM kills.

- `GRAD_ACCUM_STEPS=4`:
  - Effective batch size remains `64` (`16 * 4`) without requiring memory for 64 samples at once.
  - Preserves optimization behavior closer to the original larger-batch setup.

- `AMP=1` (default on CUDA):
  - FP16 activations/gradients significantly reduce VRAM usage.
  - Usually gives speedup on T4 while staying stable with `GradScaler`.

- `MAX_TOKENS=300000`:
  - Caps host RAM and preprocessing footprint in notebook environments.
  - Useful for Kaggle sessions where RAM spikes can trigger process kills.

- `EVAL_ITERS=10`:
  - Validation becomes much cheaper, reducing periodic memory/time spikes.
  - Enough for trend monitoring during long runs; can be increased later for final reporting.

- `LOSS_CURVE_EVERY=50`:
  - Reduces per-step eval/log overhead and memory churn from frequent val probes.
  - Keeps diagnostics without stressing notebook runtime.

- Keep `BLOCK_SIZE=256` first, then reduce only if needed:
  - Sequence length drives activation memory strongly (`O(B * L * d)` plus attention terms).
  - Dropping to `192` or `128` is a reliable fallback when OOM persists.

## Muon + DDP Mistake (Kaggle)

### Symptom

- Error on multi-GPU run:
  - `AttributeError: 'DistributedDataParallel' object has no attribute 'blocks'`

### Root cause

- After wrapping with DDP, `model` is a `DistributedDataParallel` wrapper.
- Muon parameter grouping was trying to access internals like `model.blocks`, `model.lm_head`, etc. directly from the wrapper.

### Fix

- Build Muon param groups from the unwrapped module:
  - `core_model = unwrap_model(model)` (equivalent to `model.module` under DDP).
- Then use:
  - `core_model.blocks`, `core_model.lm_head`, `core_model.token_emb`, `core_model.timestep_emb`.

### Extra cleanup done

- Replaced deprecated `torch.cuda.amp.GradScaler(...)` path with `torch.amp.GradScaler("cuda", ...)` (fallback kept for compatibility).
- Removed duplicate scaler initialization.
