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

- Objective is full-sequence denoising: predict clean `x0` from noisy `xt`.
- Loss function is categorical cross-entropy over vocabulary logits:
  - `logits_flat = logits.view(B * Tseq, vocab_size)`
  - `targets_flat = targets.view(B * Tseq)`
  - `loss = F.cross_entropy(logits_flat, targets_flat)`
- This is what teaches each position to recover its clean character at any sampled timestep `t`.

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
- Full-sequence denoising objective: predict clean `x0` from noisy `xt`.
- Reverse iterative denoising at inference (`t = T -> 1`) with an explicit final `t=0` pass.

What is simplified vs Mercury-scale training:

- Small training setup (`~10.7M` params, `block_size=256`, `max_iters=5000`), not trillion-token scale.
- Character-level local dataset (`data.txt`), not large mixed web + curated proprietary corpora.
- No RLHF/DPO alignment stage in this codebase.
- Loss is plain cross-entropy over all positions; there is no explicit `gamma(t)` weighting term.
- Forward corruption is implemented directly from `x0` at sampled `t` (masking), rather than exposing a full formal Markov chain object in code.
