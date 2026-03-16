import os
import sys
import time
import math
import torch
import torch.nn as nn
from torch.nn import functional as F
from utils.tokenizer_utils import load_tokenizer

# Hyperparameters
batch_size = 64
block_size = 256
max_iters = 20000
eval_interval = 500
learning_rate = 3e-4
eval_iters = 200
save_interval = 500
checkpoint_path = "artifacts/models/model_stories_10k_bpe_256.pt"
loss_curve_path = "artifacts/media/loss_curves.png"

n_embd = 384
n_head = 6
n_layer = 6
head_dim = n_embd // n_head

T = 100  # diffusion steps
tokenizer_path = "artifacts/tokenizer/tokenizer.json"
stories_path = "data/stories.txt"

device = (
    "cuda"
    if torch.cuda.is_available()
    else ("mps" if torch.backends.mps.is_available() else "cpu")
)

torch.manual_seed(1337)

# Data
with open(stories_path, "r", encoding="utf-8") as f:
    text = f.read()

tokenizer = load_tokenizer(tokenizer_path)
vocab_size = tokenizer.get_vocab_size()

mask_token_id = tokenizer.token_to_id("[MASK]")
if mask_token_id is None:
    raise ValueError("Tokenizer is missing [MASK] token. Re-train tokenizer.")

def encode(s):
    return tokenizer.encode(s).ids

def decode(l):
    return tokenizer.decode(l, skip_special_tokens=False)

data = torch.tensor(encode(text), dtype=torch.long)
n = int(0.9 * len(data))
train_data = data[:n]
val_data = data[n:]

# Diffusion Schedule
def survival_prob(t):
    # cosine schedule (better than linear)
    return math.cos((t / T) * math.pi / 2)**2

# Batch Loader
def get_batch(split):
    data_split = train_data if split == "train" else val_data
    idx = torch.randint(len(data_split) - block_size, (batch_size,))
    x0 = torch.stack([data_split[i : i + block_size] for i in idx])

    t = torch.randint(1, T + 1, (batch_size,))

    xt = x0.clone()
    mask = torch.zeros_like(x0, dtype=torch.bool)

    for i in range(batch_size):
        a_t = survival_prob(t[i].item())
        token_mask = torch.rand(block_size) > a_t
        xt[i][token_mask] = mask_token_id
        mask[i] = token_mask

    return (
        xt.to(device),
        x0.to(device),
        mask.to(device),
        t.to(device),
    )


def get_batch_at_t(split, t_value):
    data_split = train_data if split == "train" else val_data
    idx = torch.randint(len(data_split) - block_size, (batch_size,))
    x0 = torch.stack([data_split[i : i + block_size] for i in idx])

    t_value = int(t_value)
    t = torch.full((batch_size,), t_value, dtype=torch.long)
    a_t = survival_prob(t_value)

    token_mask = torch.rand(batch_size, block_size) > a_t
    xt = x0.clone()
    xt[token_mask] = mask_token_id

    return (
        xt.to(device),
        x0.to(device),
        token_mask.to(device),
        t.to(device),
    )

# Model
def norm(x):
    return F.rms_norm(x, (x.size(-1),))

def apply_rotary_emb(x, cos, sin):
    d = x.shape[3] // 2
    x1, x2 = x[..., :d], x[..., d:]
    y1 = x1 * cos + x2 * sin
    y2 = x1 * (-sin) + x2 * cos
    return torch.cat([y1, y2], 3)

class MultiHeadAttention(nn.Module):
    def __init__(self):
        super().__init__()
        self.c_q = nn.Linear(n_embd, n_embd, bias=False)
        self.c_k = nn.Linear(n_embd, n_embd, bias=False)
        self.c_v = nn.Linear(n_embd, n_embd, bias=False)
        self.c_proj = nn.Linear(n_embd, n_embd, bias=False)

    def forward(self, x, cos_sin):
        B, Tseq, C = x.size()

        q = self.c_q(x).view(B, Tseq, n_head, head_dim)
        k = self.c_k(x).view(B, Tseq, n_head, head_dim)
        v = self.c_v(x).view(B, Tseq, n_head, head_dim)

        cos, sin = cos_sin
        q = apply_rotary_emb(q, cos, sin)
        k = apply_rotary_emb(k, cos, sin)

        q = norm(q)
        k = norm(k)

        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)

        y = F.scaled_dot_product_attention(q, k, v, is_causal=False)

        y = y.transpose(1, 2).contiguous().view(B, Tseq, -1)
        return self.c_proj(y)

class MLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.c_fc = nn.Linear(n_embd, 4 * n_embd, bias=False)
        self.c_proj = nn.Linear(4 * n_embd, n_embd, bias=False)

    def forward(self, x):
        return self.c_proj(F.relu(self.c_fc(x)).square())

class Block(nn.Module):
    def __init__(self):
        super().__init__()
        self.attn = MultiHeadAttention()
        self.mlp = MLP()

    def forward(self, x, cos_sin):
        x = x + self.attn(norm(x), cos_sin)
        x = x + self.mlp(norm(x))
        return x

class Model(nn.Module):
    def __init__(self):
        super().__init__()

        self.token_emb = nn.Embedding(vocab_size, n_embd)
        self.timestep_emb = nn.Embedding(T + 1, n_embd)

        self.rotary_seq_len = block_size * 2
        cos, sin = self._precompute_rotary_embeddings(self.rotary_seq_len)
        self.register_buffer("cos", cos, persistent=False)
        self.register_buffer("sin", sin, persistent=False)

        self.blocks = nn.ModuleList([Block() for _ in range(n_layer)])
        self.lm_head = nn.Linear(n_embd, vocab_size, bias=False)

        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear) or isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def _precompute_rotary_embeddings(self, seq_len, base=10000):
        channel_range = torch.arange(0, head_dim, 2, dtype=torch.float32)
        inv_freq = 1.0 / (base ** (channel_range / head_dim))
        t = torch.arange(seq_len, dtype=torch.float32)
        freqs = torch.outer(t, inv_freq)
        cos, sin = freqs.cos(), freqs.sin()
        return cos[None, :, None, :], sin[None, :, None, :]

    def forward(self, idx, t=None, targets=None, mask=None):
        B, Tseq = idx.size()

        x = self.token_emb(idx)

        if t is not None:
            x = x + self.timestep_emb(t).unsqueeze(1)

        x = norm(x)

        cos_sin = (self.cos[:, :Tseq], self.sin[:, :Tseq])

        for block in self.blocks:
            x = block(x, cos_sin)

        x = norm(x)
        logits = self.lm_head(x)

        loss = None
        if targets is not None:
            # Train denoising only on corrupted positions to avoid easy copy loss.
            if mask is not None:
                masked_logits = logits[mask]
                masked_targets = targets[mask]
                if masked_targets.numel() > 0:
                    loss = F.cross_entropy(masked_logits, masked_targets)
                else:
                    # Extremely rare edge case when no tokens are masked in batch.
                    logits_flat = logits.view(B * Tseq, -1)
                    targets_flat = targets.view(B * Tseq)
                    loss = F.cross_entropy(logits_flat, targets_flat)
            else:
                logits_flat = logits.view(B * Tseq, -1)
                targets_flat = targets.view(B * Tseq)
                loss = F.cross_entropy(logits_flat, targets_flat)

        return logits, loss


@torch.no_grad()
def estimate_loss(model):
    model.eval()
    losses = {}
    for split in ["train", "val"]:
        split_losses = torch.zeros(eval_iters)
        for k in range(eval_iters):
            xb, yb, mb, tb = get_batch(split)
            _, loss = model(xb, t=tb, targets=yb, mask=mb)
            split_losses[k] = loss.item()
        losses[split] = split_losses.mean().item()
    model.train()
    return losses


def save_loss_curves(eval_steps, train_losses, val_losses, output_path):
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib is not installed; skipping loss curve plot.")
        return

    plt.figure(figsize=(8, 5))
    plt.plot(eval_steps, train_losses, label="train masked loss", marker="o")
    plt.plot(eval_steps, val_losses, label="val masked loss", marker="o")
    plt.xlabel("step")
    plt.ylabel("loss")
    plt.title("Training and Validation Loss Curves")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()
    print(f"saved loss curves to {output_path}")


@torch.no_grad()
def evaluate_masked_metrics(model, split="val", num_batches=eval_iters):
    model.eval()
    total_loss = 0.0
    total_correct = 0
    total_masked = 0

    for _ in range(num_batches):
        xb, yb, mb, tb = get_batch(split)
        logits, loss = model(xb, t=tb, targets=yb, mask=mb)
        total_loss += loss.item()

        pred = torch.argmax(logits, dim=-1)
        total_correct += (pred[mb] == yb[mb]).sum().item()
        total_masked += mb.sum().item()

    avg_loss = total_loss / max(1, num_batches)
    masked_acc = total_correct / max(1, total_masked)
    perplexity = math.exp(avg_loss)

    model.train()
    return {
        "masked_loss": avg_loss,
        "masked_recon_acc": masked_acc,
        "perplexity": perplexity,
    }


@torch.no_grad()
def evaluate_entropy_per_timestep(model, split="val", batches_per_t=2):
    model.eval()
    entropies = []

    for t_value in range(1, T + 1):
        entropy_sum = 0.0
        entropy_count = 0
        for _ in range(batches_per_t):
            xb, _, mb, tb = get_batch_at_t(split, t_value)
            logits, _ = model(xb, t=tb)
            log_probs = F.log_softmax(logits, dim=-1)
            probs = log_probs.exp()
            token_entropy = -(probs * log_probs).sum(dim=-1)
            masked_entropy = token_entropy[mb]
            entropy_sum += masked_entropy.sum().item()
            entropy_count += masked_entropy.numel()
        entropies.append(entropy_sum / max(1, entropy_count))

    model.train()
    return entropies


@torch.no_grad()
def generate_with_trace(model, prompt_tokens, gen_len=128, temperature=0.0):
    model.eval()

    if len(prompt_tokens) == 0:
        raise ValueError("prompt_tokens cannot be empty")
    if len(prompt_tokens) >= block_size:
        raise ValueError("prompt_tokens length must be < block_size")

    max_gen = block_size - len(prompt_tokens)
    gen_len = max(1, min(gen_len, max_gen))
    total_len = len(prompt_tokens) + gen_len
    gen_slice = slice(len(prompt_tokens), total_len)

    x = torch.full((1, block_size), mask_token_id, device=device)
    x[0, : len(prompt_tokens)] = torch.tensor(prompt_tokens, device=device)

    fixed_mask = torch.ones((1, block_size), dtype=torch.bool, device=device)
    fixed_mask[:, gen_slice] = False

    states = [x[0, gen_slice].clone()]

    for t in reversed(range(1, T + 1)):
        t_tensor = torch.tensor([t], device=device)
        logits, _ = model(x, t=t_tensor)

        if temperature <= 0:
            sampled = torch.argmax(logits, dim=-1)
            probs = F.softmax(logits, dim=-1)
            sampled_conf = probs.gather(-1, sampled.unsqueeze(-1)).squeeze(-1)
        else:
            probs = F.softmax(logits / temperature, dim=-1)
            sampled = torch.multinomial(probs.view(-1, vocab_size), 1).view(1, block_size)
            sampled_conf = probs.gather(-1, sampled.unsqueeze(-1)).squeeze(-1)

        x = torch.where(fixed_mask, x, sampled)

        if t > 1:
            gen_positions = total_len - len(prompt_tokens)
            next_mask_ratio = 1.0 - survival_prob(t - 1)
            k = int(next_mask_ratio * gen_positions)
            if k > 0:
                conf = sampled_conf[:, gen_slice]
                low_conf_idx_local = torch.topk(conf, k=k, dim=1, largest=False).indices
                low_conf_idx = low_conf_idx_local + len(prompt_tokens)
                x.scatter_(1, low_conf_idx, mask_token_id)

        states.append(x[0, gen_slice].clone())

    t0 = torch.tensor([0], device=device)
    logits, _ = model(x, t=t0)
    final_tokens = torch.argmax(logits, dim=-1)
    x[:, gen_slice] = final_tokens[:, gen_slice]
    states.append(x[0, gen_slice].clone())

    change_rates = []
    for prev_state, next_state in zip(states[:-1], states[1:]):
        change_rate = (next_state != prev_state).float().mean().item()
        change_rates.append(change_rate)

    model.train()
    return x[0, gen_slice].tolist(), change_rates


@torch.no_grad()
def evaluate_generation_metrics(
    model,
    split="val",
    num_samples=16,
    prompt_len=32,
    gen_len=128,
    temperature=0.0,
):
    data_split = train_data if split == "train" else val_data
    needed = prompt_len + gen_len + 1
    if len(data_split) <= needed:
        raise ValueError(
            f"Not enough data for generation metrics: need > {needed}, got {len(data_split)}"
        )

    all_change_rates = []
    total_bigrams = 0
    unique_bigrams = set()

    for _ in range(num_samples):
        max_start = len(data_split) - needed
        start = torch.randint(max_start + 1, (1,)).item()
        prompt_tokens = data_split[start : start + prompt_len].tolist()

        gen_tokens, change_rates = generate_with_trace(
            model,
            prompt_tokens=prompt_tokens,
            gen_len=gen_len,
            temperature=temperature,
        )

        all_change_rates.extend(change_rates)

        if len(gen_tokens) >= 2:
            for i in range(len(gen_tokens) - 1):
                bg = (gen_tokens[i], gen_tokens[i + 1])
                unique_bigrams.add(bg)
                total_bigrams += 1

    distinct_2 = len(unique_bigrams) / max(1, total_bigrams)
    avg_change_rate = sum(all_change_rates) / max(1, len(all_change_rates))

    return {
        "reverse_step_token_change_rate": avg_change_rate,
        "distinct_2": distinct_2,
    }

# Reverse Diffusion Sampling
@torch.no_grad()
def generate(model, prompt_len=16):
    model.eval()

    x = torch.full((1, block_size), mask_token_id, device=device)
    x[0, :prompt_len] = data[:prompt_len].to(device)
    prompt_mask = torch.zeros((1, block_size), dtype=torch.bool, device=device)
    prompt_mask[:, :prompt_len] = True

    for t in reversed(range(1, T + 1)):
        t_tensor = torch.tensor([t], device=device)
        logits, _ = model(x, t=t_tensor)
        probs = F.softmax(logits, dim=-1)
        sampled = torch.multinomial(probs.view(-1, vocab_size), 1).view(1, block_size)
        sampled_conf = probs.gather(-1, sampled.unsqueeze(-1)).squeeze(-1)

        # Keep prompt fixed, update all generated positions each step.
        x = torch.where(prompt_mask, x, sampled)

        # Confidence-based remasking enables iterative refinement instead of one-shot fill.
        if t > 1:
            gen_positions = (~prompt_mask).sum().item()
            next_mask_ratio = 1.0 - survival_prob(t - 1)
            k = int(next_mask_ratio * gen_positions)
            if k > 0:
                conf = sampled_conf.masked_fill(prompt_mask, float("inf"))
                low_conf_idx = torch.topk(conf, k=k, dim=1, largest=False).indices
                x.scatter_(1, low_conf_idx, mask_token_id)

    # Explicit final denoise at t=0.
    t0 = torch.tensor([0], device=device)
    logits, _ = model(x, t=t0)
    final_tokens = torch.argmax(logits, dim=-1)
    x = torch.where(prompt_mask, x, final_tokens)

    return decode(x[0].tolist())

# Training
if __name__ == "__main__":
    os.makedirs(os.path.dirname(checkpoint_path), exist_ok=True)
    os.makedirs(os.path.dirname(loss_curve_path), exist_ok=True)

    model = Model().to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
    eval_steps = []
    train_loss_curve = []
    val_loss_curve = []

    for iter in range(max_iters):
        if iter % eval_interval == 0:
            losses = estimate_loss(model)
            eval_steps.append(iter)
            train_loss_curve.append(losses["train"])
            val_loss_curve.append(losses["val"])
            print(
                f"eval step {iter} | train masked loss {losses['train']:.4f} | "
                f"val masked loss {losses['val']:.4f}"
            )
            print("Generating sample...")
            print(generate(model))

        xb, yb, mb, tb = get_batch("train")
        logits, loss = model(xb, t=tb, targets=yb, mask=mb)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if iter > 0 and iter % save_interval == 0:
            torch.save(
                {
                    "iter": iter,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "loss": loss.item(),
                },
                checkpoint_path,
            )
            print(f"saved checkpoint to {checkpoint_path} at step {iter}")

        if iter % 100 == 0:
            print(f"step {iter} | loss {loss.item():.4f}")

    torch.save(
        {
            "iter": max_iters,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "loss": loss.item(),
        },
        checkpoint_path,
    )
    print(f"saved final checkpoint to {checkpoint_path}")
    save_loss_curves(eval_steps, train_loss_curve, val_loss_curve, loss_curve_path)

    final_core = evaluate_masked_metrics(model, split="val", num_batches=eval_iters)
    entropy_by_t = evaluate_entropy_per_timestep(model, split="val", batches_per_t=2)
    final_gen = evaluate_generation_metrics(
        model,
        split="val",
        num_samples=16,
        prompt_len=32,
        gen_len=128,
        temperature=0.0,
    )

    entropy_mean = sum(entropy_by_t) / max(1, len(entropy_by_t))
    entropy_t1 = entropy_by_t[0]
    entropy_tmid = entropy_by_t[(len(entropy_by_t) - 1) // 2]
    entropy_tT = entropy_by_t[-1]

    print("\n=== Final Evaluation Metrics (val) ===")
    print(f"Perplexity: {final_core['perplexity']:.4f}")
    print(f"Masked reconstruction accuracy: {final_core['masked_recon_acc']:.4f}")
    print(
        "Entropy per timestep (masked positions): "
        f"mean={entropy_mean:.4f}, t=1:{entropy_t1:.4f}, "
        f"t={1 + (T - 1) // 2}:{entropy_tmid:.4f}, t={T}:{entropy_tT:.4f}"
    )
    print(
        "Reverse-step token change rate: "
        f"{final_gen['reverse_step_token_change_rate']:.4f}"
    )
    print(f"Distinct-2 diversity (generated region): {final_gen['distinct_2']:.4f}")
