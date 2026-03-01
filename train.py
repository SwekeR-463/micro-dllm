import os
import sys
import time
import math
import torch
import torch.nn as nn
from torch.nn import functional as F


# Hyperparameters
batch_size = 64
block_size = 256
max_iters = 10000
eval_interval = 500
learning_rate = 3e-4
eval_iters = 200
save_interval = 500
checkpoint_path = "model_stories.pt"
loss_curve_path = "loss_curves.png"

n_embd = 384
n_head = 6
n_layer = 6
head_dim = n_embd // n_head

T = 30  # diffusion steps

device = (
    "cuda"
    if torch.cuda.is_available()
    else ("mps" if torch.backends.mps.is_available() else "cpu")
)

torch.manual_seed(1337)

# Data
with open("stories.txt", "r", encoding="utf-8") as f:
    text = f.read()

chars = sorted(list(set(text)))
chars = ["_"] + chars  # "_" is MASK
vocab_size = len(chars)

stoi = {ch: i for i, ch in enumerate(chars)}
itos = {i: ch for i, ch in enumerate(chars)}

mask_token_id = stoi["_"]

def encode(s):
    return [stoi[ch] for ch in s]

def decode(l):
    return "".join([itos[i] for i in l])

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
