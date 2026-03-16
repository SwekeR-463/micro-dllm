import os
import sys
import time
import math
from muon import SingleDeviceMuonWithAuxAdam
import torch
import torch.nn as nn
import torch.distributed as dist
from torch.nn import functional as F
from torch.nn.parallel import DistributedDataParallel as DDP
from utils.tokenizer_utils import load_tokenizer

# Hyperparameters
batch_size = 64
block_size = 256
max_iters = 100000
eval_interval = 500
learning_rate = 3e-4
eval_iters = 200
save_interval = 500
loss_curve_every = 1
checkpoint_path = "artifacts/models/model_stories_10k_256_muon.pt"
loss_curve_path = "artifacts/media/new_loss_curves_muon.png"

n_embd = 384
n_head = 6
n_layer = 6
head_dim = n_embd // n_head

T = 100  # diffusion steps
tokenizer_path = "artifacts/tokenizer/tokenizer.json"
stories_path = "data/stories.txt"

def setup_distributed():
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    rank = int(os.environ.get("RANK", "0"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    distributed = world_size > 1

    if distributed:
        if torch.cuda.is_available():
            backend = "nccl"
            torch.cuda.set_device(local_rank)
            device = torch.device("cuda", local_rank)
        else:
            backend = "gloo"
            device = torch.device("cpu")
        dist.init_process_group(backend=backend, rank=rank, world_size=world_size)
    else:
        device = torch.device(
            "cuda"
            if torch.cuda.is_available()
            else ("mps" if torch.backends.mps.is_available() else "cpu")
        )

    return {
        "distributed": distributed,
        "world_size": world_size,
        "rank": rank,
        "local_rank": local_rank,
        "is_master": rank == 0,
        "device": device,
    }


dist_info = setup_distributed()
distributed = dist_info["distributed"]
world_size = dist_info["world_size"]
rank = dist_info["rank"]
local_rank = dist_info["local_rank"]
is_master = dist_info["is_master"]
device = dist_info["device"]

if batch_size % world_size != 0:
    raise ValueError(
        f"batch_size ({batch_size}) must be divisible by world_size ({world_size})"
    )
local_batch_size = batch_size // world_size

torch.manual_seed(1337 + rank)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(1337 + rank)

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
    idx = torch.randint(len(data_split) - block_size, (local_batch_size,))
    x0 = torch.stack([data_split[i : i + block_size] for i in idx])

    t = torch.randint(1, T + 1, (local_batch_size,))

    xt = x0.clone()
    mask = torch.zeros_like(x0, dtype=torch.bool)

    for i in range(local_batch_size):
        a_t = survival_prob(t[i].item())
        token_mask = torch.rand(block_size) > a_t
        if not token_mask.any():
            token_mask[torch.randint(block_size, (1,)).item()] = True
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
    idx = torch.randint(len(data_split) - block_size, (local_batch_size,))
    x0 = torch.stack([data_split[i : i + block_size] for i in idx])

    t_value = int(t_value)
    t = torch.full((local_batch_size,), t_value, dtype=torch.long)
    a_t = survival_prob(t_value)

    token_mask = torch.rand(local_batch_size, block_size) > a_t
    empty_rows = ~token_mask.any(dim=1)
    if empty_rows.any():
        row_ids = empty_rows.nonzero(as_tuple=False).flatten()
        rand_cols = torch.randint(block_size, (row_ids.numel(),))
        token_mask[row_ids, rand_cols] = True
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
            # MDLM objective: CE only on tokens masked at timestep t.
            if mask is None:
                raise ValueError("mask is required when targets are provided.")
            masked_logits = logits[mask]
            masked_targets = targets[mask]
            loss = F.cross_entropy(masked_logits, masked_targets)

        return logits, loss


def unwrap_model(model):
    return model.module if isinstance(model, DDP) else model


def dist_mean_scalar(value):
    if not distributed:
        return float(value)
    t = torch.tensor(float(value), device=device)
    dist.all_reduce(t, op=dist.ReduceOp.SUM)
    t /= world_size
    return t.item()


def dist_mean_dict(values):
    if not distributed:
        return values
    return {k: dist_mean_scalar(v) for k, v in values.items()}


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
    plt.plot(eval_steps, train_losses, label="train masked loss", linewidth=1.2)
    plt.plot(eval_steps, val_losses, label="val masked loss", linewidth=1.2)
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
def estimate_step_loss(model, split="val"):
    model.eval()
    xb, yb, mb, tb = get_batch(split)
    _, loss = model(xb, t=tb, targets=yb, mask=mb)
    model.train()
    return loss.item()


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
    try:
        if is_master:
            os.makedirs(os.path.dirname(checkpoint_path), exist_ok=True)
            os.makedirs(os.path.dirname(loss_curve_path), exist_ok=True)
            if distributed:
                print(
                    "Distributed mode enabled "
                    f"(world_size={world_size}, global_batch_size={batch_size}, "
                    f"per_gpu_batch_size={local_batch_size})."
                )

        model = Model().to(device)
        if distributed:
            ddp_device_ids = [local_rank] if device.type == "cuda" else None
            model = DDP(model, device_ids=ddp_device_ids)

        optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
        # hidden_weights = [p for p in model.blocks.parameters() if p.ndim >= 2]
        # hidden_gains_biases = [p for p in model.blocks.parameters() if p.ndim < 2]
        # nonhidden_params = [
        #     *model.lm_head.parameters(),
        #     *model.token_emb.parameters(),
        #     *model.timestep_emb.parameters(),
        # ]
        # param_groups = [
        #     dict(params=hidden_weights, use_muon=True,
        #         lr=0.02, weight_decay=0.01),
        #     dict(params=hidden_gains_biases+nonhidden_params, use_muon=False,
        #         lr=3e-4, betas=(0.9, 0.95), weight_decay=0.01),
        # ]
        # optimizer = SingleDeviceMuonWithAuxAdam(param_groups)

        curve_steps = []
        train_loss_curve = []
        val_loss_curve = []
        avg_train_loss = float("nan")

        for iter in range(max_iters):
            if iter % eval_interval == 0:
                losses = dist_mean_dict(estimate_loss(model))
                if is_master:
                    curve_steps.append(iter)
                    train_loss_curve.append(losses["train"])
                    val_loss_curve.append(losses["val"])
                    print(
                        f"eval step {iter} | train masked loss {losses['train']:.4f} | "
                        f"val masked loss {losses['val']:.4f}"
                    )
                    print("Generating sample...")
                    print(generate(model))
                if distributed:
                    dist.barrier()

            xb, yb, mb, tb = get_batch("train")
            _, loss = model(xb, t=tb, targets=yb, mask=mb)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            avg_train_loss = dist_mean_scalar(loss.detach().item())

            if iter % loss_curve_every == 0 and is_master:
                curve_steps.append(iter)
                train_loss_curve.append(avg_train_loss)
                if not distributed:
                    val_loss_curve.append(estimate_step_loss(model, split="val"))
                else:
                    val_loss_curve.append(float("nan"))

            if iter > 0 and iter % save_interval == 0 and is_master:
                torch.save(
                    {
                        "iter": iter,
                        "model_state_dict": unwrap_model(model).state_dict(),
                        "optimizer_state_dict": optimizer.state_dict(),
                        "loss": avg_train_loss,
                    },
                    checkpoint_path,
                )
                print(f"saved checkpoint to {checkpoint_path} at step {iter}")

            if iter % 100 == 0 and is_master:
                print(f"step {iter} | loss {avg_train_loss:.4f}")

        if distributed:
            dist.barrier()

        if is_master:
            torch.save(
                {
                    "iter": max_iters,
                    "model_state_dict": unwrap_model(model).state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "loss": avg_train_loss,
                },
                checkpoint_path,
            )
            print(f"saved final checkpoint to {checkpoint_path}")
            save_loss_curves(curve_steps, train_loss_curve, val_loss_curve, loss_curve_path)

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

        if distributed:
            dist.barrier()

    finally:
        if distributed and dist.is_initialized():
            dist.destroy_process_group()
