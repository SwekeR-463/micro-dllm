import argparse
import os
import tempfile
import subprocess
import torch
import torch.nn.functional as F
from PIL import Image, ImageDraw, ImageFont

import train as tr


def load_model(checkpoint_path: str) -> tr.Model:
    ckpt = torch.load(checkpoint_path, map_location=tr.device)
    if isinstance(ckpt, dict) and "model_state_dict" in ckpt:
        state_dict = ckpt["model_state_dict"]
    else:
        state_dict = ckpt

    model = tr.Model().to(tr.device)
    model.load_state_dict(state_dict)
    model.eval()
    return model


def write_diffusion_gif(
    trace,
    prompt_len: int,
    output_path: str,
    frame_ms: int = 200,
    chars_per_line: int = 96,
) -> None:
    font = ImageFont.load_default()
    pad = 20
    line_h = 18
    header_h = 70
    char_w = 8
    max_lines = max(1, (max(len(item["text"]) for item in trace) + chars_per_line - 1) // chars_per_line)
    width = pad * 2 + chars_per_line * char_w
    height = header_h + pad * 2 + max_lines * line_h

    frames = []
    for item in trace:
        im = Image.new("RGB", (width, height), "#faf8f2")
        draw = ImageDraw.Draw(im)

        draw.text((pad, 14), "Diffusion Inference Trace", fill="#111827", font=font)
        draw.text(
            (pad, 34),
            f"Step: {item['label']}    Active masks: {item['masked']}",
            fill="#374151",
            font=font,
        )

        text = item["text"]
        for i, ch in enumerate(text):
            row = i // chars_per_line
            col = i % chars_per_line
            x = pad + col * char_w
            y = header_h + row * line_h

            if i < prompt_len:
                draw.text((x, y), ch, fill="#0f766e", font=font)
            elif ch == "_":
                draw.rectangle([x - 1, y - 1, x + char_w, y + line_h - 4], fill="#fef08a")
                draw.text((x, y), ch, fill="#713f12", font=font)
            else:
                draw.text((x, y), ch, fill="#111827", font=font)

        frames.append(im)

    frames[0].save(
        output_path,
        save_all=True,
        append_images=frames[1:],
        duration=max(20, frame_ms),
        loop=0,
    )


def write_diffusion_video(
    trace,
    prompt_len: int,
    output_path: str,
    frame_ms: int = 180,
    chars_per_line: int = 120,
) -> None:
    width, height = 1280, 720
    fps = max(1, int(round(1000.0 / max(20, frame_ms))))

    font_paths = [
        "/usr/share/fonts/truetype/dejavu/DejaVuSansMono.ttf",
        "/usr/share/fonts/truetype/liberation2/LiberationMono-Regular.ttf",
    ]
    font_title = None
    font_meta = None
    font_mono = None
    for path in font_paths:
        if os.path.exists(path):
            font_title = ImageFont.truetype(path, 40)
            font_meta = ImageFont.truetype(path, 22)
            font_mono = ImageFont.truetype(path, 24)
            break
    if font_title is None or font_meta is None or font_mono is None:
        font_title = ImageFont.load_default()
        font_meta = ImageFont.load_default()
        font_mono = ImageFont.load_default()

    def parse_t(label: str) -> int:
        if label == "init":
            return tr.T
        if label.startswith("t="):
            try:
                return int(label.split("=", 1)[1])
            except ValueError:
                return tr.T
        return tr.T

    mask_history = [item["masked"] for item in trace]
    max_masks = max(mask_history) if mask_history else 1

    # Layout
    margin = 44
    header_h = 138
    meta_h = 74
    text_top = margin + header_h + meta_h + 16
    text_h = height - text_top - margin
    left_w = int(width * 0.72)
    right_w = width - (2 * margin + left_w + 16)
    left_x0 = margin
    left_x1 = left_x0 + left_w
    right_x0 = left_x1 + 16
    right_x1 = right_x0 + right_w

    # Typography metrics
    char_w = max(8, font_mono.getbbox("M")[2] - font_mono.getbbox("M")[0])
    line_h = max(24, font_mono.getbbox("Ag")[3] - font_mono.getbbox("Ag")[1] + 8)
    inner_pad = 22
    max_chars = max(20, min(chars_per_line, (left_w - 2 * inner_pad) // char_w))
    max_lines = max(4, (text_h - 2 * inner_pad) // line_h)

    with tempfile.TemporaryDirectory(prefix="diff_trace_") as tmpdir:
        for frame_idx, item in enumerate(trace):
            im = Image.new("RGB", (width, height), "#f7f7f5")
            draw = ImageDraw.Draw(im)

            # Panels
            draw.rounded_rectangle(
                [margin, margin, width - margin, margin + header_h],
                radius=20,
                fill=(255, 255, 255),
                outline=(214, 214, 214),
                width=1,
            )
            draw.rounded_rectangle(
                [margin, margin + header_h + 10, width - margin, margin + header_h + meta_h + 10],
                radius=16,
                fill=(252, 252, 252),
                outline=(220, 220, 220),
                width=1,
            )
            draw.rounded_rectangle(
                [left_x0, text_top, left_x1, text_top + text_h],
                radius=18,
                fill=(255, 255, 255),
                outline=(220, 220, 220),
                width=1,
            )
            draw.rounded_rectangle(
                [right_x0, text_top, right_x1, text_top + text_h],
                radius=18,
                fill=(255, 255, 255),
                outline=(220, 220, 220),
                width=1,
            )

            # Header text
            draw.text((margin + 24, margin + 20), "Diffusion Decoding", fill=(25, 25, 25), font=font_title)

            # Progress bar
            current_t = parse_t(item["label"])
            progress = min(1.0, max(0.0, 1.0 - (current_t / max(1, tr.T))))
            bar_x0 = width - margin - 430
            bar_y0 = margin + 44
            bar_x1 = width - margin - 26
            bar_y1 = bar_y0 + 22
            draw.rounded_rectangle([bar_x0, bar_y0, bar_x1, bar_y1], radius=11, fill=(234, 234, 234))
            fill_w = int((bar_x1 - bar_x0) * progress)
            if fill_w > 0:
                draw.rounded_rectangle([bar_x0, bar_y0, bar_x0 + fill_w, bar_y1], radius=11, fill=(65, 133, 243))
            draw.text((bar_x0, bar_y1 + 10), f"progress {int(progress * 100):3d}%", fill=(95, 95, 95), font=font_meta)

            # Meta row
            draw.text(
                (margin + 24, margin + header_h + 30),
                f"step {item['label']}    active_masks {item['masked']}    total_steps {tr.T}",
                fill=(60, 60, 60),
                font=font_meta,
            )

            text = item["text"]
            max_chars_total = max_chars * max_lines
            if len(text) > max_chars_total:
                text = text[: max_chars_total - 3] + "..."

            # Token grid
            for i, ch in enumerate(text):
                row = i // max_chars
                col = i % max_chars
                x = left_x0 + inner_pad + col * char_w
                y = text_top + inner_pad + row * line_h

                if i < prompt_len:
                    draw.text((x, y), ch, fill=(45, 212, 191), font=font_mono)
                elif ch == "_":
                    draw.rounded_rectangle(
                        [x - 2, y - 2, x + char_w + 2, y + line_h - 6],
                        radius=4,
                        fill=(252, 214, 139),
                    )
                    draw.text((x, y), ch, fill=(70, 60, 40), font=font_mono)
                else:
                    draw.text((x, y), ch, fill=(30, 30, 30), font=font_mono)

            # Mask-count mini chart
            chart_pad = 18
            cx0 = right_x0 + chart_pad
            cy0 = text_top + chart_pad + 16
            cx1 = right_x1 - chart_pad
            cy1 = text_top + text_h - chart_pad - 24
            draw.text((cx0, text_top + chart_pad - 4), "Mask Count Timeline", fill=(70, 70, 70), font=font_meta)
            draw.rectangle([cx0, cy0, cx1, cy1], outline=(195, 195, 195), width=1)

            if len(mask_history) > 1:
                points = []
                for j, m in enumerate(mask_history):
                    tx = cx0 + int((cx1 - cx0) * (j / (len(mask_history) - 1)))
                    ty = cy1 - int((cy1 - cy0) * (m / max(1, max_masks)))
                    points.append((tx, ty))
                for p0, p1 in zip(points[:-1], points[1:]):
                    draw.line([p0, p1], fill=(65, 133, 243), width=3)

                cur_x = cx0 + int((cx1 - cx0) * (frame_idx / max(1, len(mask_history) - 1)))
                cur_y = cy1 - int((cy1 - cy0) * (item["masked"] / max(1, max_masks)))
                draw.ellipse([cur_x - 6, cur_y - 6, cur_x + 6, cur_y + 6], fill=(230, 129, 57))
            draw.text((cx0, cy1 + 8), f"current masks: {item['masked']}", fill=(95, 95, 95), font=font_meta)

            frame_path = os.path.join(tmpdir, f"frame_{frame_idx:05d}.png")
            im.save(frame_path)

        cmd = [
            "ffmpeg",
            "-y",
            "-framerate",
            str(fps),
            "-i",
            os.path.join(tmpdir, "frame_%05d.png"),
            "-vcodec",
            "libx264",
            "-crf",
            "18",
            "-preset",
            "slow",
            "-pix_fmt",
            "yuv420p",
            output_path,
        ]
        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode != 0:
            raise RuntimeError(f"ffmpeg failed:\n{result.stderr}")


@torch.no_grad()
def generate_from_prompt(
    model: tr.Model,
    prompt: str,
    gen_len: int = 128,
    temperature: float = 1.0,
    capture_trace: bool = False,
    trace_every: int = 1,
):
    prompt_tokens = tr.encode(prompt)
    if len(prompt_tokens) == 0:
        raise ValueError("Prompt cannot be empty.")
    if len(prompt_tokens) >= tr.block_size:
        raise ValueError(f"Prompt too long. Max length is {tr.block_size - 1} chars.")

    max_gen = tr.block_size - len(prompt_tokens)
    gen_len = max(1, min(gen_len, max_gen))
    total_len = len(prompt_tokens) + gen_len

    x = torch.full((1, tr.block_size), tr.mask_token_id, device=tr.device)
    x[0, : len(prompt_tokens)] = torch.tensor(prompt_tokens, device=tr.device)

    prompt_mask = torch.zeros((1, tr.block_size), dtype=torch.bool, device=tr.device)
    prompt_mask[:, : len(prompt_tokens)] = True
    gen_slice = slice(len(prompt_tokens), total_len)
    trace = []

    if capture_trace:
        trace.append(
            {
                "label": "init",
                "masked": int((x[0, gen_slice] == tr.mask_token_id).sum().item()),
                "text": tr.decode(x[0, :total_len].tolist()),
            }
        )

    for t in reversed(range(1, tr.T + 1)):
        t_tensor = torch.tensor([t], device=tr.device)
        logits, _ = model(x, t=t_tensor)

        if temperature <= 0:
            sampled = torch.argmax(logits, dim=-1)
            probs = F.softmax(logits, dim=-1)
            sampled_conf = probs.gather(-1, sampled.unsqueeze(-1)).squeeze(-1)
        else:
            probs = F.softmax(logits / temperature, dim=-1)
            sampled = torch.multinomial(probs.view(-1, tr.vocab_size), 1).view(
                1, tr.block_size
            )
            sampled_conf = probs.gather(-1, sampled.unsqueeze(-1)).squeeze(-1)

        x = torch.where(prompt_mask, x, sampled)

        if t > 1:
            gen_positions = (~prompt_mask).sum().item()
            next_mask_ratio = 1.0 - tr.survival_prob(t - 1)
            k = int(next_mask_ratio * gen_positions)
            if k > 0:
                conf = sampled_conf.masked_fill(prompt_mask, float("inf"))
                low_conf_idx = torch.topk(conf, k=k, dim=1, largest=False).indices
                x.scatter_(1, low_conf_idx, tr.mask_token_id)

        if capture_trace and (t % trace_every == 0 or t == 1):
            trace.append(
                {
                    "label": f"t={t}",
                    "masked": int((x[0, gen_slice] == tr.mask_token_id).sum().item()),
                    "text": tr.decode(x[0, :total_len].tolist()),
                }
            )

    t0 = torch.tensor([0], device=tr.device)
    logits, _ = model(x, t=t0)
    final_tokens = torch.argmax(logits, dim=-1)
    x = torch.where(prompt_mask, x, final_tokens)

    output = tr.decode(x[0, :total_len].tolist())
    if capture_trace:
        trace.append(
            {
                "label": "t=0",
                "masked": int((x[0, gen_slice] == tr.mask_token_id).sum().item()),
                "text": output,
            }
        )

    return output, trace, len(prompt_tokens)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str, default="model.pt")
    parser.add_argument("--prompt", type=str, required=True)
    parser.add_argument("--gen-len", type=int, default=128)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=1337)
    parser.add_argument("--viz-gif", type=str, default="")
    parser.add_argument("--viz-video", type=str, default="")
    parser.add_argument("--trace-every", type=int, default=10)
    parser.add_argument("--gif-frame-ms", type=int, default=180)
    args = parser.parse_args()

    if not args.viz_gif and not args.viz_video:
        raise ValueError("Provide at least one output: --viz-video and/or --viz-gif")

    torch.manual_seed(args.seed)

    try:
        _ = tr.encode(args.prompt)
    except KeyError as e:
        ch = e.args[0]
        raise ValueError(
            f"Prompt contains out-of-vocabulary char {repr(ch)}. "
            "Use only characters present in data.txt."
        ) from e

    model = load_model(args.checkpoint)
    out, trace, prompt_len = generate_from_prompt(
        model,
        prompt=args.prompt,
        gen_len=args.gen_len,
        temperature=args.temperature,
        capture_trace=True,
        trace_every=max(1, args.trace_every),
    )
    print(out)
    if args.viz_gif:
        write_diffusion_gif(
            trace, prompt_len, args.viz_gif, frame_ms=args.gif_frame_ms
        )
        print(f"saved diffusion gif: {args.viz_gif}")
    if args.viz_video:
        write_diffusion_video(
            trace, prompt_len, args.viz_video, frame_ms=args.gif_frame_ms
        )
        print(f"saved diffusion video: {args.viz_video}")


if __name__ == "__main__":
    main()
