"""One dataset epoch; 20 detached, directly supervised recurrent transitions."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import platform
import random
import time
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader, TensorDataset
from torchvision.datasets import CIFAR10


class Block(nn.Module):
    def __init__(self, incoming, outgoing):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(incoming, outgoing, 3, padding=1),
            nn.GroupNorm(4, outgoing), nn.SiLU(),
            nn.Conv2d(outgoing, outgoing, 3, padding=1),
            nn.GroupNorm(4, outgoing), nn.SiLU(),
        )

    def forward(self, x):
        return self.net(x)


class Refiner(nn.Module):
    """Small U-Net, direct RGB prediction; no time, labels or output residual."""
    def __init__(self, width=24):
        super().__init__()
        self.enc1 = Block(3, width)
        self.enc2 = Block(width, 2 * width)
        self.middle = Block(2 * width, 4 * width)
        self.dec2 = Block(6 * width, 2 * width)
        self.dec1 = Block(3 * width, width)
        self.out = nn.Conv2d(width, 3, 1)

    def forward(self, x):
        a = self.enc1(x)
        b = self.enc2(F.avg_pool2d(a, 2))
        c = self.middle(F.avg_pool2d(b, 2))
        b2 = self.dec2(torch.cat((F.interpolate(c, size=b.shape[-2:], mode="nearest"), b), 1))
        a2 = self.dec1(torch.cat((F.interpolate(b2, size=a.shape[-2:], mode="nearest"), a), 1))
        return self.out(a2)


def train_trajectory(model, optimizer, y, steps, *, initial=None):
    x = torch.randn_like(y) if initial is None else initial.detach()
    losses = []
    for _ in range(steps):
        assert not x.requires_grad and x.grad_fn is None
        next_x = model(x)
        loss = F.mse_loss(next_x, y)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        # Carry exactly the pre-update forward output. Do not recompute it.
        x = next_x.detach()
        losses.append(loss.detach())
    values = torch.stack(losses).cpu().tolist()
    if not all(math.isfinite(v) for v in values):
        raise RuntimeError("Nonfinite training loss; stopping rather than changing the experiment.")
    return values


def save_grid(images, path, columns=8, scale=4):
    images = ((images.detach().cpu().clamp(-1, 1) + 1) * 127.5).round().byte()
    n, _, h, w = images.shape
    grid = Image.new("RGB", (columns * w, math.ceil(n / columns) * h), (235, 235, 235))
    for i, pixels in enumerate(images.permute(0, 2, 3, 1).numpy()):
        grid.paste(Image.fromarray(pixels), ((i % columns) * w, (i // columns) * h))
    grid.resize((grid.width * scale, grid.height * scale), Image.Resampling.NEAREST).save(path)


def save_trajectory(states, path):
    n = min(8, len(next(iter(states.values()))))
    tile, margin, label = 128, 8, 65
    canvas = Image.new("RGB", (label + n * (tile + margin), len(states) * (tile + 28)), "white")
    draw = ImageDraw.Draw(canvas)
    for row, (step, images) in enumerate(states.items()):
        draw.text((8, row * (tile + 28) + 58), f"x_{step}", fill="black")
        pixels = ((images[:n].clamp(-1, 1) + 1) * 127.5).round().byte().permute(0, 2, 3, 1).numpy()
        for col, pixel in enumerate(pixels):
            canvas.paste(Image.fromarray(pixel).resize((tile, tile), Image.Resampling.NEAREST),
                         (label + col * (tile + margin), row * (tile + 28) + 20))
    canvas.save(path)


@torch.no_grad()
def evaluate(model, noise, mean, steps, directory, seen):
    was_training = model.training
    model.eval()
    chosen = {0, 1, 2, 5, 10, steps}
    states, metrics = {}, []
    x = noise.clone()
    for step in range(steps + 1):
        if step in chosen:
            cpu = x.detach().cpu()
            if not torch.isfinite(cpu).all():
                raise RuntimeError(f"Nonfinite evaluation state at step {step}")
            states[step] = cpu
            metrics.append({
                "step": step,
                "pixel_std_across_samples": cpu.std(dim=0, unbiased=False).mean().item(),
                "spatial_std_within_samples": cpu.flatten(2).std(dim=2, unbiased=False).mean().item(),
                "mse_to_dataset_mean": F.mse_loss(cpu, mean.expand_as(cpu)).item(),
                "out_of_display_range_fraction": ((cpu < -1) | (cpu > 1)).float().mean().item(),
                "minimum": cpu.min().item(), "maximum": cpu.max().item(),
            })
        if step < steps:
            x = model(x)
    save_grid(states[steps], directory / f"samples_seen_{seen:05d}.png")
    save_trajectory(states, directory / f"trajectory_seen_{seen:05d}.png")
    record = {"images_seen": seen, "metrics": metrics}
    with (directory / "evaluation.jsonl").open("a") as stream:
        stream.write(json.dumps(record) + "\n")
    model.train(was_training)
    return record, states


def choose_device(requested):
    if requested != "auto":
        return torch.device(requested)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=Path("data"))
    parser.add_argument("--output", type=Path, default=Path("runs/baseline"))
    parser.add_argument("--steps", type=int, default=20)
    parser.add_argument("--resume", type=Path, help="Continue model and Adam state from a trusted local checkpoint")
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--width", type=int, default=24)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--eval-seed", type=int, default=1042)
    parser.add_argument("--eval-samples", type=int, default=64)
    parser.add_argument("--device", choices=["auto", "cpu", "mps", "cuda"], default="auto")
    parser.add_argument("--limit-images", type=int, default=0, help="Smoke test only; zero uses all 50,000 images")
    parser.add_argument("--log-every", type=int, default=10)
    args = parser.parse_args()
    if args.width <= 0 or args.width % 4 or min(args.steps, args.batch_size, args.eval_samples, args.log_every) <= 0:
        parser.error("width must be a positive multiple of 4; other counts must be positive")
    if args.limit_images < 0 or args.lr <= 0:
        parser.error("limit-images must be nonnegative and lr positive")
    args.output.mkdir(parents=True, exist_ok=True)
    if any(args.output.iterdir()):
        parser.error("output directory must be empty; choose a fresh run name")
    torch.set_num_threads(6)
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = choose_device(args.device)
    dataset = CIFAR10(str(args.data), train=True, download=True)
    data = torch.from_numpy(dataset.data).permute(0, 3, 1, 2)
    if args.limit_images:
        data = data[:args.limit_images]
    mean = torch.from_numpy(data.numpy().mean(axis=0, dtype=np.float64)).float().div(127.5).sub(1)
    loader = DataLoader(TensorDataset(data), batch_size=args.batch_size, shuffle=True,
                        generator=torch.Generator().manual_seed(args.seed), num_workers=0, drop_last=False)
    model = Refiner(args.width).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    parent_updates = 0
    parent_hash = None
    if args.resume:
        # Earlier runs stored torch.__version__ as a TorchVersion object.
        with torch.serialization.safe_globals([torch.torch_version.TorchVersion]):
            checkpoint = torch.load(args.resume, map_location=device, weights_only=True)
        if checkpoint["config"]["width"] != args.width:
            raise ValueError("Checkpoint width differs from requested model width")
        model.load_state_dict(checkpoint["model"], strict=True)
        optimizer.load_state_dict(checkpoint["optimizer"])
        for group in optimizer.param_groups:
            group["lr"] = args.lr
        parent_updates = checkpoint["optimizer_updates"]
        parent_hash = hashlib.sha256(args.resume.read_bytes()).hexdigest()
    noise = torch.randn(args.eval_samples, 3, 32, 32, generator=torch.Generator().manual_seed(args.eval_seed)).to(device)
    config = {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()}
    config.update(device=str(device), dataset="CIFAR-10 train", images=len(data), epochs=1,
                  optimizer="Adam", weight_decay=0, parameters=sum(p.numel() for p in model.parameters()),
                  torch_version=str(torch.__version__), python=platform.python_version(),
                  source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                  state_timing="pre-update output carried detached; optimizer update after every transition",
                  target_scaling="uint8 / 127.5 - 1", input_noise="N(0,I), fresh per minibatch",
                  output_constraints="none; clamp only for PNG rendering",
                  parent_checkpoint_sha256=parent_hash, parent_optimizer_updates=parent_updates,
                  optimizer_state_restored=bool(args.resume))
    (args.output / "config.json").write_text(json.dumps(config, indent=2) + "\n")
    (args.output / "train.py").write_text(Path(__file__).read_text())
    save_grid(data[:64].float().div(127.5).sub(1), args.output / "real_images.png")
    save_grid(mean.unsqueeze(0), args.output / "dataset_mean.png", columns=1, scale=8)
    real = data[:1024].float().div(127.5).sub(1)
    real_std = real.std(dim=0, unbiased=False).mean().item()
    print(json.dumps(config), flush=True)
    initial_eval, initial_states = evaluate(model, noise, mean, args.steps, args.output, 0)
    torch.save(initial_states, args.output / "initial_evaluation_states.pt")
    milestones = {math.ceil(len(loader) * q / 4) for q in range(1, 5)}
    if args.resume:
        milestones |= {b for b in (1, 10, 50) if b <= len(loader)}
    seen = 0
    started = time.monotonic()
    totals = np.zeros(args.steps, dtype=np.float64)
    with (args.output / "training.csv").open("w", buffering=1) as stream:
        writer = csv.writer(stream)
        writer.writerow(["batch", "images_seen", "optimizer_updates", "elapsed_seconds"] + [f"mse_step_{i}" for i in range(1, args.steps + 1)])
        for batch, (raw,) in enumerate(loader, start=1):
            model.train()
            y = raw.to(device=device, dtype=torch.float32).div_(127.5).sub_(1)
            losses = train_trajectory(model, optimizer, y, args.steps)
            seen += len(y)
            totals += np.array(losses) * len(y)
            elapsed = time.monotonic() - started
            writer.writerow([batch, seen, batch * args.steps, elapsed] + losses)
            if batch == 1 or batch % args.log_every == 0 or batch == len(loader):
                print(json.dumps({"batch": batch, "batches": len(loader), "images_seen": seen,
                                  "updates": batch * args.steps, "elapsed_seconds": round(elapsed, 1),
                                  "eta_seconds": round(elapsed / batch * (len(loader) - batch), 1),
                                  "loss_first": losses[0], "loss_last": losses[-1]}), flush=True)
            if batch in milestones:
                final_eval, final_states = evaluate(model, noise, mean, args.steps, args.output, seen)
                torch.save({"model": model.state_dict(), "optimizer": optimizer.state_dict(), "config": config, "images_seen": seen,
                            "optimizer_updates": parent_updates + batch * args.steps},
                           args.output / f"checkpoint_seen_{seen:05d}.pt")
                print(json.dumps({"evaluation": final_eval}), flush=True)
    assert seen == len(data)
    checkpoint = {"model": model.state_dict(), "optimizer": optimizer.state_dict(), "config": config,
                  "images_seen": seen, "optimizer_updates": parent_updates + len(loader) * args.steps}
    torch.save(checkpoint, args.output / "checkpoint.pt")
    torch.save(final_states, args.output / "evaluation_states.pt")
    save_grid(final_states[args.steps], args.output / "samples_final.png")
    save_trajectory(final_states, args.output / "trajectory_final.png")
    summary = {"supervised_image_pairs": seen * args.steps,
               "parent_optimizer_updates": parent_updates,
               "cumulative_optimizer_updates": parent_updates + len(loader) * args.steps,
               "images_seen": seen, "optimizer_updates": len(loader) * args.steps,
               "elapsed_seconds": time.monotonic() - started,
               "mean_training_mse_by_step": (totals / seen).tolist(),
               "real_pixel_std_across_samples": real_std, "real_reference_samples": len(real),
               "initial_evaluation": initial_eval, "final_evaluation": final_eval}
    (args.output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps({"complete": str(args.output), "summary": summary}), flush=True)


if __name__ == "__main__":
    main()
