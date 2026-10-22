"""Fixed noise/image pairs with an epoch-by-epoch recurrent curriculum."""
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
from torch.nn import functional as F
from torch.utils.data import DataLoader, TensorDataset
from torchvision.datasets import CIFAR10

from train import Refiner, choose_device, save_grid, train_trajectory


def make_fixed_noise(count, seed):
    return torch.randn(count, 3, 32, 32, generator=torch.Generator().manual_seed(seed))


def noise_digest(noise):
    digest = hashlib.sha256()
    for chunk in noise.split(1024):
        digest.update(chunk.numpy().tobytes())
    return digest.hexdigest()


def save_rows(rows, path):
    tile, gap, label = 112, 8, 90
    count = min(8, len(next(iter(rows.values()))))
    canvas = Image.new('RGB', (label + count * (tile + gap), len(rows) * (tile + 24)), 'white')
    draw = ImageDraw.Draw(canvas)
    for row, (name, images) in enumerate(rows.items()):
        draw.text((8, row * (tile + 24) + 55), name, fill='black')
        pixels = ((images[:count].cpu().clamp(-1, 1) + 1) * 127.5).round().byte().permute(0, 2, 3, 1).numpy()
        for col, pixel in enumerate(pixels):
            canvas.paste(Image.fromarray(pixel).resize((tile, tile), Image.Resampling.NEAREST),
                         (label + col * (tile + gap), row * (tile + 24) + 12))
    canvas.save(path)


@torch.no_grad()
def evaluate_fixed(model, noise, mean, chosen, targets=None):
    was_training = model.training
    model.eval()
    x = noise.clone()
    states, metrics = {}, []
    for step in range(max(chosen) + 1):
        if step in chosen:
            cpu = x.cpu()
            if not torch.isfinite(cpu).all():
                raise RuntimeError(f'Nonfinite evaluation at step {step}')
            states[step] = cpu
            row = {
                'step': step,
                'pixel_std_across_samples': cpu.std(0, unbiased=False).mean().item(),
                'spatial_std_within_samples': cpu.flatten(2).std(2, unbiased=False).mean().item(),
                'mse_to_dataset_mean': F.mse_loss(cpu, mean.expand_as(cpu)).item(),
                'out_of_display_range_fraction': ((cpu < -1) | (cpu > 1)).float().mean().item(),
                'minimum': cpu.min().item(), 'maximum': cpu.max().item(),
            }
            if targets is not None:
                row['mse_to_paired_target'] = F.mse_loss(cpu, targets).item()
                row['mse_to_permuted_target'] = F.mse_loss(cpu, targets.roll(1, dims=0)).item()
            metrics.append(row)
        if step < max(chosen):
            x = model(x)
    model.train(was_training)
    return metrics, states


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--schedule', type=int, nargs='+', default=[1, 2, 3, 5, 8])
    parser.add_argument('--data', type=Path, default=Path('data'))
    parser.add_argument('--output', type=Path, default=Path('runs/cifar10-fixed-progressive'))
    parser.add_argument('--batch-size', type=int, default=128)
    parser.add_argument('--width', type=int, default=24)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--pair-seed', type=int, default=2042)
    parser.add_argument('--eval-seed', type=int, default=1042)
    parser.add_argument('--eval-samples', type=int, default=64)
    parser.add_argument('--eval-steps', type=int, default=20)
    parser.add_argument('--device', choices=['auto', 'cpu', 'mps', 'cuda'], default='auto')
    parser.add_argument('--limit-images', type=int, default=0)
    parser.add_argument('--log-every', type=int, default=50)
    args = parser.parse_args()
    if min(*args.schedule, args.batch_size, args.eval_samples, args.eval_steps, args.log_every) < 1:
        parser.error('counts and transition depths must be positive')
    if args.width <= 0 or args.width % 4 or args.limit_images < 0 or args.lr <= 0:
        parser.error('invalid width, image limit, or learning rate')
    if args.eval_steps < max(args.schedule):
        parser.error('eval-steps must cover the training schedule')
    if args.pair_seed == args.eval_seed:
        parser.error('pair-seed and eval-seed must differ')
    args.output.mkdir(parents=True, exist_ok=True)
    if any(args.output.iterdir()):
        parser.error('output directory must be empty')
    torch.set_num_threads(6)
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = choose_device(args.device)
    raw = CIFAR10(str(args.data), train=True, download=True).data
    data = torch.from_numpy(raw).permute(0, 3, 1, 2)
    if args.limit_images:
        data = data[:args.limit_images]
    # Index by original dataset index, never by shuffled batch position.
    fixed_noise = make_fixed_noise(len(data), args.pair_seed)
    pair_hash = noise_digest(fixed_noise)
    loader = DataLoader(TensorDataset(torch.arange(len(data))), batch_size=args.batch_size,
                        shuffle=True, generator=torch.Generator().manual_seed(args.seed),
                        num_workers=0, drop_last=False)
    mean = torch.from_numpy(data.numpy().mean(0, dtype=np.float64)).float().div(127.5).sub(1)
    model = Refiner(args.width).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    # Random training subset; exactly the same original indices at every checkpoint.
    eval_ids = torch.randperm(len(data), generator=torch.Generator().manual_seed(args.eval_seed + 1))[:args.eval_samples]
    train_noise = fixed_noise[eval_ids].to(device)
    targets = data[eval_ids].float().div(127.5).sub(1)
    unseen_noise = make_fixed_noise(len(eval_ids), args.eval_seed).to(device)
    chosen = sorted({0, 1, 2, 3, 5, 8, 10, args.eval_steps, *args.schedule} & set(range(args.eval_steps + 1)))
    total_updates = len(loader) * sum(args.schedule)
    config = {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()}
    config.update(device=str(device), unique_images=len(data), epochs=len(args.schedule),
                  total_image_visits=len(data) * len(args.schedule), expected_optimizer_updates=total_updates,
                  parameters=sum(p.numel() for p in model.parameters()), optimizer='Adam',
                  betas=[0.9, 0.999], epsilon=1e-8, weight_decay=0,
                  torch_version=torch.__version__, python=platform.python_version(),
                  pair_noise_sha256=pair_hash, eval_indices=eval_ids.tolist(),
                  initialization='fresh model, seed 42 by default; no baseline checkpoint',
                  pairing='fixed CPU Gaussian tensor indexed by original CIFAR-10 training index',
                  state_timing='pre-update output carried detached; optimizer update every transition',
                  target_scaling='uint8 / 127.5 - 1', output_constraints='none; rendering clips only',
                  model_source_sha256=hashlib.sha256(Path(__file__).with_name('train.py').read_bytes()).hexdigest(),
                  source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
    (args.output / 'config.json').write_text(json.dumps(config, indent=2) + '\n')
    for name in ('train.py', 'train_fixed.py'):
        (args.output / name).write_text(Path(__file__).with_name(name).read_text())
    save_grid(targets, args.output / 'paired_training_targets.png')
    save_grid(mean.unsqueeze(0), args.output / 'dataset_mean.png', columns=1, scale=8)
    real = data[:1024].float().div(127.5).sub(1)
    real_std = real.std(0, unbiased=False).mean().item()
    mean_target_mse = F.mse_loss(mean.expand_as(targets), targets).item()
    print(json.dumps(config), flush=True)
    evaluations = []

    def assess(epoch, depth, visits, updates):
        entry = {'epoch': epoch, 'trained_depth': depth, 'image_visits': visits, 'optimizer_updates': updates}
        all_states = {}
        for kind, noise, target in [('train_pairs', train_noise, targets), ('unseen_noise', unseen_noise, None)]:
            metrics, states = evaluate_fixed(model, noise, mean, chosen, target)
            entry[kind] = metrics
            all_states[kind] = states
            picture_rows = {'target': targets} if target is not None else {}
            picture_rows.update({f'x_{k}': v for k, v in states.items()})
            save_rows(picture_rows, args.output / f'{kind}_epoch_{epoch:02d}.png')
            save_grid(states[max(1, depth)], args.output / f'{kind}_samples_epoch_{epoch:02d}.png')
        evaluations.append(entry)
        with (args.output / 'evaluation.jsonl').open('a') as stream:
            stream.write(json.dumps(entry) + '\n')
        print(json.dumps({'evaluation': entry}), flush=True)
        return all_states

    assess(0, 0, 0, 0)
    started = time.monotonic()
    visits = updates = 0
    epochs = []
    with (args.output / 'training.csv').open('w', buffering=1) as stream:
        writer = csv.writer(stream)
        writer.writerow(['epoch', 'depth', 'batch', 'image_visits', 'optimizer_updates', 'elapsed_seconds'] +
                        [f'mse_step_{k}' for k in range(1, max(args.schedule) + 1)])
        for epoch, depth in enumerate(args.schedule, start=1):
            counts = torch.zeros(len(data), dtype=torch.int32)
            sums = np.zeros(depth)
            for batch, (ids,) in enumerate(loader, start=1):
                counts[ids] += 1
                y = data[ids].to(device=device, dtype=torch.float32).div_(127.5).sub_(1)
                initial = fixed_noise[ids].to(device)
                model.train()
                losses = train_trajectory(model, optimizer, y, depth, initial=initial)
                sums += np.array(losses) * len(ids)
                visits += len(ids)
                updates += depth
                elapsed = time.monotonic() - started
                writer.writerow([epoch, depth, batch, visits, updates, elapsed] + losses + [''] * (max(args.schedule) - depth))
                if batch == 1 or batch % args.log_every == 0 or batch == len(loader):
                    print(json.dumps({'epoch': epoch, 'epochs': len(args.schedule), 'depth': depth,
                                      'batch': batch, 'batches': len(loader), 'image_visits': visits,
                                      'optimizer_updates': updates, 'elapsed_seconds': round(elapsed, 1),
                                      'eta_seconds': round(elapsed / updates * (total_updates - updates), 1),
                                      'loss_first': losses[0], 'loss_last': losses[-1]}), flush=True)
            assert torch.all(counts == 1), 'Each dataset index must appear exactly once per epoch'
            states = assess(epoch, depth, visits, updates)
            record = {'epoch': epoch, 'depth': depth, 'unique_images': len(data),
                      'image_visits': visits, 'optimizer_updates': updates,
                      'mean_training_mse_by_step': (sums / len(data)).tolist()}
            epochs.append(record)
            torch.save({'model': model.state_dict(), 'optimizer': optimizer.state_dict(), 'config': config,
                        'epoch': epoch, 'optimizer_updates': updates}, args.output / f'checkpoint_epoch_{epoch:02d}.pt')
            torch.save(states, args.output / f'evaluation_states_epoch_{epoch:02d}.pt')
    assert visits == len(data) * len(args.schedule)
    assert updates == total_updates
    assert noise_digest(fixed_noise) == pair_hash, 'Fixed noise bank was modified during training'
    summary = {'unique_images': len(data), 'image_visits': visits, 'optimizer_updates': updates,
               'schedule': args.schedule, 'elapsed_seconds': time.monotonic() - started,
               'noise_bank_unchanged': True, 'epoch_visit_counts_verified': True,
               'real_reference_samples': len(real), 'real_pixel_std_across_samples': real_std,
               'paired_subset_mean_predictor_mse': mean_target_mse,
               'epochs': epochs, 'initial_evaluation': evaluations[0], 'final_evaluation': evaluations[-1]}
    (args.output / 'summary.json').write_text(json.dumps(summary, indent=2) + '\n')
    print(json.dumps({'complete': str(args.output), 'seconds': summary['elapsed_seconds'],
                      'image_visits': visits, 'optimizer_updates': updates}), flush=True)


if __name__ == '__main__':
    main()
