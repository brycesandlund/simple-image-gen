"""Direct image regression from independently sampled noise/image interpolants."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import platform
import time
from pathlib import Path

import numpy as np
import torch
from torch.nn import functional as F
from torch.utils.data import DataLoader, TensorDataset
from torchvision.datasets import CIFAR10

from train import Refiner, choose_device, evaluate, save_grid, save_trajectory
from train_fixed import save_rows


def interpolate(y, noise, image_fraction):
    return (1 - image_fraction) * noise + image_fraction * y


def train_interpolations(model, optimizer, y, draws, fixed_fraction=None):
    losses, fractions = [], []
    for _ in range(draws):
        noise = torch.randn_like(y)
        t = (torch.rand(len(y), 1, 1, 1, device=y.device) if fixed_fraction is None
             else torch.full((len(y), 1, 1, 1), fixed_fraction, device=y.device))
        x = interpolate(y, noise, t)
        prediction = model(x)  # No t input, and predictions never become training inputs.
        loss = F.mse_loss(prediction, y)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        losses.append(loss.detach())
        fractions.append(t.mean())
    loss_values = torch.stack(losses).cpu().tolist()
    if not all(math.isfinite(v) for v in loss_values):
        raise RuntimeError('Nonfinite training loss')
    return loss_values, torch.stack(fractions).cpu().tolist()


@torch.no_grad()
def evaluate_denoising(model, targets, noise, fractions, directory, seen):
    was_training = model.training
    model.eval()
    metrics, states = [], {'targets': targets.cpu()}
    for t in fractions:
        x = interpolate(targets, noise, t)
        prediction = model(x)
        if not torch.isfinite(prediction).all():
            raise RuntimeError('Nonfinite held-out reconstruction')
        metrics.append({'image_fraction': t, 'mse_to_target': F.mse_loss(prediction, targets).item(),
                        'corrupted_input_mse_to_target': F.mse_loss(x, targets).item(),
                        'pixel_std_across_samples': prediction.std(0, unbiased=False).mean().item()})
        states[f'input_{t:g}'] = x.cpu()
        states[f'prediction_{t:g}'] = prediction.cpu()
    save_rows(states, directory / f'heldout_denoising_seen_{seen:05d}.png')
    with (directory / 'denoising_evaluation.jsonl').open('a') as stream:
        stream.write(json.dumps({'images_seen': seen, 'metrics': metrics}) + '\n')
    model.train(was_training)
    return metrics, states


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data', type=Path, default=Path('data'))
    parser.add_argument('--output', type=Path, default=Path('runs/cifar10-interpolation'))
    parser.add_argument('--draws', type=int, default=20)
    parser.add_argument('--fixed-image-fraction', type=float, default=None)
    parser.add_argument('--batch-size', type=int, default=128)
    parser.add_argument('--width', type=int, default=24)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--eval-seed', type=int, default=1042)
    parser.add_argument('--eval-samples', type=int, default=64)
    parser.add_argument('--eval-steps', type=int, default=20)
    parser.add_argument('--denoise-eval-seed', type=int, default=3042)
    parser.add_argument('--device', choices=['auto', 'cpu', 'mps', 'cuda'], default='auto')
    parser.add_argument('--limit-images', type=int, default=0)
    parser.add_argument('--log-every', type=int, default=10)
    args = parser.parse_args()
    if min(args.draws, args.batch_size, args.eval_samples, args.eval_steps, args.log_every) < 1:
        parser.error('all counts must be positive')
    if args.width <= 0 or args.width % 4 or args.limit_images < 0 or args.lr <= 0:
        parser.error('invalid width, image limit, or learning rate')
    if args.fixed_image_fraction is not None and not 0 <= args.fixed_image_fraction <= 1:
        parser.error('fixed-image-fraction must be in [0,1]')
    args.output.mkdir(parents=True, exist_ok=True)
    if any(args.output.iterdir()):
        parser.error('output directory must be empty')
    torch.set_num_threads(6)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = choose_device(args.device)
    data = torch.from_numpy(CIFAR10(str(args.data), train=True, download=True).data).permute(0,3,1,2)
    if args.limit_images:
        data = data[:args.limit_images]
    loader = DataLoader(TensorDataset(torch.arange(len(data))), batch_size=args.batch_size, shuffle=True,
                        generator=torch.Generator().manual_seed(args.seed), drop_last=False, num_workers=0)
    mean = torch.from_numpy(data.numpy().mean(0, dtype=np.float64)).float().div(127.5).sub(1)
    model = Refiner(args.width).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    eval_noise = torch.randn(args.eval_samples,3,32,32,generator=torch.Generator().manual_seed(args.eval_seed)).to(device)
    test = CIFAR10(str(args.data), train=False, download=False).data[:args.eval_samples]
    test_targets = torch.from_numpy(test).permute(0,3,1,2).to(device, dtype=torch.float32).div(127.5).sub(1)
    test_noise = torch.randn(test_targets.shape, generator=torch.Generator().manual_seed(args.denoise_eval_seed)).to(device)
    fractions = sorted({0., .1, .2, .5, .8, 1., *([] if args.fixed_image_fraction is None else [args.fixed_image_fraction])})
    config = {k: str(v) if isinstance(v, Path) else v for k,v in vars(args).items()}
    config.update(dataset='CIFAR-10 train', images=len(data), epochs=1, device=str(device),
                  parameters=sum(p.numel() for p in model.parameters()), optimizer='Adam', betas=[.9,.999], epsilon=1e-8,
                  weight_decay=0, torch_version=torch.__version__, python=platform.python_version(),
                  training_input='x=(1-t)*z+t*y; fresh independent z and per-image t each update',
                  image_fraction_distribution='Uniform[0,1]' if args.fixed_image_fraction is None else f'constant {args.fixed_image_fraction}',
                  target='clean image y, not velocity or noise', timestep_conditioning=False,
                  recurrence='none in training; direct f composition at evaluation only',
                  target_scaling='uint8 / 127.5 - 1', output_constraints='none; PNG rendering clips only',
                  expected_updates=len(loader)*args.draws, supervised_image_pairs=len(data)*args.draws,
                  denoise_eval_split='CIFAR-10 test, first eval-samples images', denoise_eval_fractions=fractions,
                  source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                  model_source_sha256=hashlib.sha256(Path(__file__).with_name('train.py').read_bytes()).hexdigest())
    (args.output/'config.json').write_text(json.dumps(config,indent=2)+'\n')
    for name in ('train.py','train_fixed.py','train_interpolation.py'):
        (args.output/name).write_text(Path(__file__).with_name(name).read_text())
    save_grid(data[:64].float().div(127.5).sub(1),args.output/'real_images.png')
    save_grid(mean.unsqueeze(0),args.output/'dataset_mean.png',columns=1,scale=8)
    print(json.dumps(config),flush=True)
    initial_eval,_=evaluate(model,eval_noise,mean,args.eval_steps,args.output,0)
    initial_denoise,_=evaluate_denoising(model,test_targets,test_noise,fractions,args.output,0)
    milestones={math.ceil(len(loader)*q/4) for q in range(1,5)}
    counts=torch.zeros(len(data),dtype=torch.int32)
    seen=0
    totals=np.zeros(args.draws)
    fraction_sum=0.
    started=time.monotonic()
    with (args.output/'training.csv').open('w',buffering=1) as stream:
        writer=csv.writer(stream)
        writer.writerow(['batch','images_seen','optimizer_updates','elapsed_seconds','mean_image_fraction']+[f'mse_draw_{i}' for i in range(1,args.draws+1)])
        for batch,(ids,) in enumerate(loader,start=1):
            counts[ids]+=1
            y=data[ids].to(device,dtype=torch.float32).div_(127.5).sub_(1)
            model.train()
            losses,ts=train_interpolations(model,optimizer,y,args.draws,args.fixed_image_fraction)
            seen+=len(ids)
            totals+=np.array(losses)*len(ids)
            fraction_sum+=sum(ts)*len(ids)
            elapsed=time.monotonic()-started
            writer.writerow([batch,seen,batch*args.draws,elapsed,np.mean(ts)]+losses)
            if batch==1 or batch%args.log_every==0 or batch==len(loader):
                print(json.dumps({'batch':batch,'batches':len(loader),'images_seen':seen,'updates':batch*args.draws,
                                  'elapsed_seconds':round(elapsed,1),'eta_seconds':round(elapsed/batch*(len(loader)-batch),1),
                                  'mean_mse':float(np.mean(losses)),'mean_image_fraction':float(np.mean(ts))}),flush=True)
            if batch in milestones:
                final_eval,states=evaluate(model,eval_noise,mean,args.eval_steps,args.output,seen)
                final_denoise,denoise_states=evaluate_denoising(model,test_targets,test_noise,fractions,args.output,seen)
                torch.save({'model':model.state_dict(),'optimizer':optimizer.state_dict(),'config':config,
                            'images_seen':seen,'optimizer_updates':batch*args.draws},args.output/f'checkpoint_seen_{seen:05d}.pt')
                print(json.dumps({'evaluation':final_eval,'heldout_denoising':final_denoise}),flush=True)
    assert seen==len(data) and torch.all(counts==1)
    torch.save(states,args.output/'evaluation_states.pt')
    torch.save(denoise_states,args.output/'heldout_denoising_states.pt')
    save_grid(states[1],args.output/'samples_x1.png')
    save_grid(states[args.eval_steps],args.output/'samples_final.png')
    save_trajectory(states,args.output/'trajectory_final.png')
    real=data[:1024].float().div(127.5).sub(1)
    summary={'images_seen':seen,'optimizer_updates':len(loader)*args.draws,'supervised_image_pairs':seen*args.draws,
             'elapsed_seconds':time.monotonic()-started,'epoch_visit_counts_verified':True,
             'mean_training_mse_by_draw':(totals/seen).tolist(),'mean_training_mse':float(totals.sum()/(seen*args.draws)),
             'mean_image_fraction':fraction_sum/(seen*args.draws),
             'real_reference_samples':len(real),'real_pixel_std_across_samples':real.std(0,unbiased=False).mean().item(),
             'heldout_dataset_mean_predictor_mse':F.mse_loss(mean.to(device).expand_as(test_targets),test_targets).item(),
             'initial_evaluation':initial_eval,'final_evaluation':final_eval,
             'initial_heldout_denoising':initial_denoise,'final_heldout_denoising':final_denoise}
    (args.output/'summary.json').write_text(json.dumps(summary,indent=2)+'\n')
    print(json.dumps({'complete':str(args.output),'seconds':summary['elapsed_seconds'],'images':seen,
                      'optimizer_updates':summary['optimizer_updates']}),flush=True)


if __name__=='__main__':
    main()
