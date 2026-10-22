"""Categorical RGB prediction with independently sampled, detached rollouts."""
from __future__ import annotations
import argparse, csv, hashlib, json, math, random, time
from pathlib import Path
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader, TensorDataset
from torchvision.datasets import CIFAR10
from train import Refiner, choose_device, save_grid, save_trajectory


def quantize(raw, bins):
    return (raw.to(torch.int64) * (bins - 1) + 127) // 255


def decode(indices, bins):
    return indices.float().mul(2 / (bins - 1)).sub(1)


def pixel_histograms(raw, bins):
    # Chunked counts avoid constructing a dataset-sized one-hot tensor.
    counts = torch.zeros(3 * 32 * 32, bins, dtype=torch.int64)
    for chunk in raw.split(512):
        ids = quantize(chunk, bins).reshape(len(chunk), -1).T
        counts.scatter_add_(1, ids, torch.ones_like(ids))
    assert torch.all(counts.sum(1) == len(raw))
    return counts.float().div(len(raw))


def sample_histograms(probabilities, count, generator=None):
    ids = torch.multinomial(probabilities, count, replacement=True, generator=generator).T
    return ids.reshape(count, 3, 32, 32)


class CategoricalRefiner(Refiner):
    def __init__(self, width=24, bins=200):
        super().__init__(width)
        self.bins = bins
        self.out = nn.Conv2d(width, 3 * bins, 1)

    def forward(self, x):
        return super().forward(x).reshape(len(x), 3, self.bins, *x.shape[-2:])


@torch.no_grad()
def sample_logits(logits):
    # One independent categorical draw for each pixel/channel, temperature 1.
    b, c, k, h, w = logits.shape
    probabilities = logits.detach().permute(0, 1, 3, 4, 2).reshape(-1, k).softmax(-1)
    ids = torch.multinomial(probabilities, 1).reshape(b, c, h, w)
    return decode(ids, k)


def train_trajectory(model, optimizer, target, initial, steps):
    x = initial.detach()
    values = []
    for _ in range(steps):
        assert not x.requires_grad and x.grad_fn is None
        logits = model(x)
        b, c, k, h, w = logits.shape
        loss = F.cross_entropy(logits.reshape(b*c, k, h, w), target.reshape(b*c, h, w))
        # Sample the pre-update output; never backpropagate through the choice.
        next_x = sample_logits(logits)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        x = next_x
        values.append(loss.detach())
    result = torch.stack(values).cpu().tolist()
    if not all(math.isfinite(v) for v in result):
        raise RuntimeError('Nonfinite training loss')
    return result


def metrics(x, mean):
    x = x.cpu()
    # Adjacent-pixel differences distinguish static from smooth image structure.
    tv = ((x[:, :, 1:, :] - x[:, :, :-1, :]).abs().mean() +
          (x[:, :, :, 1:] - x[:, :, :, :-1]).abs().mean()) / 2
    centered = x - mean
    cov = ((centered[:, :, 1:, :] * centered[:, :, :-1, :]).mean() +
           (centered[:, :, :, 1:] * centered[:, :, :, :-1]).mean()) / 2
    return {'pixel_std_across_samples': x.std(0, unbiased=False).mean().item(),
            'adjacent_pixel_abs_difference': tv.item(), 'adjacent_centered_product': cov.item(),
            'mse_to_dataset_mean': F.mse_loss(x, mean.expand_as(x)).item()}


@torch.no_grad()
def evaluate(model, initial, mean, directory, seen, steps, seed, device):
    cpu_rng = torch.get_rng_state()
    accelerator_rng = torch.mps.get_rng_state() if device.type == 'mps' else None
    if device.type == 'cuda':
        accelerator_rng = torch.cuda.get_rng_state(device)
    was_training = model.training
    try:
        torch.manual_seed(seed)
        model.eval()
        x = initial.to(device)
        states, records = {}, []
        for step in range(steps + 1):
            if step in {0, 1, 2, 5, 10, steps}:
                assert torch.isfinite(x).all()
                states[step] = x.cpu()
                records.append({'step': step, **metrics(states[step], mean)})
            if step < steps:
                x = sample_logits(model(x))
        save_trajectory(states, directory / f'trajectory_seen_{seen:05d}.png')
        record = {'images_seen': seen, 'metrics': records}
        with (directory/'evaluation.jsonl').open('a') as f:
            f.write(json.dumps(record)+'\n')
        return record, states
    finally:
        torch.set_rng_state(cpu_rng)
        if device.type == 'mps': torch.mps.set_rng_state(accelerator_rng)
        if device.type == 'cuda': torch.cuda.set_rng_state(accelerator_rng, device)
        model.train(was_training)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--data', type=Path, default=Path('data'))
    p.add_argument('--output', type=Path, default=Path('runs/categorical'))
    p.add_argument('--bins', type=int, default=200)
    p.add_argument('--steps', type=int, default=20)
    p.add_argument('--batch-size', type=int, default=128)
    p.add_argument('--width', type=int, default=24)
    p.add_argument('--lr', type=float, default=1e-4)
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--eval-seed', type=int, default=1042)
    p.add_argument('--eval-samples', type=int, default=64)
    p.add_argument('--device', choices=['auto','cpu','mps','cuda'], default='auto')
    p.add_argument('--limit-images', type=int, default=0)
    a=p.parse_args()
    if not 2 <= a.bins <= 256 or min(a.steps,a.batch_size,a.eval_samples,a.width)<=0 or a.width%4 or a.lr<=0 or a.limit_images<0:
        p.error('Invalid counts, width, learning rate, or bins (2..256)')
    a.output.mkdir(parents=True,exist_ok=True)
    if any(a.output.iterdir()): p.error('Output directory must be empty')
    torch.set_num_threads(6)
    random.seed(a.seed); np.random.seed(a.seed); torch.manual_seed(a.seed)
    device=choose_device(a.device)
    raw=torch.from_numpy(CIFAR10(str(a.data),train=True,download=False).data).permute(0,3,1,2)
    # Always use full-dataset histograms, including in smoke runs.
    probabilities=pixel_histograms(raw,a.bins)
    mean=(probabilities*torch.linspace(-1,1,a.bins)).sum(1).reshape(3,32,32)
    histogram_entropy=-(probabilities*probabilities.clamp_min(1e-30).log()).sum(1).mean().item()
    reference=decode(quantize(raw[:1024],a.bins),a.bins)
    data=raw[:a.limit_images] if a.limit_images else raw
    loader=DataLoader(TensorDataset(torch.arange(len(data))),batch_size=a.batch_size,shuffle=True,
                      generator=torch.Generator().manual_seed(a.seed),num_workers=0,drop_last=False)
    model=CategoricalRefiner(a.width,a.bins).to(device)
    optimizer=torch.optim.Adam(model.parameters(),lr=a.lr)
    init_generator=torch.Generator().manual_seed(a.seed+2000)
    eval_generator=torch.Generator().manual_seed(a.eval_seed)
    initial=decode(sample_histograms(probabilities,a.eval_samples,eval_generator),a.bins)
    control=decode(sample_histograms(probabilities,a.eval_samples,eval_generator),a.bins)
    config={k:str(v) if isinstance(v,Path) else v for k,v in vars(a).items()}
    config.update(device=str(device),torch_version=str(torch.__version__),parameters=sum(p.numel() for p in model.parameters()),
        images=len(data),epochs=1,histogram_images=len(raw),histogram_scope='separate spatial position and RGB channel',
        quantization='nearest of 200 evenly spaced levels in [-1,1]' if a.bins==200 else 'nearest evenly spaced levels in [-1,1]',
        temperature=1.0,optimizer='Adam',initialization='fresh model; independent samples from full-dataset categorical marginals',
        objective='mean cross entropy over batch, RGB channels, and positions; fixed target throughout each trajectory',
        state_timing='sample pre-update logits independently per pixel/channel; carry detached sample',
        initial_sample_seed=a.seed+2000,eval_rollout_seed=a.eval_seed+10000,
        source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
    (a.output/'config.json').write_text(json.dumps(config,indent=2)+'\n')
    (a.output/'train_categorical.py').write_text(Path(__file__).read_text())
    (a.output/'train.py').write_text(Path(__file__).with_name('train.py').read_text())
    torch.save({'probabilities':probabilities,'mean':mean,'initial':initial,'control':control},a.output/'histograms.pt')
    save_grid(reference[:64],a.output/'real_images.png');save_grid(control,a.output/'histogram_control.png')
    initial_eval,_=evaluate(model,initial,mean,a.output,0,a.steps,a.eval_seed+10000,device)
    milestones={math.ceil(len(loader)*q/4) for q in range(1,5)}
    totals=np.zeros(a.steps);counts=torch.zeros(len(data),dtype=torch.int32);seen=0;started=time.monotonic()
    print(json.dumps(config),flush=True)
    with (a.output/'training.csv').open('w',buffering=1) as f:
        writer=csv.writer(f);writer.writerow(['batch','images_seen','optimizer_updates','elapsed_seconds']+[f'ce_step_{i}' for i in range(1,a.steps+1)])
        for batch,(ids,) in enumerate(loader,1):
            model.train();counts[ids]+=1
            target=quantize(data[ids],a.bins).to(device)
            x=decode(sample_histograms(probabilities,len(ids),init_generator),a.bins).to(device)
            losses=train_trajectory(model,optimizer,target,x,a.steps)
            seen+=len(ids);totals+=np.array(losses)*len(ids);elapsed=time.monotonic()-started
            writer.writerow([batch,seen,batch*a.steps,elapsed]+losses)
            if batch==1 or batch%10==0 or batch==len(loader):
                print(json.dumps({'batch':batch,'batches':len(loader),'images_seen':seen,'updates':batch*a.steps,
                    'elapsed_seconds':round(elapsed,1),'eta_seconds':round(elapsed/batch*(len(loader)-batch),1),
                    'ce_first':losses[0],'ce_last':losses[-1]}),flush=True)
            if batch in milestones:
                final_eval,states=evaluate(model,initial,mean,a.output,seen,a.steps,a.eval_seed+10000,device)
                torch.save({'model':model.state_dict(),'optimizer':optimizer.state_dict(),'config':config,
                    'images_seen':seen,'optimizer_updates':batch*a.steps},a.output/f'checkpoint_seen_{seen:05d}.pt')
                print(json.dumps({'evaluation':final_eval}),flush=True)
    assert seen==len(data) and torch.all(counts==1)
    torch.save(states,a.output/'evaluation_states.pt')
    save_trajectory(states,a.output/'trajectory_final.png');save_grid(states[a.steps],a.output/'samples_final.png')
    summary={'images_seen':seen,'optimizer_updates':len(loader)*a.steps,'supervised_image_pairs':seen*a.steps,
        'elapsed_seconds':time.monotonic()-started,'epoch_visit_counts_verified':True,
        'mean_training_ce_by_step':(totals/seen).tolist(),'histogram_baseline_ce':histogram_entropy,
        'histogram_control_metrics':metrics(control,mean),'real_metrics':metrics(reference,mean),
        'initial_evaluation':initial_eval,'final_evaluation':final_eval}
    (a.output/'summary.json').write_text(json.dumps(summary,indent=2)+'\n')
    print(json.dumps({'complete':str(a.output),'summary':summary}),flush=True)

if __name__=='__main__': main()
