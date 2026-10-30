"""Continuous MSE denoising pretraining with fresh pixel replacement each epoch."""
from __future__ import annotations
import argparse, csv, hashlib, json, math, time
from pathlib import Path
import numpy as np
import torch
from torch.nn import functional as F
from torch.utils.data import DataLoader, TensorDataset
from torchvision.datasets import CIFAR10
from train import Refiner, choose_device, evaluate, save_grid, save_trajectory
from train_fixed import save_rows
from train_categorical import pixel_histograms, sample_histograms


def replace_pixels(target, replacement, fraction, uniforms=None):
    """One RGB-shared Bernoulli mask per pixel; level is hidden from the model."""
    if uniforms is None:
        uniforms = torch.rand(len(target), 1, *target.shape[-2:], device=target.device)
    mask = uniforms < fraction
    return torch.where(mask, replacement, target)


def sample_colors(probabilities, count, generator):
    # 256 empirical intensities; model input/output and loss are continuous floats.
    return sample_histograms(probabilities, count, generator).float().div(127.5).sub(1)


def train_update(model, optimizer, target, replacement, fraction):
    x = replace_pixels(target, replacement, fraction)
    prediction = model(x)
    loss = F.mse_loss(prediction, target)
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    optimizer.step()
    value = loss.detach().item()
    if not math.isfinite(value): raise RuntimeError('Nonfinite loss')
    return value


def state_metrics(x, target, mean, mask):
    x = x.cpu(); target = target.cpu()
    if not torch.isfinite(x).all(): raise RuntimeError('Nonfinite evaluation state')
    error=(x-target).square()
    mask=mask.cpu().expand_as(x)
    return {'mse_to_target':error.mean().item(),
        'mse_replaced_pixels':error[mask].mean().item() if mask.any() else None,
        'mse_retained_pixels':error[~mask].mean().item() if (~mask).any() else None,
        'pixel_std_across_samples':x.std(0,unbiased=False).mean().item(),
        'mse_to_dataset_mean':F.mse_loss(x,mean.expand_as(x)).item(),
        'out_of_display_range_fraction':((x < -1)|(x > 1)).float().mean().item()}


@torch.no_grad()
def evaluate_denoising(model, targets, replacement, uniforms, mean, fractions, steps, directory, epoch):
    was_training=model.training;model.eval()
    records=[];states={}
    try:
        for r in fractions:
            x=replace_pixels(targets,replacement,r,uniforms)
            current={};metrics=[]
            for step in range(steps+1):
                if step in {0,1,2,5,10,steps}:
                    current[step]=x.cpu()
                    metrics.append({'step':step,**state_metrics(x,targets,mean,uniforms < r)})
                if step<steps: x=model(x)
            states[f'{r:g}']=current
            records.append({'corruption_fraction':r,'metrics':metrics})
        record={'epoch':epoch,'metrics':records}
        with (directory/'denoising_evaluation.jsonl').open('a') as f: f.write(json.dumps(record)+'\n')
        rows={'target':targets.cpu()}
        for r in (.2,.5,.8,.95):
            rows[f'input r={r:g}']=states[f'{r:g}'][0]
            rows[f'one pass r={r:g}']=states[f'{r:g}'][1]
        save_rows(rows,directory/f'denoising_epoch_{epoch:02d}.png')
        return record,states
    finally: model.train(was_training)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--data',type=Path,default=Path('data'))
    p.add_argument('--output',type=Path,default=Path('runs/cifar10-denoising'))
    p.add_argument('--epochs',type=int,default=20)
    p.add_argument('--batch-size',type=int,default=128)
    p.add_argument('--width',type=int,default=24)
    p.add_argument('--lr',type=float,default=1e-4)
    p.add_argument('--seed',type=int,default=42)
    p.add_argument('--eval-seed',type=int,default=1042)
    p.add_argument('--eval-samples',type=int,default=64)
    p.add_argument('--eval-steps',type=int,default=20)
    p.add_argument('--device',choices=['auto','cpu','mps','cuda'],default='auto')
    p.add_argument('--limit-images',type=int,default=0)
    a=p.parse_args()
    if min(a.epochs,a.batch_size,a.width,a.eval_samples,a.eval_steps)<1 or a.width%4 or a.lr<=0 or a.limit_images<0:
        p.error('Invalid arguments')
    a.output.mkdir(parents=True,exist_ok=True)
    if any(a.output.iterdir()):p.error('Output directory must be empty')
    torch.set_num_threads(6);torch.manual_seed(a.seed);np.random.seed(a.seed)
    device=choose_device(a.device)
    raw=torch.from_numpy(CIFAR10(str(a.data),train=True,download=False).data).permute(0,3,1,2)
    probabilities=pixel_histograms(raw,256)
    mean=(probabilities*torch.linspace(-1,1,256)).sum(1).reshape(3,32,32)
    data=raw[:a.limit_images] if a.limit_images else raw
    loader=DataLoader(TensorDataset(torch.arange(len(data))),batch_size=a.batch_size,shuffle=True,
        generator=torch.Generator().manual_seed(a.seed),drop_last=False,num_workers=0)
    model=Refiner(a.width).to(device);optimizer=torch.optim.Adam(model.parameters(),lr=a.lr)
    train_generator=torch.Generator().manual_seed(a.seed+2000)
    histogram_initial=sample_colors(probabilities,a.eval_samples,torch.Generator().manual_seed(a.eval_seed)).to(device)
    test=torch.from_numpy(CIFAR10(str(a.data),train=False,download=False).data[:a.eval_samples]).permute(0,3,1,2)
    targets=test.to(device,dtype=torch.float32).div(127.5).sub(1)
    eval_generator=torch.Generator().manual_seed(a.eval_seed+2000)
    replacement=sample_colors(probabilities,a.eval_samples,eval_generator).to(device)
    uniforms=torch.rand(a.eval_samples,1,32,32,generator=eval_generator).to(device)
    fractions=[0.,.2,.5,.8,.95,1.]
    config={k:str(v) if isinstance(v,Path) else v for k,v in vars(a).items()}
    config.update(images=len(data),device=str(device),parameters=sum(v.numel() for v in model.parameters()),
        torch_version=str(torch.__version__),optimizer='Adam',betas=[.9,.999],epsilon=1e-8,weight_decay=0,
        objective='continuous clean-image MSE; one update per minibatch; fresh corruption every image visit',
        corruption='per-image r~Uniform[0,1]; RGB-shared per-pixel mask; replaced channels independently sampled from full-dataset per-position histograms',
        replacement_histogram_images=len(raw),replacement_bins=256,timestep_conditioning=False,recurrence_in_training=False,
        expected_updates=len(loader)*a.epochs,supervised_image_pairs=len(data)*a.epochs,
        target_scaling='uint8 / 127.5 - 1; float input/output; no output clipping',
        eval_fractions=fractions,denoising_split='first eval-samples CIFAR-10 test images',
        source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
    (a.output/'config.json').write_text(json.dumps(config,indent=2)+'\n')
    for name in ('train_denoising.py','train.py','train_fixed.py','train_categorical.py'):
        (a.output/name).write_text(Path(__file__).with_name(name).read_text())
    torch.save({'probabilities':probabilities,'mean':mean,'histogram_initial':histogram_initial.cpu(),
        'test_targets':targets.cpu(),'replacement':replacement.cpu(),'uniforms':uniforms.cpu()},a.output/'evaluation_inputs.pt')
    save_grid(raw[:64].float().div(127.5).sub(1),a.output/'real_images.png')
    print(json.dumps(config),flush=True)
    initial_eval,_=evaluate(model,histogram_initial,mean,a.eval_steps,a.output,0)
    initial_denoising,_=evaluate_denoising(model,targets,replacement,uniforms,mean,fractions,a.eval_steps,a.output,0)
    milestones={1,a.epochs,*[math.ceil(a.epochs*q/4) for q in range(1,5)]}
    seen=updates=0;epoch_records=[];started=time.monotonic()
    with (a.output/'training.csv').open('w',buffering=1) as f:
        writer=csv.writer(f);writer.writerow(['epoch','batch','images_seen','optimizer_updates','elapsed_seconds','mse','mean_corruption_fraction'])
        for epoch in range(1,a.epochs+1):
            counts=torch.zeros(len(data),dtype=torch.int32);loss_sum=fraction_sum=0.
            for batch,(ids,) in enumerate(loader,1):
                model.train();counts[ids]+=1
                target=data[ids].to(device,dtype=torch.float32).div(127.5).sub(1)
                colors=sample_colors(probabilities,len(ids),train_generator).to(device)
                fraction=torch.rand(len(ids),1,1,1,device=device)
                loss=train_update(model,optimizer,target,colors,fraction)
                mean_fraction=fraction.mean().item()
                seen+=len(ids);updates+=1;loss_sum+=loss*len(ids);fraction_sum+=mean_fraction*len(ids)
                elapsed=time.monotonic()-started
                writer.writerow([epoch,batch,seen,updates,elapsed,loss,mean_fraction])
                if updates==1 or updates%200==0:
                    print(json.dumps({'epoch':epoch,'batch':batch,'updates':updates,'images_seen':seen,
                        'elapsed_seconds':round(elapsed,1),'eta_seconds':round(elapsed/updates*(len(loader)*a.epochs-updates),1),'mse':loss}),flush=True)
            assert torch.all(counts==1)
            record={'epoch':epoch,'mean_training_mse':loss_sum/len(data),'mean_corruption_fraction':fraction_sum/len(data),'epoch_visit_counts_verified':True}
            epoch_records.append(record);print(json.dumps(record),flush=True)
            if epoch in milestones:
                final_eval,states=evaluate(model,histogram_initial,mean,a.eval_steps,a.output,seen)
                final_denoising,denoising_states=evaluate_denoising(model,targets,replacement,uniforms,mean,fractions,a.eval_steps,a.output,epoch)
                torch.save({'model':model.state_dict(),'optimizer':optimizer.state_dict(),'config':config,'epoch':epoch,
                    'images_seen':seen,'optimizer_updates':updates},a.output/f'checkpoint_epoch_{epoch:02d}.pt')
                torch.save({'generation':states,'denoising':denoising_states},a.output/f'evaluation_states_epoch_{epoch:02d}.pt')
                print(json.dumps({'evaluation_epoch':epoch,'generation':final_eval,'denoising':final_denoising}),flush=True)
    assert seen==len(data)*a.epochs and updates==len(loader)*a.epochs
    save_grid(states[a.eval_steps],a.output/'samples_final.png');save_trajectory(states,a.output/'trajectory_final.png')
    summary={'images':len(data),'image_visits':seen,'optimizer_updates':updates,'elapsed_seconds':time.monotonic()-started,
        'epochs':epoch_records,'epoch_visit_counts_verified':True,'initial_evaluation':initial_eval,'final_evaluation':final_eval,
        'initial_denoising':initial_denoising,'final_denoising':final_denoising,
        'heldout_mean_predictor_mse':F.mse_loss(mean.to(device).expand_as(targets),targets).item()}
    (a.output/'summary.json').write_text(json.dumps(summary,indent=2)+'\n')
    print(json.dumps({'complete':str(a.output),'seconds':summary['elapsed_seconds'],'updates':updates}),flush=True)

if __name__=='__main__':main()
