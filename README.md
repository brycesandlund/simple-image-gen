# Simple image generation

Experiment history: [training_runs.md](training_runs.md).

Minimal experiment: start with Gaussian noise, repeatedly apply a shared image-to-image network, and train every transition by MSE to the same real image.

```python
for y in dataset_minibatches:             # exactly one shuffled epoch
    x = torch.randn_like(y)
    for _ in range(20):
        next_x = model(x)
        loss = mse(next_x, y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        x = next_x.detach()
```

The next input is the **pre-update forward output**. There is no recomputation after the optimizer update and no backpropagation through preceding transitions. Inference uses frozen weights, fresh noise, and 20 applications without target images or optimizer updates.

## Run

```sh
uv sync
uv run train.py --output runs/baseline
```

Defaults: all 50,000 CIFAR-10 training images once, batch size 128 (including the final partial batch), 20 transitions per batch (7,820 optimizer updates), Adam at 1e-4. The small U-Net uses width 24, GroupNorm, spatial skip connections, and direct RGB output. It has no class or time input, output residual, dropout, fresh intermediate noise, output clipping, weight decay, or learning-rate schedule. Images are scaled to [-1, 1]; initial noise is N(0,I). CPU, Apple Metal, and CUDA are supported.

For a separate mechanics/performance smoke test:

```sh
uv run train.py --limit-images 256 --output runs/smoke
```

Outputs include exact config and source snapshot, per-transition training losses, frozen-weight rollout images at steps 0/1/2/5/10/20, variation across generated samples, distance to the dataset mean image, real-image references, and the final checkpoint. The same held-out noise is evaluated before training and at each quarter epoch. PNGs clamp to [-1,1] for display; tensor states and metrics do not. No FID is reported from this small diagnostic sample.

Runs require a new empty output directory so previous experiments cannot be overwritten accidentally. Dataset files and checkpoints are ignored by Git.

## First baseline result (2026-10-21)

`runs/cifar10-baseline`: all 50,000 images, 7,820 updates, seed 42, default configuration, Apple Metal, approximately 342 seconds. The model has 266,547 parameters. Frozen 20-step rollouts from 64 fresh noise samples produced nearly identical smooth images, without recognizable objects. Mean per-pixel standard deviation across samples fell from 0.00535 at step 1 to 0.0000268 at step 20; the real-image reference was 0.494. Step-20 MSE to the dataset mean image was 0.00127 on [-1,1].

This is a result for the initial direct-output architecture and training order, not a conclusion about all recurrent variants. Raw evaluation states, quarter-epoch checkpoints, source snapshot, logs, and the final checkpoint are saved in that run directory. Training counts and finite losses were verified; the detached state-carry behavior was checked separately. No FID was computed.

## Fixed pairs and progressive recurrence

```sh
uv run train_fixed.py --device mps --schedule 1 2 3 5 8 --output runs/cifar10-fixed-progressive
uv run python -m unittest test_fixed -v
```

This starts a fresh model with the same architecture and optimizer. Each original CIFAR-10 training index receives one fixed CPU-generated Gaussian noise tensor (pair seed 2042), reused across all epochs despite reshuffling. The schedule specifies the number of detached transitions per minibatch in each epoch. Optimizer state persists across epochs; every new minibatch starts from its original fixed noise, not from a preceding epoch's output.

The default run makes five full dataset passes (250,000 image visits), 950,000 supervised image transitions, and 7,429 optimizer updates. It keeps batch size 128, learning rate 1e-4, and model/shuffle seed 42. A fixed subset of 64 training pairs and 64 unseen noise samples (seed 1042) are evaluated with frozen weights before training and after every epoch, through 20 steps. Evaluation includes the trained depths and the longer rollout; depths beyond the current curriculum stage are extrapolations.

Training-pair diagnostics compare MSE against the assigned target, a cyclic permutation of those targets, and the dataset mean. Unseen noise has no assigned ground-truth image; it is inspected for appearance and variation only. Checkpoints, raw evaluation states, sample grids, pairing checksum, source snapshots, and logs are saved each epoch. The test checks fixed pair identity through reshuffling and training, coverage including partial minibatches, independent evaluation noise, and optimizer update counts.

### First fixed-pair result (2026-10-21)

The five-epoch default run completed in approximately 338 seconds. At the final trained depth of 8, the 64 monitored training pairs had MSE 0.235486 against assigned targets versus 0.235485 against permuted targets; there was no meaningful assignment advantage. Unseen outputs were nearly identical smooth gray images. Their mean per-pixel standard deviation was 0.00403 at step 1, 0.0000404 at step 8, and 0.000000274 at step 20 (the latter is beyond the trained depth). Real-image reference standard deviation was 0.494.

All 50,000 indices appeared exactly once in each epoch, the fixed-noise bank checksum was unchanged after training, all logged losses were finite, and the final totals matched 250,000 image visits and 7,429 updates. This run did not demonstrate learning of the arbitrary fixed assignments at this model size and training budget. Both the pairings and curriculum changed relative to the first baseline, so this is not a single-variable ablation.

## Image/noise interpolation training

```sh
uv run train_interpolation.py --device mps --output runs/interpolation-repeat
uv run python -m unittest test_interpolation -v
```

Train direct clean-image prediction on `x = (1-t)z + ty`, sampling a fresh Gaussian `z` and independent `t ~ Uniform[0,1]` for every image/update. There is no timestep conditioning or recurrence during training. Defaults match the baseline's one-epoch, 20-draw, 7,820-update budget. Evaluation applies the model directly to pure noise for 1–20 steps and separately measures reconstruction on held-out CIFAR-10 test images. See E003 in the experiment ledger for results. `--fixed-image-fraction 0.2` is supported as a separate optional experiment; it has not been run here.

## Continue interpolation pretraining on model-generated states

```sh
uv run train.py --device mps --resume runs/cifar10-interpolation/checkpoint_seen_50000.pt --output runs/rollout-repeat
```

`--resume` restores model weights and Adam state, then trains one additional epoch with the original detached recurrent objective. The learning rate is set by `--lr` (default 1e-4). Early evaluations capture rapid changes. E004 records the run: output variation collapses after the first minibatch and does not recover during the epoch. Previous run artifacts are preserved.

## Categorical sampled rollouts

```sh
uv run train_categorical.py --device mps --output runs/categorical-repeat
uv run python test_categorical.py
```

Predict 200 bins per RGB channel, train with cross-entropy against the same target throughout 20 recurrent steps, and feed independently sampled categorical outputs into the next step. Initialization samples the full dataset's per-position/channel histograms. Temperature is 1; choices are detached and sampled from pre-update logits. E005 in `training_runs.md` records the result and comparison with independent histogram samples.
