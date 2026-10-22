# Training runs

Experiment ledger for `simple-image-gen`. Dates use America/Los_Angeles. Values below come from the saved run configs, summaries, and evaluation logs. Runtime means the measured training/evaluation loop; it excludes setup and dataset download.

## Run index

| ID | Date | Run directory | Purpose | Epochs / supervised updates per minibatch | Optimizer updates | Runtime | Result |
| --- | --- | --- | --- | --- | --- | --- | --- |
| E001 | 2026-10-21 | `runs/cifar10-baseline` | Fresh-noise recurrent MSE baseline | 1 epoch / 20 | 7,820 | 341.8 s | Nearly identical gray outputs |
| E002 | 2026-10-21 | `runs/cifar10-fixed-progressive` | Fixed noise/image pairs with progressive recurrence | 5 epochs / 1, 2, 3, 5, 8 | 7,429 | 338.3 s | Gray outputs; no meaningful assigned-target advantage on monitored training pairs |
| E003 | 2026-10-21 | `runs/cifar10-interpolation` | Uniform noise/image interpolation regression | 1 epoch / 20 independent draws | 7,820 | 357.7 s | Distinct blurry outputs; no uniform-gray collapse |
| E004 | 2026-10-21 | `runs/cifar10-interpolation-rollout` | E003 checkpoint continued on model-generated states | 1 additional epoch / 20 | 7,820 new | 429.9 s | Gray collapse after first batch; persists through epoch |
| E005 | 2026-10-21 | `runs/cifar10-categorical` | Categorical CE with sampled recurrent states | 1 epoch / 20 | 7,820 | 929.4 s | Colorful static; no visible spatial structure |
| S001 | 2026-10-21 | `runs/smoke` | Baseline mechanics check on 256 images | 1 epoch / 20 | 40 | 3.44 s | Completed; not a quality experiment |
| S002 | 2026-10-21 | `runs/fixed-progressive-smoke` | Fixed-pair curriculum check on 256 images | 5 epochs / 1, 2, 3, 5, 8 | 38 | 3.97 s | Completed; pairing and coverage checks passed |
| S003 | 2026-10-21 | `runs/interpolation-smoke` | Interpolation mechanics check on 256 images | 1 epoch / 20 independent draws | 40 | 2.47 s | Completed; interpolation tests passed |
| S004 | 2026-10-21 | `runs/rollout-finetune-smoke` | Checkpoint continuation on 256 images | 1 epoch / 20 | 40 new | 2.71 s | Exact starting outputs and Adam continuation verified |
| S005 | 2026-10-21 | `runs/categorical-smoke` | Categorical mechanics on 256 images | 1 epoch / 20 | 40 | 7.00 s | Passed three categorical tests and GPU smoke run |

## Shared setup and metric definitions

- **Data:** CIFAR-10 training images, 32×32 RGB, scaled as `uint8 / 127.5 - 1` in regression runs; E005 quantizes to 200 evenly spaced channel values in [-1,1]. Labels are unused. Full experiments use all 50,000 images, shuffled each epoch, without augmentation.
- **Model:** shared direct-output U-Net, base width 24, 266,547 parameters, SiLU, GroupNorm, and internal spatial skip connections. E001–E003 start from fresh initialization; E004 continues the final E003 weights and Adam state. E005 starts fresh with the same U-Net body and a 600-logit categorical head (281,472 total parameters).
- **Optimizer:** Adam, learning rate `1e-4`, betas `(0.9, 0.999)`, epsilon `1e-8`, no weight decay. Batch size 128, including the final batch of 80 on full-data epochs.
- **Training:** MSE averaged over batch, channels, and pixels, with one optimizer update per supervised minibatch. In recurrent runs E001/E002/E004, the pre-update forward output is detached and carried to the next transition, without recomputation or rollout backpropagation. E003 instead samples each training input independently from an image/noise interpolation. E005 uses cross-entropy and carries detached samples from categorical output distributions.
- **Excluded features:** no time/class conditioning, output residual connection, additive intermediate noise injection, output clipping, dropout, gradient clipping, or learning-rate schedule.
- **Seeds/runtime:** model and shuffle seed 42; unseen evaluation noise seed 1042; PyTorch 2.14.1; Apple Metal (`mps`) on an M5 Pro.
- **Evaluation:** frozen weights, no target-dependent updates, 64 unseen noise samples reused across checkpoints. Metrics use raw tensors; PNG rendering alone clips to [-1,1].
- **Pixel SD:** compute standard deviation across samples at each channel/pixel, then average those standard deviations. This measures output variation, not perceptual quality. The reference from 1,024 real images is **0.493877**.
- **MSE to mean:** error between generated images and the per-pixel mean of the training dataset. This is distinct from error against an individual paired target.
- **FID:** not computed for these runs. Findings are limited to these architectures, seeds, and budgets.

For recurrent experiments E001/E002/E004, the core update is:

```python
next_x = model(x)
loss = mse(next_x, y)
optimizer.zero_grad(set_to_none=True)
loss.backward()
optimizer.step()
x = next_x.detach()
```

## E001 — Fresh-noise recurrent MSE baseline

**Question:** Can repeated direct MSE supervision teach a shared model to turn independent noise into an image through its own intermediate predictions?

**Setup:** one epoch; fresh `N(0,I)` noise for each image at its minibatch start; 20 detached transitions toward the same image target. The run contains 50,000 image visits, 1,000,000 supervised image-transition examples, and 391 minibatches × 20 updates = 7,820 updates.

**Observation:** most sample variation disappeared on the first application. Further applications converged toward nearly identical smooth gray images without recognizable objects.

| Frozen inference step | Pixel SD across unseen samples | MSE to dataset mean |
| --- | --- | --- |
| 1 | 0.00535407 | 0.00146556 |
| 5 | 0.000753569 | 0.00131438 |
| 10 | 0.000216426 | 0.00126487 |
| 20 | 0.0000268194 | 0.00127360 |

Mean training MSE across the epoch was 0.249861 at transition 1 and 0.247116 at transition 20. These online training losses are separate from frozen-model evaluation.

**Validation:** 391 training-log rows, exactly 50,000 image visits and 7,820 updates, finite losses, and five evaluation checkpoints (initialization plus each quarter epoch). A separate loop check confirmed detached inputs, exact pre-update state carry, and 20 optimizer updates.

**Reproduce from repo root** (use an unused output directory):

```sh
uv run train.py --device mps --output runs/cifar10-baseline-repeat
```

**Saved evidence:** [config](runs/cifar10-baseline/config.json), [summary](runs/cifar10-baseline/summary.json), [training losses](runs/cifar10-baseline/training.csv), [checkpoint evaluations](runs/cifar10-baseline/evaluation.jsonl), [rollout](runs/cifar10-baseline/trajectory_final.png), [final samples](runs/cifar10-baseline/samples_final.png), [final checkpoint](runs/cifar10-baseline/checkpoint.pt), and [source snapshot](runs/cifar10-baseline/train.py). Quarter-epoch checkpoints are in the same directory.

## E002 — Fixed pairs with progressive recurrent depth

**Question:** Does revisiting a fixed noise/image assignment, while gradually increasing the recurrent depth, allow the model to learn particular outputs instead of the mean?

**Changes from E001:** each original image index gets one fixed CPU-generated Gaussian noise tensor, using pair seed **2042**. Pairings survive reshuffling. Every minibatch starts from its original fixed noise, not an earlier epoch's generated state. The model starts fresh; optimizer state persists across the five epochs.

| Epoch | Transitions per minibatch | Image visits this epoch | Cumulative optimizer updates |
| --- | --- | --- | --- |
| 1 | 1 | 50,000 | 391 |
| 2 | 2 | 50,000 | 1,173 |
| 3 | 3 | 50,000 | 2,346 |
| 4 | 5 | 50,000 | 4,301 |
| 5 | 8 | 50,000 | 7,429 |

Total: **250,000 image visits and 950,000 supervised image-transition examples**. Architecture and optimizer match E001. Since pairing, epoch count, and depth schedule all changed, this is not a single-variable ablation.

**Evaluation:** in addition to unseen noise, monitor 64 fixed training pairs selected once and reused after every epoch. Compare reconstruction MSE against their assigned targets and a cyclic permutation of those same targets. The final trained depth is 8; generation through 20 steps is additional extrapolation, not a trained 20-step result.

**Observation:** both training seeds and unseen seeds produced nearly identical gray outputs. At step 8, monitored training-pair MSE was **0.23548585** against assigned targets versus **0.23548472** against permuted targets. There was no meaningful advantage for the assigned image in this run. The dataset-mean predictor's MSE on this subset was 0.23557006.

| Final model, frozen inference step | Pixel SD across unseen samples | MSE to dataset mean |
| --- | --- | --- |
| 1 | 0.00402502 | 0.000655798 |
| 3 | 0.000409581 | 0.000663012 |
| 5 | 0.000145664 | 0.000672270 |
| 8 — trained depth | 0.0000404463 | 0.000674827 |
| 20 — extrapolation | 0.000000273995 | 0.000674173 |

Mean training MSE in epoch 5 was 0.248362 at transition 1 and 0.247858 at transition 8. The experiment did not demonstrate learning the arbitrary fixed assignments at this model size and training budget; it does not establish that fixed assignments cannot be learned with other configurations.

**Validation:** all 50,000 original indices appeared exactly once per epoch; the fixed-noise bank's SHA-256 checksum was unchanged after training; all losses were finite; 1,955 minibatch rows and 7,429 updates matched expectations. `test_fixed.py` checked pairing through reshuffling, unchanged noise after updates, partial batches, independent evaluation noise, and update counts.

**Reproduce from repo root:**

```sh
uv run train_fixed.py --device mps --schedule 1 2 3 5 8 --output runs/fixed-progressive-repeat
uv run python -m unittest test_fixed -v
```

**Saved evidence:** [config and pairing checksum](runs/cifar10-fixed-progressive/config.json), [summary](runs/cifar10-fixed-progressive/summary.json), [training losses](runs/cifar10-fixed-progressive/training.csv), [epoch evaluations](runs/cifar10-fixed-progressive/evaluation.jsonl), [training targets and rollouts](runs/cifar10-fixed-progressive/train_pairs_epoch_05.png), [unseen-noise rollout](runs/cifar10-fixed-progressive/unseen_noise_epoch_05.png), [final checkpoint](runs/cifar10-fixed-progressive/checkpoint_epoch_05.pt), and [training source](runs/cifar10-fixed-progressive/train_fixed.py). Checkpoints, raw evaluation tensors, and images are saved for each epoch; the model-source snapshot is also in the run directory.

## E003 — Regression from uniformly interpolated images and noise

**Question:** Does training on informative image/noise mixtures, rather than on the model's own states, prevent uniform-gray outputs when evaluated from pure noise?

**Setup:** fresh initialization of the same U-Net and Adam settings as E001. For every supervised update, independently sample Gaussian noise `z` and a scalar image fraction `t ~ Uniform[0,1]` for each image. Form `x = (1-t)z + ty` and minimize `MSE(f(x), y)`. The model receives only `x`: no timestep input, velocity/noise target, or previous model prediction. There is no recurrence during training.

**Budget:** one pass over all 50,000 images; 20 independent corruption draws per image; 1,000,000 supervised image pairs and 7,820 updates. Batch size 128, seed 42, learning rate 1e-4, matching E001's model and update budget.
Mean sampled image fraction was 0.499725; mean training MSE was 0.066321; measured runtime was 357.7 seconds.

**Result:** Pure-noise inputs produce faint, distinct patterns after one step, which amplify into colored, blurry forms over 20 direct applications. This avoids the near-identical gray outputs of E001/E002/E004, but does not produce crisp recognizable samples. At step 20, 3.99% of raw values lie outside [-1,1]; increased variation includes saturation/drift and is not evidence of good image quality.

**Evaluation:** 64 pure-noise seeds (1042, shared with E001), applying `f` directly for 1–20 steps with frozen weights. This repeated composition is not a diffusion or flow solver. Separately, reconstruct the first 64 CIFAR-10 test images at fixed interpolation fractions using fixed noise seed 3042. No test images are used for training.

| Frozen generation step | Pixel SD across unseen samples | MSE to dataset mean | Fraction outside display range [-1,1] |
| --- | --- | --- | --- |
| 0 | 0.98826152 | 1.0076908 | 0.317464 |
| 1 | 0.086843818 | 0.0089770043 | 0 |
| 2 | 0.099129044 | 0.012053783 | 0 |
| 5 | 0.14165622 | 0.036254566 | 0 |
| 10 | 0.21289052 | 0.11457757 | 0.000605265 |
| 20 | 0.3701818 | 0.30092511 | 0.039861 |

| Image fraction in held-out input | Model MSE to target | Corrupted input MSE to target |
| --- | --- | --- |
| 0 | 0.26604384 | 1.2546284 |
| 0.1 | 0.17147395 | 1.0162491 |
| 0.2 | 0.10992868 | 0.80296218 |
| 0.5 | 0.038532615 | 0.3136571 |
| 0.8 | 0.010955388 | 0.050185136 |
| 1 | 0.0043841517 | 0 |

Dataset-mean predictor MSE on this held-out subset: 0.24327716. Noise-only inputs have no informative relationship to their paired target; the reconstruction score at image fraction zero is not an unconditional generation-quality metric.

**Validation:** every training index visited exactly once; 391 training rows, 7,820 updates, five generation evaluation checkpoints, and finite losses. Unit tests verify interpolation endpoints, per-image coefficients, fresh draws, and that training inputs are unchanged when model outputs change. PNGs clip for display; the metrics and stored tensors do not. No FID was computed.

**Reproduce from repo root:**

```sh
uv run train_interpolation.py --device mps --output runs/interpolation-repeat
uv run python -m unittest test_interpolation -v
```

**Saved evidence:** [config](runs/cifar10-interpolation/config.json), [summary](runs/cifar10-interpolation/summary.json), [training losses](runs/cifar10-interpolation/training.csv), [generation evaluations](runs/cifar10-interpolation/evaluation.jsonl), [held-out denoising evaluations](runs/cifar10-interpolation/denoising_evaluation.jsonl), [one-step samples](runs/cifar10-interpolation/samples_x1.png), [rollout](runs/cifar10-interpolation/trajectory_final.png), [held-out reconstructions](runs/cifar10-interpolation/heldout_denoising_seen_50000.png), [final checkpoint](runs/cifar10-interpolation/checkpoint_seen_50000.pt), and [source snapshot](runs/cifar10-interpolation/train_interpolation.py). Quarter-epoch checkpoints and final raw tensors are saved in the same directory.

## E004 — Continue the interpolation checkpoint on model-generated states

**Question:** Does the interpolation-trained prior survive or improve when training switches to the original recurrent objective?

**Setup:** continue E003 `checkpoint_seen_50000.pt`, restoring both model weights and Adam state (7,820 previous updates). Keep Adam learning rate 1e-4 and the same architecture. Start each minibatch from fresh independent Gaussian noise; target its assigned CIFAR-10 image throughout 20 transitions. Apply the existing detached, pre-update-output recurrence. No image/noise interpolations are included in this stage.

**Budget:** one additional pass over 50,000 images, 20 transitions each, 1,000,000 supervised image-transition pairs; 7,820 new updates (15,640 cumulative). Runtime 429.9 seconds. Seed 42; evaluation uses the same 64 noise seeds (1042) as E003.

**Result:** Near-identical gray outputs; step-20 variation collapses after the first 128-image batch (20 updates). The faint variation at x1 also decreases. Warm-starting this particular objective from E003 does not preserve its varied 20-step outputs. This is an observation about this configuration, not a claim about all possible fine-tuning schedules.

| Fine-tuning images seen | x1 pixel SD | x20 pixel SD | x20 MSE to dataset mean |
| --- | --- | --- | --- |
| 0 | 0.0868438184 | 0.370181799 | 0.300925106 |
| 128 | 0.0175776854 | 2.79331172e-08 | 0.00302306842 |
| 1,280 | 0.0139138987 | 2.55892516e-08 | 0.00152220216 |
| 6,400 | 0.0130852014 | 2.77432708e-08 | 0.00475253584 |
| 12,544 | 0.00857007504 | 2.31502213e-08 | 0.000718350289 |
| 25,088 | 0.00737331994 | 2.51757015e-08 | 0.000508847879 |
| 37,632 | 0.00610152492 | 1.86939335e-08 | 0.000432101428 |
| 50,000 | 0.00618236931 | 1.83925941e-08 | 0.00181189133 |

**Validation:** initial raw evaluation tensors exactly match E003 final tensors. All Adam counters end at 15,640, the parent checkpoint hash is unchanged, all losses are finite, and all 391 minibatches complete. Early frozen evaluations were added after 1, 10 and 50 minibatches, in addition to quarter-epoch evaluations. Existing unit tests and S004 pass. No FID computed.

**Reproduce:**

```sh
uv run train.py --device mps --resume runs/cifar10-interpolation/checkpoint_seen_50000.pt --output runs/rollout-repeat
```

**Saved evidence:** [config](runs/cifar10-interpolation-rollout/config.json), [summary](runs/cifar10-interpolation-rollout/summary.json), [losses](runs/cifar10-interpolation-rollout/training.csv), [evaluations](runs/cifar10-interpolation-rollout/evaluation.jsonl), [first-batch rollout](runs/cifar10-interpolation-rollout/trajectory_seen_00128.png), [final rollout](runs/cifar10-interpolation-rollout/trajectory_final.png), [final checkpoint](runs/cifar10-interpolation-rollout/checkpoint.pt), [source snapshot](runs/cifar10-interpolation-rollout/train.py).

## E005 — Categorical sampled rollouts

**Setup:** train a fresh network on all 50,000 CIFAR-10 training images for one epoch, with 20 sampled transitions per image. Quantize each RGB channel to its nearest of 200 evenly spaced values. Initialize each trajectory by independently sampling the full training dataset's per-position, per-channel color histograms. Predict 600 logits per pixel, apply mean cross-entropy against the same quantized target throughout the trajectory, sample each pixel/channel at temperature 1, update Adam, and carry the detached sample of pre-update logits. No MSE, time conditioning, or backpropagation through sampling.

**Budget:** same U-Net body, width 24; categorical head increases total parameters to 281,472. Adam 1e-4; batch 128, final batch 80; seed 42. 7,820 updates, 1,000,000 supervised image-transition pairs. Full histogram uses all 50,000 images. Evaluate 64 frozen trajectories using the same initial samples and sampling RNG seeds at each checkpoint. Evaluation restores training RNG state. Compare with independent histogram samples and quantized real images.

**Result:** The model produces colorful static throughout the sampled rollout; no recognizable objects or coherent regions are visible in the inspected final samples. At step 20, across-sample pixel SD is 0.49672, but adjacent-pixel absolute difference is 0.57399, close to independent histogram samples (0.56828) and far above quantized real images (0.11681). Thus sampling avoids uniform gray while this run shows no visible emergence of image structure across 20 passes. Similar aggregate statistics do not establish that the model exactly ignores its inputs or reproduces the histogram distribution.

Runtime: 929.4 seconds. Mean online training CE: 5.1703839 nats/channel/pixel. Fixed histogram CE on its source dataset: 5.1580868. These are training-set loss diagnostics, not held-out likelihood or quality scores.

| Source | Pixel SD across samples | Mean adjacent-pixel absolute difference |
| --- | --- | --- |
| Quantized real images (1,024) | 0.49380901 | 0.1168149 |
| Independent histogram samples (64) | 0.49184322 | 0.56827891 |
| Model x0 | 0.49169135 | 0.56771481 |
| Model x1 | 0.49540374 | 0.5729754 |
| Model x2 | 0.4963502 | 0.5739044 |
| Model x5 | 0.49699497 | 0.57370615 |
| Model x10 | 0.49575433 | 0.57253242 |
| Model x20 | 0.49672005 | 0.57399195 |

Adjacent-pixel differences measure roughness; they are not a semantic quality metric. Sample variation alone cannot distinguish static from images. No FID computed.

**Validation:** all 50,000 indices visited once; 391 batches and 7,820 updates; finite losses; five evaluation checkpoints; full histogram rows sum to one; final Adam counters equal 7,820; saved source hash verified; final x0 matches stored initial samples. Three targeted tests and the S005 GPU smoke run passed.

**Reproduce:**

```sh
uv run train_categorical.py --device mps --output runs/categorical-repeat
uv run python test_categorical.py
```

**Saved evidence:** [config](runs/cifar10-categorical/config.json), [summary](runs/cifar10-categorical/summary.json), [training losses](runs/cifar10-categorical/training.csv), [evaluations](runs/cifar10-categorical/evaluation.jsonl), [histogram samples](runs/cifar10-categorical/histogram_control.png), [rollout](runs/cifar10-categorical/trajectory_final.png), [final checkpoint](runs/cifar10-categorical/checkpoint_seen_50000.pt), [source snapshot](runs/cifar10-categorical/train_categorical.py).

## Smoke runs

These validate mechanics before full training; their sample quality is not treated as experimental evidence.

| ID | Details | Evidence |
| --- | --- | --- |
| S001 | First 256 training images, one epoch, 20 transitions, 40 updates; 3.44 s. | [Config](runs/smoke/config.json), [summary](runs/smoke/summary.json) |
| S002 | First 256 training images, fixed pairs, five-stage schedule, 1,280 image visits, 38 updates; 3.97 s. Pairing checksum and epoch coverage passed. | [Config](runs/fixed-progressive-smoke/config.json), [summary](runs/fixed-progressive-smoke/summary.json) |

**S003:** first 256 training images, 20 independent interpolation draws each, one epoch, 40 updates; 2.47 seconds. Completed with finite losses and full index coverage. [Config](runs/interpolation-smoke/config.json), [summary](runs/interpolation-smoke/summary.json).

**S004:** continue E003 on 256 images, 20 transitions each, 40 updates; 2.71 seconds. Exact initial output match and Adam counters advancing from 7,820 to 7,860 verified. First launch failed before training because older checkpoints stored TorchVersion metadata; loader allowlists this type and the rerun passed. [Config](runs/rollout-finetune-smoke/config.json), [summary](runs/rollout-finetune-smoke/summary.json).

**S005:** 256 training images, full-dataset initialization histograms, 20 transitions, 40 updates; 7.00 seconds. Three tests cover histogram/quantization correctness, categorical sampling frequencies, and detached sampled state carry with optimizer updates. [Config](runs/categorical-smoke/config.json), [summary](runs/categorical-smoke/summary.json).

## Adding a run

Append each new experiment with a stable ID, date, question, changes from its reference run, exact command, pairing and optimizer/state timing, seeds, training budget, validation, observations, and links to its config, summary, samples, and checkpoint. Label smoke tests and interrupted runs explicitly. Preserve prior entries and artifact directories; distinguish measured results from proposed follow-ups. New runs should use an unused output directory.

Run artifacts live under `runs/`, which is Git-ignored; their links work only where the artifacts have been retained. This ledger preserves the main findings and settings in a repository file. Saved source snapshots and config hashes identify the code used for each run even if the working scripts later change.
