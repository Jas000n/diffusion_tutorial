# CIFAR-10 conditional diffusion training with diffusers UNet2DModel
# - Class-conditional via num_class_embeds
# - Classifier-Free Guidance (drop cond during training)
# - Step-based training (target_steps), not epochs
# - EMA, AMP (fp16/bf16), grad accumulation
# - Periodic checkpointing & sampling
# - Resume from checkpoint support
# - Progress bar with tqdm

import os, math, random, time, argparse
from pathlib import Path
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import datasets, transforms, utils as vutils
from diffusers import UNet2DModel, DDPMScheduler
from tqdm import tqdm

# -----------------------
# Args & Config
# -----------------------
parser = argparse.ArgumentParser()
parser.add_argument("--resume", type=str, default="", help="Path to checkpoint to resume from")
parser.add_argument("--auto_resume", action="store_true", help="Auto resume from latest.pt if exists")


args = parser.parse_args()

seed = 42
torch.manual_seed(seed); random.seed(seed)
torch.backends.cudnn.benchmark = True
device = "cuda" if torch.cuda.is_available() else "cpu"

# Data
image_size    = 32                  # CIFAR-10 native
num_classes   = 10                  # CIFAR-10  
null_class_id = num_classes         # for CFG null token
data_root     = "./data"            # will auto-download

# Train schedule
batch_size            = 16         # physical batch per step
grad_accum_steps      = 2           # effective batch = 512 (256 * 2)
target_steps          = 300_000     # total optimizer steps
log_every             = 500         
sample_every          = 10_000     
ckpt_every            = 10_000      
save_dir              = Path("./cifar10_runs"); save_dir.mkdir(parents=True, exist_ok=True)
ckpt_dir              = save_dir / "checkpoints"; ckpt_dir.mkdir(exist_ok=True, parents=True)
sample_dir            = save_dir / "samples"; sample_dir.mkdir(exist_ok=True, parents=True)

# Optim & misc
lr                    = 2e-4
weight_decay          = 1e-2
grad_clip_norm        = 1.0
ema_decay             = 0.9999
cfg_drop_p            = 0.15        # probability to drop condition during training
use_bf16              = torch.cuda.is_available() and torch.cuda.get_device_capability()[0] >= 8
amp_dtype             = torch.bfloat16 if use_bf16 else torch.float16
amp_enabled           = (device == "cuda")



# -----------------------
# DataLoader
# -----------------------
tfm = transforms.Compose([
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize([0.5]*3, [0.5]*3),
])
train_set = datasets.CIFAR10(root=data_root, train=True, download=True, transform=tfm)
train_ld  = DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=16, pin_memory=True, drop_last=True)

# -----------------------
# Model & Noise Scheduler
# -----------------------
model = UNet2DModel(
    sample_size=image_size,
    in_channels=3,
    out_channels=3,
    down_block_types=("DownBlock2D","DownBlock2D","AttnDownBlock2D","DownBlock2D"),
    up_block_types=("UpBlock2D","AttnUpBlock2D","UpBlock2D","UpBlock2D"),
    block_out_channels=(128, 256, 256, 256),
    layers_per_block=8,
    num_class_embeds=num_classes + 1,  # +1 for null class (CFG)
).to(device)

noise_scheduler = DDPMScheduler(num_train_timesteps=1000)

# -----------------------
# EMA helper
# -----------------------
class EMA:
    def __init__(self, model, decay=0.9999):
        self.decay = decay
        self.shadow = {k: v.detach().clone() for k, v in model.state_dict().items() if v.dtype.is_floating_point}

    @torch.no_grad()
    def update(self, model):
        for k, v in model.state_dict().items():
            if k in self.shadow and v.dtype.is_floating_point:
                self.shadow[k].mul_(self.decay).add_(v, alpha=1.0 - self.decay)

    @torch.no_grad()
    def copy_to(self, model):
        msd = model.state_dict()
        for k, v in self.shadow.items():
            if k in msd:
                msd[k].copy_(v)

ema = EMA(model, decay=ema_decay)

# -----------------------
# Optimizer & LR schedule
# -----------------------
optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay, betas=(0.9, 0.999))
scaler = torch.cuda.amp.GradScaler(enabled=(amp_enabled and (not use_bf16)))

warmup_steps = 10_000
def lr_schedule(step):
    if step < warmup_steps:
        return step / max(1, warmup_steps)
    # cosine decay to 10% of base lr
    progress = (step - warmup_steps) / max(1, target_steps - warmup_steps)
    return 0.1 + 0.9 * 0.5 * (1 + math.cos(math.pi * min(progress, 1.0)))

# -----------------------
# Checkpoint Resume Logic
# -----------------------
def save_checkpoint(step, model, optimizer, scaler, ema, rng_states, is_best=False):
    """Save training checkpoint with all necessary states"""
    ckpt = {
        "step": step,
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(), 
        "scaler": scaler.state_dict(),
        "ema": ema.shadow,
        "rng_states": rng_states,
        "config": {
            "num_classes": num_classes,
            "null_class_id": null_class_id,
            "image_size": image_size,
            "model_config": model.config,
        },
    }
    # Save regular checkpoint
    path_step = ckpt_dir / f"step_{step:07d}.pt"
    torch.save(ckpt, path_step)
    torch.save(ckpt, ckpt_dir / "latest.pt")
    tqdm.write(f"[Checkpoint] saved at step {step} -> {path_step}")
    # Save best if specified
    if is_best:
        torch.save(ckpt, ckpt_dir / "best.pt")
        tqdm.write(f"[Checkpoint] best updated")

def load_checkpoint(ckpt_path, model, optimizer, scaler, ema):
    """Load checkpoint and restore all states"""
    print(f"Loading checkpoint from {ckpt_path}")
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)

    # Restore model
    model.load_state_dict(ckpt["model"])
    print(f"✓ Model state restored")

    # Restore optimizer
    optimizer.load_state_dict(ckpt["optimizer"])
    print(f"✓ Optimizer state restored")

    # Restore scaler
    if "scaler" in ckpt:
        try:
            scaler.load_state_dict(ckpt["scaler"])
            print(f"✓ Scaler state restored")
        except Exception as e:
            print(f"⚠ Skipped scaler state: {e}")

    # Restore EMA
    if "ema" in ckpt:
        ema.shadow = ckpt["ema"]
        print(f"✓ EMA state restored")

    # Restore RNG states for reproducibility
    if "rng_states" in ckpt:
        try:
            torch.set_rng_state(ckpt["rng_states"]["torch"])
            if torch.cuda.is_available():
                torch.cuda.set_rng_state_all(ckpt["rng_states"]["cuda"])
            random.setstate(ckpt["rng_states"]["python"])
            print(f"✓ RNG states restored")
        except Exception as e:
            print(f"⚠ Skipped RNG restore: {e}")

    step = ckpt["step"]
    print(f"✓ Resuming from step {step}")
    return step

def get_rng_states():
    """Capture current RNG states for checkpointing"""
    return {
        "torch": torch.get_rng_state(),
        "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else [],
        "python": random.getstate(),
    }

# Determine starting step
global_step = 0
resume_path = None

if args.resume:
    resume_path = args.resume
elif args.auto_resume:
    latest_path = ckpt_dir / "latest.pt"
    if latest_path.exists():
        resume_path = latest_path

if resume_path:
    global_step = load_checkpoint(resume_path, model, optimizer, scaler, ema)
    print(f"🚀 Resumed training from step {global_step}")
else:
    print(f"🆕 Starting fresh training")


# -----------------------
# Sampling function
# -----------------------
@torch.no_grad()
def sample_grid(step, nrow=10, per_class=10, cfg_scale=3.0, num_inference_steps=50, seed=1234):
    # Use EMA weights for sampling
    tmp = UNet2DModel(**model.config).to(device)
    ema.copy_to(tmp)
    tmp.eval()

    scheduler = DDPMScheduler(num_train_timesteps=1000)
    scheduler.set_timesteps(num_inference_steps)

    gens = []
    g = torch.Generator(device=device).manual_seed(seed)

    for cls in range(num_classes):
        y = torch.full((per_class,), cls, device=device, dtype=torch.long)
        y_null = torch.full((per_class,), null_class_id, device=device, dtype=torch.long)
        x = torch.randn((per_class, 3, image_size, image_size), device=device, generator=g)
        x_null = x.clone()

        for t in scheduler.timesteps:
            with torch.autocast(device_type="cuda", dtype=amp_dtype, enabled=amp_enabled):
                eps   = tmp(x, t, class_labels=y).sample
                eps_u = tmp(x_null, t, class_labels=y_null).sample
            eps_cfg = eps_u + cfg_scale * (eps - eps_u)
            x = scheduler.step(model_output=eps_cfg, timestep=t, sample=x).prev_sample
            x_null = x

        gens.append(x)

    imgs = torch.cat(gens, dim=0)
    imgs = torch.clamp(imgs, -1, 1)
    grid = vutils.make_grid((imgs + 1) * 0.5, nrow=nrow)  # [0,1]

    out_path = sample_dir / f"step_{step:07d}.png"
    vutils.save_image(grid, out_path)
    tqdm.write(f"[Sample] saved {out_path}")


# -----------------------
# Training Loop with Progress Bar
# -----------------------
best_loss = float("inf")
running_loss = 0.0
model.train()

print(f"🎯 Target steps: {target_steps:,} | Starting from: {global_step:,}")

# Initialize progress bar
pbar = tqdm(
    total=target_steps,
    initial=global_step,
    desc="Training",
    unit="step",
    dynamic_ncols=True,
    bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}] {postfix}"
)

# Create infinite data iterator for step-based training
def infinite_dataloader(dataloader):
    while True:
        for batch in dataloader:
            yield batch

data_iter = infinite_dataloader(train_ld)

while global_step < target_steps:
    x0, y = next(data_iter)
    x0 = x0.to(device, non_blocking=True)
    y  = y.to(device, non_blocking=True)

    # CFG drop: set part of labels to null
    if cfg_drop_p > 0:
        drop_mask = (torch.rand_like(y.float()) < cfg_drop_p)
        y_cf = y.clone()
        y_cf[drop_mask] = null_class_id
    else:
        y_cf = y

    b = x0.size(0)
    t = torch.randint(0, noise_scheduler.config.num_train_timesteps, (b,), device=device, dtype=torch.long)
    noise = torch.randn_like(x0)
    xt = noise_scheduler.add_noise(x0, noise, t)

    # forward + loss
    with torch.autocast(device_type="cuda", dtype=amp_dtype, enabled=amp_enabled):
        noise_pred = model(xt, t, class_labels=y_cf).sample
        loss = F.mse_loss(noise_pred, noise) / grad_accum_steps

    scaler.scale(loss).backward()

    if (global_step + 1) % grad_accum_steps == 0:
        # grad clip, step
        if grad_clip_norm is not None:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip_norm)

        # update lr
        current_lr = lr * lr_schedule(global_step)
        for pg in optimizer.param_groups:
            pg["lr"] = current_lr

        scaler.step(optimizer)
        scaler.update()
        optimizer.zero_grad(set_to_none=True)

        # EMA update
        ema.update(model)

    # accumulate loss for logging
    running_loss += loss.item() * grad_accum_steps

    # progress bar + periodic log
    current_lr = optimizer.param_groups[0]['lr']
    if (global_step + 1) % log_every == 0:
        avg_loss = running_loss / log_every
        pbar.set_postfix({
            'loss': f'{avg_loss:.4f}',
            'lr': f'{current_lr:.2e}',
            'best': f'{best_loss:.4f}' if best_loss != float("inf") else 'N/A'
        })
        running_loss = 0.0
    else:
        # Update postfix less frequently to reduce overhead
        if global_step % 50 == 0:
            denom = ((global_step + 1) % log_every)
            recent_loss = running_loss / max(1, denom)
            pbar.set_postfix({
                'loss': f'{recent_loss:.4f}',
                'lr': f'{current_lr:.2e}',
                'best': f'{best_loss:.4f}' if best_loss != float("inf") else 'N/A'
            })

    # periodic sampling
    if (global_step + 1) % sample_every == 0:
        model.eval()
        try:
            sample_grid(global_step + 1, nrow=10, per_class=10, cfg_scale=3.0, num_inference_steps=50)
        except Exception as e:
            tqdm.write(f"[Sample] failed: {e}")
        model.train()

    # checkpointing
    if (global_step + 1) % ckpt_every == 0:
        steps_since_log = (global_step + 1) % log_every
        if steps_since_log == 0:
            avg = running_loss / log_every if running_loss > 0 else best_loss
        else:
            avg = running_loss / steps_since_log if running_loss > 0 else best_loss

        is_best = avg < best_loss
        if is_best:
            best_loss = avg

        rng_states = get_rng_states()
        save_checkpoint(global_step + 1, model, optimizer, scaler, ema, rng_states, is_best)
    # Update progress bar
    pbar.update(1)
    global_step += 1

# Close progress bar
pbar.close()

print("🎉 Training completed!")

# Final EMA export
final_ema = {
    "step": global_step,
    "ema": ema.shadow,
    "model_config": model.config,
}
torch.save(final_ema, ckpt_dir / "ema_only.pt")
print("💾 Saved EMA weights.")