import os
import torch
import torch.nn.functional as F
from pathlib import Path
import matplotlib.pyplot as plt
from torchvision import transforms, utils as vutils
from torchvision.utils import save_image
from diffusers import UNet2DModel, DDPMScheduler
import numpy as np
import imageio
from PIL import Image
from tqdm import tqdm

# CIFAR-10 class names
CIFAR10_CLASSES = [
    'airplane', 'automobile', 'bird', 'cat', 'deer',
    'dog', 'frog', 'horse', 'ship', 'truck'
]

def load_model_and_scheduler(ckpt_path, device):
    """Load the trained model and scheduler"""
    print(f"Loading checkpoint from {ckpt_path}")
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    
    # Get config from checkpoint
    if "config" in ckpt:
        model_config = ckpt["config"]["model_config"]
        num_classes = ckpt["config"]["num_classes"]
        null_class_id = ckpt["config"]["null_class_id"]
        image_size = ckpt["config"]["image_size"]
    else:
        # Default config (if no config info is found in checkpoint)
        model_config = {
            'sample_size': 32,
            'in_channels': 3,
            'out_channels': 3,
            'down_block_types': ("DownBlock2D","DownBlock2D","AttnDownBlock2D","DownBlock2D"),
            'up_block_types': ("UpBlock2D","AttnUpBlock2D","UpBlock2D","UpBlock2D"),
            'block_out_channels': (128, 256, 256, 256),
            'layers_per_block': 8,
            'num_class_embeds': 11,  # 10 classes + 1 null
        }
        num_classes = 10
        null_class_id = 10
        image_size = 32
    
    # Create model
    model = UNet2DModel(**model_config).to(device)
    
    # Load weights
    if "ema" in ckpt:
        # Use EMA weights
        model_state = {}
        for k, v in ckpt["ema"].items():
            model_state[k] = v
        model.load_state_dict(model_state)
        print("✓ Loaded EMA weights")
    elif "model" in ckpt:
        # Use regular weights
        model.load_state_dict(ckpt["model"])
        print("✓ Loaded model weights")
    else:
        raise ValueError("No valid model weights found in checkpoint")
    
    model.eval()
    
    # Create scheduler
    scheduler = DDPMScheduler(num_train_timesteps=1000)
    
    step = ckpt.get("step", 0)
    print(f"✓ Model loaded from step {step}")
    
    return model, scheduler, num_classes, null_class_id, image_size

@torch.no_grad()
def sample_with_process_visualization(
    model, scheduler, device, class_label, null_class_id,
    cfg_scale=3.0, num_inference_steps=50, seed=42, save_every=1
):
    """Generate an image and save the entire denoising process"""
    
    # Create output directory
    class_name = CIFAR10_CLASSES[class_label]
    save_dir = Path(f'./cifar10_generation_class_{class_label}_{class_name}')
    save_dir.mkdir(exist_ok=True)
    
    # Set random seed
    generator = torch.Generator(device=device).manual_seed(seed)
    
    # Initialize scheduler
    scheduler.set_timesteps(num_inference_steps)
    
    # Prepare labels
    y_cond = torch.tensor([class_label], device=device, dtype=torch.long)
    y_null = torch.tensor([null_class_id], device=device, dtype=torch.long)
    
    # Initial noise
    x = torch.randn((1, 3, 32, 32), device=device, generator=generator)
    x_null = x.clone()
    
    # Save all intermediate steps
    all_images = []
    step_info = []
    
    print(f"Starting generation for CIFAR-10 class: {class_name} (label: {class_label})")
    print(f"CFG scale: {cfg_scale}, inference steps: {num_inference_steps}")
    
    # Save initial noise
    save_step_image(x, save_dir, 0, scheduler.timesteps[0].item(), "noise")
    all_images.append(denormalize_image(x).cpu())
    step_info.append(("Initial noise", scheduler.timesteps[0].item()))
    
    # Denoising loop
    for i, timestep in enumerate(tqdm(scheduler.timesteps, desc="Denoising")):
        # Conditional prediction
        noise_pred_cond = model(x, timestep, class_labels=y_cond).sample
        
        # Unconditional prediction
        noise_pred_uncond = model(x_null, timestep, class_labels=y_null).sample
        
        # Classifier-Free Guidance
        noise_pred = noise_pred_uncond + cfg_scale * (noise_pred_cond - noise_pred_uncond)
        
        # Scheduler step
        x = scheduler.step(model_output=noise_pred, timestep=timestep, sample=x).prev_sample
        x_null = x.clone()  # keep in sync
        
        # Save intermediate steps
        if i % save_every == 0 or i == len(scheduler.timesteps) - 1:
            save_step_image(x, save_dir, i+1, timestep.item(), f"step")
            all_images.append(denormalize_image(x).cpu())
            step_info.append((f"Step {i+1}", timestep.item()))
            
            if i % 10 == 0:
                print(f"  Step {i+1}/{len(scheduler.timesteps)}, timestep={timestep}")
    
    print(f"✅ Generation complete! Images saved to: {save_dir}")
    
    # Create process visualizations
    create_process_grid(all_images, step_info, class_label, class_name, save_dir)
    create_process_video(all_images, step_info, class_label, class_name, save_dir, fps=10)
    
    return x, save_dir

def denormalize_image(tensor):
    """Convert tensor from [-1,1] to [0,1] for display"""
    return torch.clamp((tensor + 1.0) / 2.0, 0.0, 1.0)

def save_step_image(tensor, save_dir, step_idx, timestep, prefix):
    """Save the image for a single step"""
    img_norm = denormalize_image(tensor)
    filename = f"{prefix}_{step_idx:04d}_t{timestep:04.0f}.png"
    save_image(img_norm, save_dir / filename)

def create_process_grid(images_list, step_info, class_label, class_name, save_dir):
    """Create a grid image showing the denoising process"""
    
    # Select key steps to display
    n_total = len(images_list)
    if n_total > 20:
        # If too many steps, pick some key ones
        indices = [0, 1, 2, 3, 4, 5, 10, 15, 20, 25, 30, 40, n_total-1]
        indices = [i for i in indices if i < n_total]
    else:
        indices = list(range(n_total))
    
    selected_images = [images_list[i] for i in indices]
    selected_info = [step_info[i] for i in indices]
    
    # Compute grid layout
    n_images = len(selected_images)
    cols = min(6, n_images)
    rows = (n_images + cols - 1) // cols
    
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 3, rows * 3))
    if rows == 1:
        axes = axes.reshape(1, -1)
    
    for i, (img, (desc, timestep)) in enumerate(zip(selected_images, selected_info)):
        row = i // cols
        col = i % cols
        
        # Convert image format for display
        img_np = img.squeeze().permute(1, 2, 0).numpy()
        
        axes[row, col].imshow(img_np)
        axes[row, col].set_title(f'{desc}\nt={timestep:.0f}', fontsize=10)
        axes[row, col].axis('off')
    
    # Hide extra subplots
    for i in range(n_images, rows * cols):
        row = i // cols
        col = i % cols
        axes[row, col].axis('off')
    
    plt.suptitle(f'CIFAR-10 generation process - {class_name} (class {class_label})', fontsize=16)
    plt.tight_layout()
    
    # Save grid image
    grid_path = save_dir / f"generation_grid_{class_name}.png"
    plt.savefig(grid_path, dpi=150, bbox_inches='tight')
    plt.show()
    
    print(f"Process grid image saved to: {grid_path}")

def create_process_video(images_list, step_info, class_label, class_name, save_dir, fps=10):
    """Create a video of the denoising process"""
    
    print(f"Creating video... (total {len(images_list)} frames)")
    
    # Video file path
    video_path = save_dir / f"generation_process_{class_name}.mp4"
    
    try:
        # Upscale images for better viewing (from 32x32 to 320x320)
        upscaled_images = []
        for i, (img, (desc, timestep)) in enumerate(zip(images_list, step_info)):
            # Convert to numpy array
            img_np = img.squeeze().permute(1, 2, 0).numpy()
            img_np = (img_np * 255).astype(np.uint8)
            
            # Upscale with PIL
            pil_img = Image.fromarray(img_np)
            upscaled_img = pil_img.resize((320, 320), Image.NEAREST)
            upscaled_images.append(np.array(upscaled_img))
        
        # Create video
        with imageio.get_writer(video_path, fps=fps, codec='libx264') as writer:
            for i, img in enumerate(upscaled_images):
                writer.append_data(img)
                
                if i % 10 == 0:
                    print(f"  Progress: {i+1}/{len(upscaled_images)} frames")
        
        print(f"✅ Video created successfully!")
        print(f"   - Video path: {video_path}")
        print(f"   - FPS: {fps} FPS")
        print(f"   - Total frames: {len(images_list)}")
        print(f"   - Duration: {len(images_list)/fps:.2f} seconds")
        print(f"   - Resolution: 320x320 (upscaled from original 32x32)")
        
    except Exception as e:
        print(f"❌ Failed to create video: {e}")
        print("Please make sure imageio and imageio-ffmpeg are installed:")
        print("pip install imageio imageio-ffmpeg")

@torch.no_grad()
def generate_multiple_samples(model, scheduler, device, null_class_id, 
                            class_label=None, num_samples=16, cfg_scale=3.0, 
                            num_inference_steps=50, seed=42):
    """Generate a grid of multiple samples"""
    
    if class_label is not None:
        class_name = CIFAR10_CLASSES[class_label]
        print(f"Generating {num_samples} samples of class {class_name}...")
        y_cond = torch.full((num_samples,), class_label, device=device, dtype=torch.long)
    else:
        print(f"Generating {num_samples} samples of random classes...")
        y_cond = torch.randint(0, 10, (num_samples,), device=device)
    
    y_null = torch.full((num_samples,), null_class_id, device=device, dtype=torch.long)
    
    # Set scheduler
    scheduler.set_timesteps(num_inference_steps)
    
    # Initial noise
    generator = torch.Generator(device=device).manual_seed(seed)
    x = torch.randn((num_samples, 3, 32, 32), device=device, generator=generator)
    x_null = x.clone()
    
    # Denoising loop
    for timestep in tqdm(scheduler.timesteps, desc="Generating samples"):
        # Conditional and unconditional predictions
        noise_pred_cond = model(x, timestep, class_labels=y_cond).sample
        noise_pred_uncond = model(x_null, timestep, class_labels=y_null).sample
        
        # CFG
        noise_pred = noise_pred_uncond + cfg_scale * (noise_pred_cond - noise_pred_uncond)
        
        # Scheduler step
        x = scheduler.step(model_output=noise_pred, timestep=timestep, sample=x).prev_sample
        x_null = x.clone()
    
    # Save results
    images = denormalize_image(x)
    
    if class_label is not None:
        save_path = f"./cifar10_samples_{class_name}_{num_samples}samples.png"
    else:
        save_path = f"./cifar10_samples_mixed_{num_samples}samples.png"
    
    # Create grid
    grid = vutils.make_grid(images, nrow=int(np.sqrt(num_samples)), padding=2)
    save_image(grid, save_path)
    
    print(f"✅ Sample grid saved to: {save_path}")
    
    return images

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Check dependencies
    try:
        import imageio
    except ImportError:
        print("❌ Missing imageio package, please run: pip install imageio imageio-ffmpeg")
        return
    
    # Model path
    ckpt_path = "./cifar10_runs/checkpoints/latest.pt"  # e.g. latest.pt, best.pt
    
    try:
        # Load model
        print("Loading model...")
        model, scheduler, num_classes, null_class_id, image_size = load_model_and_scheduler(ckpt_path, device)
        print("Model loaded successfully!")
        
        # Show class info
        print("\n📋 CIFAR-10 classes:")
        for i, cls_name in enumerate(CIFAR10_CLASSES):
            print(f"  {i}: {cls_name}")
        
        while True:
            print("\n" + "="*50)
            print("Select a function:")
            print("1. Generate a single image and show the full process")
            print("2. Generate a grid of multiple samples")
            print("3. Exit")
            
            choice = input("Please choose (1-3): ").strip()
            
            if choice == '1':
                # Single image generation
                try:
                    class_input = input(f"Enter class (0-9) or class name: ").strip()
                    
                    # Parse input
                    if class_input.isdigit():
                        class_label = int(class_input)
                        if class_label < 0 or class_label >= num_classes:
                            print(f"Class must be between 0 and {num_classes-1}!")
                            continue
                    else:
                        # Try to match class name
                        class_label = None
                        for i, name in enumerate(CIFAR10_CLASSES):
                            if name.lower() == class_input.lower():
                                class_label = i
                                break
                        if class_label is None:
                            print("No matching class name found!")
                            continue
                    
                    # Get parameters
                    cfg_scale = float(input("CFG scale (default 3.0): ").strip() or "3.0")
                    steps = int(input("Inference steps (default 50): ").strip() or "50")
                    seed = int(input("Random seed (default 42): ").strip() or "42")
                    
                    print(f"\nStarting generation for class {class_label} ({CIFAR10_CLASSES[class_label]})...")
                    
                    # Generation process
                    final_image, save_dir = sample_with_process_visualization(
                        model, scheduler, device, class_label, null_class_id,
                        cfg_scale=cfg_scale, num_inference_steps=steps, seed=seed
                    )
                    
                    print(f"\n🎉 Done! Results saved to: {save_dir}")
                    
                except ValueError as e:
                    print(f"Input error: {e}")
                except Exception as e:
                    print(f"Error during generation: {e}")
            
            elif choice == '2':
                # Multiple sample generation
                try:
                    class_input = input("Class (0-9, name, or press Enter for mixed): ").strip()
                    
                    class_label = None
                    if class_input:
                        if class_input.isdigit():
                            class_label = int(class_input)
                        else:
                            for i, name in enumerate(CIFAR10_CLASSES):
                                if name.lower() == class_input.lower():
                                    class_label = i
                                    break
                    
                    num_samples = int(input("Number of samples (default 16): ").strip() or "16")
                    cfg_scale = float(input("CFG scale (default 3.0): ").strip() or "3.0")
                    steps = int(input("Inference steps (default 50): ").strip() or "50")
                    seed = int(input("Random seed (default 42): ").strip() or "42")
                    
                    # Generate samples
                    images = generate_multiple_samples(
                        model, scheduler, device, null_class_id,
                        class_label=class_label, num_samples=num_samples,
                        cfg_scale=cfg_scale, num_inference_steps=steps, seed=seed
                    )
                    
                except ValueError as e:
                    print(f"Input error: {e}")
                except Exception as e:
                    print(f"Error during generation: {e}")
            
            elif choice == '3':
                print("Goodbye!")
                break
            
            else:
                print("Invalid choice, please try again!")
    
    except FileNotFoundError:
        print(f"Model file {ckpt_path} not found!")
        print("Please make sure that:")
        print("1. You have already run the training code")
        print("2. The model file path is correct")
        print("3. Possible filenames: latest.pt, best.pt, ema_only.pt")
    except Exception as e:
        print(f"An error occurred while loading the model: {e}")

if __name__ == "__main__":
    main()
