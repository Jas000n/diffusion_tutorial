import os
import torch
import torchvision.transforms as transforms
from unet import UNet
import matplotlib.pyplot as plt
from torchvision.utils import save_image
import numpy as np
import imageio
from PIL import Image


# ========= Automatically find the latest model =========
def find_latest_model(model_dir="."):
    """
    Automatically find the latest unet_diffusion_*.pth model in the directory:
    1. If 'unet_diffusion_final.pth' exists, use it.
    2. Otherwise, use the checkpoint with the highest epoch number.
    """
    model_files = [
        f for f in os.listdir(model_dir)
        if f.startswith("unet_diffusion_") and f.endswith(".pth")
    ]
    if not model_files:
        raise FileNotFoundError("No model checkpoint found in the directory.")

    # 1. Prefer the final model
    if "unet_diffusion_final.pth" in model_files:
        print("✅ Found final model: unet_diffusion_final.pth")
        return os.path.join(model_dir, "unet_diffusion_final.pth")

    # 2. Otherwise, find the latest epoch model
    epoch_models = []
    for f in model_files:
        if "epoch_" in f:
            try:
                epoch_num = int(f.split("epoch_")[1].split(".pth")[0])
                epoch_models.append((epoch_num, f))
            except ValueError:
                continue

    if epoch_models:
        latest_epoch, latest_file = max(epoch_models, key=lambda x: x[0])
        print(f"✅ Found latest checkpoint: {latest_file} (epoch {latest_epoch})")
        return os.path.join(model_dir, latest_file)

    raise FileNotFoundError("No valid model checkpoint (with epoch_) found.")


# ========= Load model and diffusion parameters =========
def load_model_and_params(model_path, device):
    timesteps = 1000
    betas = torch.linspace(0.0001, 0.02, timesteps).to(device)
    alphas = 1 - betas
    alphas_cumprod = torch.cumprod(alphas, dim=0)
    
    unet = UNet(in_channels=1, out_channels=1).to(device)
    state_dict = torch.load(model_path, map_location=device)
    unet.load_state_dict(state_dict)
    unet.eval()
    
    print(f"✅ Model loaded from: {model_path}")
    return unet, betas, alphas, alphas_cumprod, timesteps


# ========= Sampling with denoising visualization =========
@torch.no_grad()
def sample_with_visualization(unet, device, class_label, betas, alphas, alphas_cumprod, timesteps, save_steps=1):
    """Generate a single sample and save the denoising process."""
    
    save_dir = f'./denoising_process_class_{class_label}'
    os.makedirs(save_dir, exist_ok=True)
    
    x = torch.randn(1, 1, 28, 28, device=device)
    class_labels = torch.tensor([class_label], device=device)
    save_image((x + 1) / 2, f'{save_dir}/step_0000_noise.png')
    
    images_to_show = []
    all_images = []
    step_indices = []
    
    print(f"Starting denoising process for digit {class_label}...")
    
    for i, t in enumerate(reversed(range(timesteps))):
        t_tensor = torch.full((1,), t, device=device, dtype=torch.long)
        predicted_noise = unet(x, time_step=t_tensor, class_label=class_labels)
        
        alpha_t = alphas[t].to(device)
        alpha_cumprod_t = alphas_cumprod[t].to(device)
        beta_t = betas[t].to(device)
        
        pred_x0 = (x - torch.sqrt(1 - alpha_cumprod_t) * predicted_noise) / torch.sqrt(alpha_cumprod_t)
        pred_x0 = torch.clamp(pred_x0, -1, 1)
        
        if t > 0:
            noise = torch.randn_like(x)
            x = torch.sqrt(alpha_t) * (x - beta_t / torch.sqrt(1 - alpha_cumprod_t) * predicted_noise) + torch.sqrt(beta_t) * noise
        else:
            x = pred_x0
        
        img_normalized = (x + 1) / 2
        img_normalized = torch.clamp(img_normalized, 0, 1)
        
        img_np = img_normalized.cpu().squeeze().numpy()
        img_np = (img_np * 255).astype(np.uint8)
        all_images.append(img_np)
        
        if i % 50 == 0 or t == 0:
            save_image(img_normalized, f'{save_dir}/step_{i:04d}_t_{t:04d}.png')
            images_to_show.append(img_normalized.cpu())
            step_indices.append(t)
            print(f"  Step {i:4d}/1000, t={t:4d}")
    
    create_denoising_grid(images_to_show, step_indices, class_label, save_dir)
    create_denoising_video(all_images, class_label, save_dir, fps=60)
    
    return x, save_dir


def create_denoising_grid(images_list, step_indices, class_label, save_dir):
    """Create a grid image showing the denoising steps."""
    n_images = len(images_list)
    cols = min(8, n_images)
    rows = (n_images + cols - 1) // cols
    
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 2, rows * 2))
    if rows == 1:
        axes = axes.reshape(1, -1)
    
    for i, (img, step) in enumerate(zip(images_list, step_indices)):
        row, col = i // cols, i % cols
        axes[row, col].imshow(img.squeeze(), cmap='gray')
        axes[row, col].set_title(f't={step}', fontsize=10)
        axes[row, col].axis('off')
    
    for i in range(n_images, rows * cols):
        row, col = i // cols, i % cols
        axes[row, col].axis('off')
    
    plt.suptitle(f'Denoising Process - Digit {class_label}', fontsize=16)
    plt.tight_layout()
    grid_path = f'{save_dir}/denoising_grid.png'
    plt.savefig(grid_path, dpi=150, bbox_inches='tight')
    plt.show()
    
    print(f"Denoising grid saved to: {grid_path}")


def create_denoising_video(images_list, class_label, save_dir, fps=60):
    """Convert image sequence into a video."""
    print(f"Creating video... (Total {len(images_list)} frames)")
    
    video_path = f'{save_dir}/denoising_process_class_{class_label}.mp4'
    
    try:
        upscaled_images = []
        for img in images_list:
            pil_img = Image.fromarray(img, mode='L')
            upscaled_img = pil_img.resize((280, 280), Image.NEAREST)
            upscaled_images.append(np.array(upscaled_img))
        
        with imageio.get_writer(video_path, fps=fps, codec='libx264') as writer:
            for i, img in enumerate(upscaled_images):
                rgb_img = np.stack([img, img, img], axis=2)
                writer.append_data(rgb_img)
                if i % 100 == 0:
                    print(f"  Progress: {i}/{len(upscaled_images)} frames")
        
        print(f"✅ Video created successfully!")
        print(f"   - Path: {video_path}")
        print(f"   - FPS: {fps}")
        print(f"   - Total frames: {len(images_list)}")
        print(f"   - Duration: {len(images_list)/fps:.2f} seconds")
        print(f"   - Resolution: 280x280 (upscaled from 28x28)")
        
    except Exception as e:
        print(f"❌ Failed to create video: {e}")
        print("Please ensure 'imageio' and 'imageio-ffmpeg' are installed:")
        print("pip install imageio imageio-ffmpeg")


# ========= Main =========
def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    try:
        import imageio  # noqa: F401
        print("✅ imageio is installed")
    except ImportError:
        print("❌ Missing imageio package, please run: pip install imageio imageio-ffmpeg")
        return
    
    try:
        print("Searching for the latest model checkpoint...")
        model_path = find_latest_model(".")
        
        print("Loading model...")
        unet, betas, alphas, alphas_cumprod, timesteps = load_model_and_params(model_path, device)
        print("Model loaded successfully!")
        
        while True:
            try:
                class_label = input("\nEnter the digit to generate (0-9), or 'q' to quit: ")
                if class_label.lower() == 'q':
                    print("Goodbye!")
                    break
                
                class_label = int(class_label)
                if class_label < 0 or class_label > 9:
                    print("Please enter a number between 0 and 9!")
                    continue
                
                print(f"\nStarting denoising process for digit {class_label}...")
                print("⚠️  Note: Video generation involves 1000 frames; this may take a while...")
                
                final_image, save_dir = sample_with_visualization(
                    unet, device, class_label, betas, alphas, alphas_cumprod, timesteps
                )
                
                print(f"\n🎉 Done! Denoising process for digit {class_label} saved to: {save_dir}")
                print(f"   🖼️  Grid image: {save_dir}/denoising_grid.png")
                print(f"   🎬 Video file: {save_dir}/denoising_process_class_{class_label}.mp4")
                
            except ValueError:
                print("Please enter a valid number!")
            except KeyboardInterrupt:
                print("\nProgram interrupted. Goodbye!")
                break
            except Exception as e:
                print(f"An error occurred: {e}")
    
    except FileNotFoundError as e:
        print(str(e))
        print("Make sure you have:")
        print("1. Trained the model and have unet_diffusion_epoch_XXX.pth or unet_diffusion_final.pth.")
        print("2. The script is running in the same directory as the model file.")
    except Exception as e:
        print(f"Error while loading model: {e}")


if __name__ == "__main__":
    main()
