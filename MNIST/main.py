import torch
import torchvision.transforms as transforms
from torchvision.datasets import MNIST
from torch.utils.data import DataLoader
from unet import UNet
import torch.nn.functional as F
import matplotlib.pyplot as plt
import torchvision.utils as vutils
import os
from tqdm import tqdm

# ========= Dataset: MNIST 1x28x28 =========
transform = transforms.Compose([
    transforms.ToTensor(),                    
    transforms.Normalize((0.5,), (0.5,))      
])
dataset = MNIST(root='./data', train=True, transform=transform, download=True)
dataloader = DataLoader(dataset, batch_size=64, shuffle=True, num_workers=0, pin_memory=True)

# ========= diffusion schedule =========
def linear_beta_schedule(timesteps, start=0.0001, end=0.02):
    return torch.linspace(start, end, timesteps)

timesteps = 1000
betas = linear_beta_schedule(timesteps)
alphas = 1 - betas
alphas_cumprod = torch.cumprod(alphas, dim=0)

# ========= Training =========
def train_step(unet, x0, labels):
    device = x0.device
    t = torch.randint(0, timesteps, (x0.shape[0],), device=device, dtype=torch.long)
    noise = torch.randn_like(x0)
    sqrt_alphas_cumprod_t = torch.sqrt(alphas_cumprod[t]).to(device)
    sqrt_one_minus_alphas_cumprod_t = torch.sqrt(1 - alphas_cumprod[t]).to(device)
    
    x_t = sqrt_alphas_cumprod_t.view(-1,1,1,1) * x0 + \
          sqrt_one_minus_alphas_cumprod_t.view(-1,1,1,1) * noise
    
    predicted_noise = unet(x_t, time_step=t, class_label=labels)
    loss = F.mse_loss(predicted_noise, noise)
    return loss

# ========= sample and visualization =========
@torch.no_grad()
def sample_images(unet, device, num_samples=16, class_labels=None):
    unet.eval()
    if class_labels is None:
        class_labels = torch.randint(0, 10, (num_samples,), device=device)
    x = torch.randn(num_samples, 1, 28, 28, device=device)
    for t in reversed(range(timesteps)):
        t_tensor = torch.full((num_samples,), t, device=device, dtype=torch.long)
        predicted_noise = unet(x, time_step=t_tensor, class_label=class_labels)
        alpha_t = alphas[t].to(device)
        alpha_cumprod_t = alphas_cumprod[t].to(device)
        beta_t = betas[t].to(device)
        alpha_cumprod_prev = alphas_cumprod[t-1].to(device) if t > 0 else torch.tensor(1.0, device=device)
        
        # Predict x0 (optional for constraint)
        pred_x0 = (x - torch.sqrt(1 - alpha_cumprod_t) * predicted_noise) / torch.sqrt(alpha_cumprod_t)
        pred_x0 = torch.clamp(pred_x0, -1, 1)
        
        if t > 0:
            noise = torch.randn_like(x)
            # Simplified DDPM step (maintaining original style)
            x = torch.sqrt(alpha_t) * (x - beta_t / torch.sqrt(1 - alpha_cumprod_t) * predicted_noise) + torch.sqrt(beta_t) * noise
        else:
            x = pred_x0
    unet.train()
    return x

from torchvision.utils import save_image

def visualize_samples(unet, device, epoch, save_dir='./samples'):
    os.makedirs(save_dir, exist_ok=True)
    samples = sample_images(unet, device, num_samples=16)
    samples = (samples + 1) / 2
    samples = torch.clamp(samples, 0, 1)
    out_path = f'{save_dir}/epoch_{epoch:03d}.png'
    save_image(samples, out_path, nrow=4, padding=2)
    print(f"Samples saved to {out_path}")

# ========= Main training loop =========
if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    unet = UNet(in_channels=1, out_channels=1).to(device)
    optimizer = torch.optim.Adam(unet.parameters(), lr=1e-3)
    epochs = 500 
    
    # Move scheduler to device
    betas = betas.to(device)
    alphas = alphas.to(device)
    alphas_cumprod = alphas_cumprod.to(device)
    
    for epoch in range(epochs):
        unet.train()
        pbar = tqdm(dataloader, desc=f'Epoch {epoch+1}/{epochs}', leave=True)
        running_loss = 0.0
        
        for batch_idx, (data, labels) in enumerate(pbar):
            data, labels = data.to(device, non_blocking=True), labels.to(device, non_blocking=True)
            loss = train_step(unet, data, labels)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
            avg_loss = running_loss / (batch_idx + 1)
            pbar.set_postfix({'Loss': f'{loss.item():.4f}', 'Avg Loss': f'{avg_loss:.4f}'})
        
        # Visualize results every 10 epochs
        if (epoch + 1) % 10 == 0:
            print(f"\nGenerating samples for epoch {epoch+1}...")
            visualize_samples(unet, device, epoch+1)
        
        # Save model every 50 epochs
        if (epoch + 1) % 1 == 0:
            torch.save(unet.state_dict(), f"unet_diffusion_epoch_{epoch+1}.pth")
            print(f"Model saved to unet_diffusion_epoch_{epoch+1}.pth")
    
    torch.save(unet.state_dict(), "unet_diffusion_final.pth")
    print("Final model saved to unet_diffusion_final.pth")