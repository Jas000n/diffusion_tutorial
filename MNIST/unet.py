from torch import nn
import torch
import math

class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(out_channels)
        
        # Skip connection
        if stride != 1 or in_channels != out_channels:
            self.skip = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride),
                nn.BatchNorm2d(out_channels)
            )
        else:
            self.skip = nn.Identity()

    def forward(self, x):
        identity = self.skip(x)
        
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        
        out = self.conv2(out)
        out = self.bn2(out)
        
        out += identity
        out = self.relu(out)
        
        return out

class UNet(nn.Module):
    def __init__(self, in_channels=1, out_channels=1):
        super(UNet, self).__init__()
        
        # Encoder - MNIST: input (b, 1, 28, 28)
        self.encoder1 = nn.Sequential(
            nn.Conv2d(in_channels, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            ResidualBlock(128, 128),
            ResidualBlock(128, 128),
        )
        # Output: (b, 128, 28, 28)

        # Downsample to (b, 256, 14, 14)
        self.encoder2 = nn.Sequential(
            nn.MaxPool2d(2, 2),           # 28 -> 14
            ResidualBlock(128, 256),
            ResidualBlock(256, 256),
            ResidualBlock(256, 256),
        )
        # Output: (b, 256, 14, 14)
      
        # Neck: further downsample to (b, 512, 7, 7)
        self.neck = nn.Sequential(
            nn.MaxPool2d(2, 2),           # 14 -> 7
            ResidualBlock(256, 512),
            ResidualBlock(512, 512),
            ResidualBlock(512, 512),
            ResidualBlock(512, 512),
        )
        # Output: (b, 512, 7, 7)
        
        # Upsampling
        self.upsample1 = nn.Sequential(
            nn.ConvTranspose2d(512, 256, kernel_size=2, stride=2),  # 7 -> 14
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
        )
        
        self.upsample2 = nn.Sequential(
            nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2),  # 14 -> 28
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
        )
        
        # Decoder
        self.decoder1 = nn.Sequential(
            ResidualBlock(512, 256),  # concat: 256 (upsample1) + 256 (skip from encoder2)
            ResidualBlock(256, 256),
            ResidualBlock(256, 256),
        )
        
        self.decoder2 = nn.Sequential(
            ResidualBlock(256, 128),  # concat: 128 (upsample2) + 128 (skip from encoder1)
            ResidualBlock(128, 128),
            ResidualBlock(128, 128),
        )
        
        self.decoder3 = nn.Sequential(
            nn.Conv2d(128, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, out_channels, kernel_size=1),
        )
        
        # Class embedding: 10 classes (MNIST), output 49, reshaped to 7x7
        self.language_embedding = nn.Sequential(
            nn.Embedding(10, 128),
            nn.Linear(128, 128),
            nn.ReLU(inplace=True),
            nn.Linear(128, 49),   # 7*7
            nn.ReLU(inplace=True),
        )
        
        # Fuse neck features with time/class embeddings (all expanded to 512 channels)
        self.fuse = nn.Sequential(
            nn.Conv2d(512+512+512, 512, kernel_size=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            ResidualBlock(512, 512),
        )

    def get_sinusoidal_positional_embedding(self, timesteps, dim):
        half_dim = dim // 2
        emb = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, dtype=torch.float32, device=timesteps.device) * -emb)
        emb = timesteps[:, None].float() * emb[None, :]
        emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=1)
        if dim % 2 == 1:
            emb = torch.nn.functional.pad(emb, (0,1))
        return emb
    
    def forward(self, x, time_step=None, class_label=None):
        device = x.device
        
        # Encoder path
        x1 = self.encoder1(x)     # (b, 128, 28, 28)
        x2 = self.encoder2(x1)    # (b, 256, 14, 14)
        x_neck = self.neck(x2)    # (b, 512, 7, 7)
        
        # Time & class embeddings -> 7x7
        # Time embedding dimension is set to 49, reshaped into (7,7)
        timestep_embed = self.get_sinusoidal_positional_embedding(time_step, 49)  # (b, 49)
        timestep_embed = timestep_embed.reshape(-1, 7, 7).unsqueeze(1).expand(-1, 512, -1, -1).to(device)  # (b, 512, 7, 7)
        
        class_embed = self.language_embedding(class_label)  # (b, 49)
        class_embed = class_embed.reshape(-1, 7, 7).unsqueeze(1).expand(-1, 512, -1, -1).to(device)        # (b, 512, 7, 7)

        # Fuse embeddings
        x_neck = torch.cat([x_neck, timestep_embed, class_embed], dim=1)  # (b, 1536, 7, 7)
        x_neck = self.fuse(x_neck)  # (b, 512, 7, 7)
        
        # Decoder path with skip connections
        x_up1 = self.upsample1(x_neck)                       # (b, 256, 14, 14)
        x3 = self.decoder1(torch.cat([x_up1, x2], dim=1))    # (b, 256, 14, 14)
        
        x_up2 = self.upsample2(x3)                           # (b, 128, 28, 28)
        x4 = self.decoder2(torch.cat([x_up2, x1], dim=1))    # (b, 128, 28, 28)
        
        x4 = self.decoder3(x4)                               # (b, out_channels, 28, 28)
        return x4

if __name__ == "__main__":
    # MNIST: (b, 1, 28, 28)
    model = UNet(in_channels=1, out_channels=1)
    x = torch.randn(1, 1, 28, 28)
    out = model(x, time_step=torch.tensor([10]), class_label=torch.tensor([1]))
    print(f"Output shape: {out.shape}")
    print(f"Total parameters: {sum(p.numel() for p in model.parameters()):,}")
