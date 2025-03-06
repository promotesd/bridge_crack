# F:\大智若愚\桥梁裂缝数据集\bridge_crack\code\models\Unetmodel1.py

import torch
import torch.nn as nn
import torch.nn.functional as F

# Squeeze-and-Excitation Block using Convolutional Layers
class SEBlock(nn.Module):
    def __init__(self, in_channels, reduction=16):
        super(SEBlock, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.conv_reduce = nn.Conv2d(in_channels, in_channels // reduction, kernel_size=1)
        self.gelu = nn.GELU()
        self.conv_expand = nn.Conv2d(in_channels // reduction, in_channels, kernel_size=1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        y = self.avg_pool(x)  # Shape: (b, c, 1, 1)
        y = self.conv_reduce(y)
        y = self.gelu(y)
        y = self.conv_expand(y)
        y = self.sigmoid(y)
        return x * y  # Element-wise multiplication

# Attention Block
class AttentionBlock(nn.Module):
    def __init__(self, F_g, F_l, F_int):
        super(AttentionBlock, self).__init__()
        self.W_g = nn.Sequential(
            nn.Conv2d(F_g, F_int, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(F_int)
        )
        self.W_x = nn.Sequential(
            nn.Conv2d(F_l, F_int, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(F_int)
        )
        self.psi = nn.Sequential(
            nn.Conv2d(F_int, 1, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(1),
            nn.Sigmoid()
        )
        self.gelu = nn.GELU()

    def forward(self, g, x):
        g1 = self.W_g(g)
        x1 = self.W_x(x)
        psi = self.gelu(g1 + x1)
        psi = self.psi(psi)
        return x * psi

# U-Net Model
class Unet(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(Unet, self).__init__()

        # Encoder (Downsampling path)
        self.enc1 = self.conv_block(in_channels, 64)
        self.enc2 = self.conv_block(64, 128)
        self.enc3 = self.conv_block(128, 256)
        self.enc4 = self.conv_block(256, 512)

        # Bottleneck
        self.bottleneck = self.conv_block(512, 1024)

        # Attention blocks
        self.att4 = AttentionBlock(F_g=512, F_l=512, F_int=256)
        self.att3 = AttentionBlock(F_g=256, F_l=256, F_int=128)
        self.att2 = AttentionBlock(F_g=128, F_l=128, F_int=64)

        # Decoder (Upsampling path with skip connections and attention blocks)
        self.upconv4 = nn.ConvTranspose2d(1024, 512, kernel_size=2, stride=2)
        self.dec4 = self.conv_block(1024, 512)
        self.upconv3 = nn.ConvTranspose2d(512, 256, kernel_size=2, stride=2)
        self.dec3 = self.conv_block(512, 256)
        self.upconv2 = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)
        self.dec2 = self.conv_block(256, 128)
        self.upconv1 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
        self.dec1 = self.conv_block(128, 64)

        # Final 1x1 Convolution to reduce to the number of output channels
        self.final_conv = nn.Conv2d(64, out_channels, kernel_size=1)

    def conv_block(self, in_channels, out_channels, use_se=True):
        layers = [
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.GELU(),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.GELU()
        ]
        if use_se:
            layers.append(SEBlock(out_channels))
        return nn.Sequential(*layers)

    def forward(self, x):
        # Encoder Path
        enc1 = self.enc1(x)
        enc2 = self.enc2(F.max_pool2d(enc1, kernel_size=2))
        enc3 = self.enc3(F.max_pool2d(enc2, kernel_size=2))
        enc4 = self.enc4(F.max_pool2d(enc3, kernel_size=2))

        # Bottleneck
        bottleneck = self.bottleneck(F.max_pool2d(enc4, kernel_size=2))

        # Decoder Path with attention blocks
        dec4 = self.upconv4(bottleneck)
        enc4 = self.att4(dec4, enc4)  # Apply attention
        dec4 = torch.cat((dec4, enc4), dim=1)  # Skip connection
        dec4 = self.dec4(dec4)

        dec3 = self.upconv3(dec4)
        enc3 = self.att3(dec3, enc3)  # Apply attention
        dec3 = torch.cat((dec3, enc3), dim=1)  # Skip connection
        dec3 = self.dec3(dec3)

        dec2 = self.upconv2(dec3)
        enc2 = self.att2(dec2, enc2)  # Apply attention
        dec2 = torch.cat((dec2, enc2), dim=1)  # Skip connection
        dec2 = self.dec2(dec2)

        dec1 = self.upconv1(dec2)
        dec1 = torch.cat((dec1, enc1), dim=1)  # Skip connection
        dec1 = self.dec1(dec1)

        # Final Convolution
        out = self.final_conv(dec1)
        return out

