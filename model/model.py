import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import resnet50

        
class UStyleEncoder(nn.Module):
    def __init__(self, pretrained=True):
        """
        UStyleEncoder initializes a ResNet50-based encoder that extracts hierarchical
        feature maps at different depths. These features can later be used as skip connections
        for the decoder.
        
        Args:
            pretrained (bool): If True, loads pretrained ResNet50 weights.
        """
        super(UStyleEncoder, self).__init__()
        
        # Load Pretrained ResNet50
        resnet = resnet50(pretrained=pretrained)

        # Extract layers for hierarchical feature maps
        self.stage0 = nn.Sequential(  # Low-level details (edges, textures, colors)
            resnet.conv1,
            resnet.bn1,
            resnet.relu,
        )

        self.stage1 = nn.Sequential(  # Mid-level details (shapes, patterns)
            resnet.maxpool,  # Apply maxpool here instead of Stage 0
            *list(resnet.layer1)  
        )

        self.stage2 = nn.Sequential(*list(resnet.layer2))  # Higher-level features (object parts, abstract textures)
        self.stage3 = nn.Sequential(*list(resnet.layer3))  # Bottleneck features (global style & content representation)
        self.bottleneck = nn.Sequential(*list(resnet.layer4))  # Deepest bottleneck layer

    def forward(self, x):
        """
        Forward pass through the encoder.
        
        Args:
            x (torch.Tensor): Input image tensor of shape (N, 3, H, W).
            
        Returns:
            x (torch.Tensor): Output from the bottleneck layer.
            skip_connections (list): A list of intermediate feature maps for skip connections.
        """
        skip_connections = []

        x = self.stage0(x)
        skip_connections.append(x)  # Low-level features (1, 64, H, W)

        x = self.stage1(x)
        skip_connections.append(x)  # Mid-level features (1, 256, H/2, W/2) 

        x = self.stage2(x)
        skip_connections.append(x)  # Higher-level features (1, 512, H/4, W/4)

        x = self.stage3(x)
        skip_connections.append(x)  # Bottleneck features (1, 1024, H/8, W/8)

        x = self.bottleneck(x)  # Final bottleneck (1, 2048, H/16, W/16)

        return x, skip_connections

    def encoder_block_0(self, x):
        x = self.stage0(x)
        return x, None  # No skip connection at this level

    def encoder_block_1(self, x):
        x = self.stage0(x)
        x = self.stage1(x)
        return x, self.stage0(x)  # Return feature map and skip connection

    def encoder_block_2(self, x):
        x = self.stage0(x)
        x = self.stage1(x)
        x = self.stage2(x)
        return x, self.stage1(x)  # Return feature map and skip connection

    def encoder_block_3(self, x):
        x = self.stage0(x)
        x = self.stage1(x)
        x = self.stage2(x)
        x = self.stage3(x)
        return x, self.stage2(x)  # Return feature map and skip connection
    
    def encoder_block_4(self, x):
        x = self.stage0(x)
        x = self.stage1(x)
        x = self.stage2(x)
        x_stage3 = self.stage3(x)  # Store stage3 separately
        x = self.bottleneck(x_stage3)  # Apply bottleneck on stage3 output
        return x, x_stage3  # Return both bottleneck output & stage3 for skip


    def get_encoder_blocks(self):
        return [
            self.stage0,
            self.stage1,
            self.stage2,
            self.stage3,
            self.bottleneck
        ]
        
        
class UStyleDecoder(nn.Module):
    def __init__(self):
        """
        UStyleDecoder reconstructs an image from the bottleneck features using a series
        of transposed convolutions. It also merges features from the encoder via skip connections.
        """
        super(UStyleDecoder, self).__init__()

        # Modified transposed convolutions with output_padding=1
        self.up1 = nn.ConvTranspose2d(2048, 1024, kernel_size=3, stride=2, padding=1, output_padding=1)
        self.conv1 = nn.Sequential(
            nn.Conv2d(2048, 1024, kernel_size=3, padding=1),
            nn.BatchNorm2d(1024),
            nn.ReLU(inplace=True)
        )

        self.up2 = nn.ConvTranspose2d(1024, 512, kernel_size=3, stride=2, padding=1, output_padding=1)
        self.conv2 = nn.Sequential(
            nn.Conv2d(1024, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True)
        )

        self.up3 = nn.ConvTranspose2d(512, 256, kernel_size=3, stride=2, padding=1, output_padding=1)
        self.conv3 = nn.Sequential(
            nn.Conv2d(512, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True)
        )

        self.up4 = nn.ConvTranspose2d(256, 64, kernel_size=3, stride=2, padding=1, output_padding=1)
        self.conv4 = nn.Sequential(
            nn.Conv2d(128, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True)
        )

        # Final upsampling layer (stride=2 for 64->128->256->512)
        self.up5 = nn.ConvTranspose2d(64, 32, kernel_size=3, stride=2, padding=1, output_padding=1)
        self.conv5 = nn.Sequential(
            nn.Conv2d(32, 3, kernel_size=3, padding=1),
            nn.Sigmoid()
        )
        
         # Initialize feats as an empty list
        self.feats = []

    def forward(self, x, skip_connections):
        """
        Forward pass through the decoder.
        
        Args:
            x (torch.Tensor): Bottleneck feature map from the encoder.
            skip_connections (list): List of feature maps from the encoder (from low to high level).
            
        Returns:
            x (torch.Tensor): Reconstructed image.
        """
        skips = list(reversed(skip_connections))

        # Ensure feats is correctly initialized and cleared
        if not hasattr(self, 'feats') or not isinstance(self.feats, list):
            self.feats = []
        self.feats.clear()  # Reset features list while keeping reference

        # Decoding steps
        x = self.up1(x, output_size=skips[0].shape[2:])
        x = torch.cat([x, skips[0]], 1)
        x = self.conv1(x)
        self.feats.append(x)  # Save features after conv1

        x = self.up2(x, output_size=skips[1].shape[2:])
        x = torch.cat([x, skips[1]], 1)
        x = self.conv2(x)
        self.feats.append(x)  # Save features after conv2

        x = self.up3(x, output_size=skips[2].shape[2:])
        x = torch.cat([x, skips[2]], 1)
        x = self.conv3(x)
        self.feats.append(x)  # Save features after conv3

        x = self.up4(x, output_size=skips[3].shape[2:])
        x = torch.cat([x, skips[3]], 1)
        x = self.conv4(x)
        self.feats.append(x)  # Save features after conv4

        x = self.up5(x, output_size=(skips[-1].shape[2] * 2, skips[-1].shape[3] * 2))
        x = self.conv5(x)

        return x

        
    def decoder_block_3(self, x, skip):
        x = self.up1(x, output_size=skip.shape[2:])
        x = torch.cat([x, skip], 1)
        x = self.conv1(x)
        return x

    def decoder_block_2(self, x, skip):
        x = self.up2(x, output_size=skip.shape[2:])
        x = torch.cat([x, skip], 1)
        x = self.conv2(x)
        return x

    def decoder_block_1(self, x, skip):
        x = self.up3(x, output_size=skip.shape[2:])
        x = torch.cat([x, skip], 1)
        x = self.conv3(x)
        return x

    def decoder_block_0(self, x, skip):
        x = self.up4(x, output_size=skip.shape[2:])
        x = torch.cat([x, skip], 1)
        x = self.conv4(x)
        return x

    def final_decoder(self, x):
        x = self.up5(x)
        x = self.conv5(x)
        return x


if __name__ == "__main__":
    # Encoder
    encoder = UStyleEncoder()
    sample_input = torch.randn(1, 3, 480, 640)
    bottleneck, skips = encoder(sample_input)

    # Decoder
    decoder = UStyleDecoder()
    reconstructed = decoder(bottleneck, skips)

    print("Reconstructed Image Shape:", reconstructed.shape)