import torch
import clip
import kornia  
import torch.nn as nn
import torch.nn.functional as F
from pytorch_msssim import SSIM  

class ReconstructionLoss(nn.Module):
    """
    Reconstruction Loss (MSE Loss) to measure pixel-wise differences between input and reconstructed images.
    """
    def __init__(self):
        super().__init__()
        self.mse = nn.MSELoss() 

    def forward(self, inputs, reconstructions):
        """
        Compute the reconstruction loss between inputs and reconstructions.

        Args:
            inputs (torch.Tensor): Original input images.
            reconstructions (torch.Tensor): Reconstructed images.

        Returns:
            torch.Tensor: Reconstruction loss value.
        """
        return self.mse(inputs, reconstructions)


class FeatureReconstructionLoss(nn.Module):
    """
    Feature Reconstruction Loss to measure differences between encoder and decoder feature maps.
    """
    def __init__(self):
        super().__init__()
        self.mse = nn.MSELoss()  

    def forward(self, enc_feats, dec_feats):
        """
        Compute the feature reconstruction loss between encoder and decoder features.

        Args:
            enc_feats (torch.Tensor): Feature maps from the encoder.
            dec_feats (torch.Tensor): Feature maps from the decoder.

        Returns:
            torch.Tensor: Feature reconstruction loss value.
        """
        return self.mse(enc_feats, dec_feats)


class PerceptualLoss(nn.Module):
    """
    Perceptual Loss to measure differences between feature representations of original and reconstructed images.
    """
    def __init__(self):
        super().__init__()
        self.mse = nn.MSELoss() 

    def forward(self, orig_feats, recon_feats):
        """
        Compute the perceptual loss between original and reconstructed feature maps.

        Args:
            orig_feats (torch.Tensor): Feature maps from the original image.
            recon_feats (torch.Tensor): Feature maps from the reconstructed image.

        Returns:
            torch.Tensor: Perceptual loss value.
        """
        return self.mse(orig_feats, recon_feats)


class SSIMLoss(nn.Module):
    """
    SSIM (Structural Similarity Index Measure) Loss to measure structural similarity between images.
    """
    def __init__(self):
        super().__init__()
        self.ssim = SSIM(data_range=1.0, size_average=True)  # SSIM with data range [0, 1]

    def forward(self, recon, orig):
        """
        Compute the SSIM loss between reconstructed and original images.

        Args:
            recon (torch.Tensor): Reconstructed images.
            orig (torch.Tensor): Original images.

        Returns:
            torch.Tensor: SSIM loss value (1 - SSIM).
        """
        return 1 - self.ssim(recon, orig)  # SSIM loss is 1 - SSIM value
        

class ColorConsistencyLoss(nn.Module):
    """
    Color Consistency Loss to ensure that the color distribution of the reconstructed image matches the input image.
    """
    def __init__(self):
        super().__init__()
        self.mse = nn.MSELoss()

    def forward(self, inputs, reconstructions):
        """
        Compute the color consistency loss between inputs and reconstructions.

        Args:
            inputs (torch.Tensor): Original input images.
            reconstructions (torch.Tensor): Reconstructed images.

        Returns:
            torch.Tensor: Color consistency loss value.
        """
        # Convert images to LAB color space for better color consistency
        inputs_lab = self.rgb_to_lab(inputs)
        reconstructions_lab = self.rgb_to_lab(reconstructions)
        return self.mse(inputs_lab, reconstructions_lab)

    def rgb_to_lab(self, image):
        """
        Convert RGB image to LAB color space using Kornia.

        Args:
            image (torch.Tensor): Input image in RGB format.

        Returns:
            torch.Tensor: Image in LAB color space.
        """
        return kornia.color.rgb_to_lab(image)

    
class FrequencyDomainLoss(nn.Module):
    """
    Frequency Domain Loss to preserve the frequency characteristics of the input image.
    """
    def __init__(self):
        super().__init__()
        self.mse = nn.MSELoss()

    def forward(self, inputs, reconstructions):
        """
        Compute the frequency domain loss between inputs and reconstructions.

        Args:
            inputs (torch.Tensor): Original input images.
            reconstructions (torch.Tensor): Reconstructed images.

        Returns:
            torch.Tensor: Frequency domain loss value.
        """
        # Compute the Fourier Transform of the input and reconstructed images
        inputs_fft = torch.fft.fft2(inputs)
        reconstructions_fft = torch.fft.fft2(reconstructions)

        # Compute magnitude of the Fourier transforms
        inputs_mag = torch.abs(inputs_fft)
        reconstructions_mag = torch.abs(reconstructions_fft)

        return self.mse(inputs_mag, reconstructions_mag)


class CLIPLoss(nn.Module):
    """
    CLIP Loss: Uses OpenAI's CLIP model to compute a semantic loss between
    the original input image and the reconstructed image.
    """
    def __init__(self, device):
        super().__init__()
        # Load the CLIP model (e.g., ViT-B/32)
        self.clip_model, self.clip_preprocess = clip.load("ViT-B/32", device=device)
        self.device = device
        self.clip_model.eval()  # Ensure the CLIP model is in evaluation mode
 
    def forward(self, input_images, reconstructed_images):
        """
        Compute CLIP loss between original and reconstructed images.
 
        Args:
            input_images (torch.Tensor): Original images (B, C, H, W)
            reconstructed_images (torch.Tensor): Reconstructed images (B, C, H, W)
 
        Returns:
            torch.Tensor: CLIP-based image similarity loss.
        """
        # Resize images to 224x224 (CLIP expects 224x224 input)
        input_images_resized = F.interpolate(input_images, size=(224, 224), mode='bilinear', align_corners=False)
        reconstructed_images_resized = F.interpolate(reconstructed_images, size=(224, 224), mode='bilinear', align_corners=False)
 
        # Extract image embeddings using CLIP’s image encoder
        with torch.no_grad():  # No gradients needed for CLIP model
            input_features = self.clip_model.encode_image(input_images_resized)  # Shape: [B, 512]
            output_features = self.clip_model.encode_image(reconstructed_images_resized)  # Shape: [B, 512]
 
        # Normalize features (ensures cosine similarity behaves correctly)
        input_features = input_features / input_features.norm(dim=-1, keepdim=True)
        output_features = output_features / output_features.norm(dim=-1, keepdim=True)
 
        # Compute cosine similarity
        similarity = torch.cosine_similarity(input_features, output_features, dim=-1)
 
        # Loss = 1 - similarity (higher similarity → lower loss)
        loss = 1 - similarity
        return loss.mean()

        