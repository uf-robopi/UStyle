import cv2
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.transforms import ToPILImage
from torchvision import transforms
from PIL import Image
import os
import random
import numpy as np
from tqdm import tqdm
from utils.photo_gif import GIFSmoothing
from utils.waterbody import estimate_waterbody
import torchvision.transforms.functional as TF
from model.model import UStyleEncoder, UStyleDecoder

# Set device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Get script directory
base_dir = os.path.dirname(os.path.abspath(__file__))

def set_random_seed(seed=45):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

set_random_seed(45)  

class TransformerAttentionFusion(nn.Module):
    def __init__(
        self,
        channels,
        down_factor=2,     
        nheads=4,         
        dim_feedforward=512,  
        dropout=0.05,
        activation='relu',
        num_layers=3,
        use_softmax=True  
    ):
        """
        Args:
            channels (int): Number of channels in each input tensor (C).
                            We'll concatenate content & style => total = 2*C.
            down_factor (int): Factor by which to downsample the spatial dims for attention.
            nheads (int): Number of attention heads in the Transformer.
            dim_feedforward (int): Hidden size of the feed-forward layer in each Transformer block.
            dropout (float): Dropout rate in the Transformer.
            activation (str): Activation in the Transformer feed-forward ("relu", "gelu", etc.).
            num_layers (int): Number of Transformer encoder layers to stack.
            use_softmax (bool): Use softmax instead of sigmoid for generating attention maps.
        """
        super(TransformerAttentionFusion, self).__init__()

        self.down_factor = down_factor
        self.use_softmax = use_softmax

        # 1) Downsampling module to reduce spatial resolution
        self.downsample = nn.Conv2d(
            in_channels=2 * channels,
            out_channels=2 * channels,
            kernel_size=down_factor,  
            stride=down_factor,      
            padding=0,
            bias=False
        )

        # 2) Transformer encoder layers 
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=2 * channels,
            nhead=nheads,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            activation=activation
        )
        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer,
            num_layers=num_layers
        )

        # 3) Linear projection to produce an attention map
        self.attention_projection = nn.Linear(2 * channels, 2 * channels)

        # 4) Upsample back to the original spatial size
        self.upsample = nn.Upsample(scale_factor=down_factor, mode='bilinear', align_corners=False)

        # 5) Learnable scaling parameter to modulate the attention map's contribution.
        self.lambda_param = nn.Parameter(torch.tensor(0.5))

    def forward(self, content, style):
        """
        Args:
            content (Tensor): shape (B, C, H, W)
            style   (Tensor): shape (B, C, H, W)
        Returns:
            fused (Tensor): shape (B, 2C, H, W)
        """
        # -- Step 1: Concatenate content & style --> (B, 2C, H, W)
        combined = torch.cat([content, style], dim=1)
        B, _, H, W = combined.shape

        # -- Step 2: Downsample to reduce memory usage in Transformer
        combined_down = self.downsample(combined)
        B2, C2, H2, W2 = combined_down.shape

        # -- Step 3: Flatten & permute for Transformer (shape: (N, B, 2C))
        combined_seq = combined_down.view(B2, C2, H2 * W2).permute(2, 0, 1)

        # -- Step 4: Process through Transformer encoder
        transformed_seq = self.transformer_encoder(combined_seq)

        # -- Step 5: Project to attention map
        N2, B_, E = transformed_seq.shape  # N2 = H2 * W2, E = 2C
        transformed_seq_2d = transformed_seq.reshape(N2 * B_, E)
        attn_map_2d = self.attention_projection(transformed_seq_2d)

        # Use softmax or sigmoid as the activation function for attention
        if self.use_softmax:
            attn_map_2d = torch.softmax(attn_map_2d, dim=-1)  
        else:
            attn_map_2d = torch.sigmoid(attn_map_2d)  

        # -- Step 6: Reshape back to spatial dimensions: (B, 2C, H2, W2)
        attn_map_seq = attn_map_2d.view(N2, B_, E).permute(1, 2, 0)
        attn_map_down = attn_map_seq.view(B2, E, H2, W2)

        # -- Step 7: Upsample attention map back to (B, 2C, H, W)
        attn_map = self.upsample(attn_map_down)

        # -- Step 8: Fuse by modulating the combined features with the scaled attention map
        fused = combined * (1 + self.lambda_param * attn_map)
        return fused

class UStyleEncDec(nn.Module):
    def __init__(self, encoder_weights, decoder_weights):
        super(UStyleEncDec, self).__init__()
        self.encoder = UStyleEncoder(pretrained=False)
        self.decoder = UStyleDecoder()
        
        self.attention_fusion1 = TransformerAttentionFusion(channels=1024)
        self.attention_fusion2 = TransformerAttentionFusion(channels=512)
        self.attention_fusion3 = TransformerAttentionFusion(channels=256)
        self.attention_fusion4 = TransformerAttentionFusion(channels=64)

        # Load pre-trained weights correctly
        checkpoint = torch.load(encoder_weights, map_location=device)
        self.encoder.load_state_dict(checkpoint["encoder"])
        self.decoder.load_state_dict(checkpoint["decoder"])

    def forward(self, x):
        bottleneck, skips = self.encoder(x)
        output = self.decoder(bottleneck, skips)
        return output

# Function to find depth map file with any valid extension
def find_matching_file(base_name, directory, valid_extensions=('.jpg', '.jpeg', '.png', '.bmp', '.tiff')):
    """Search for a file with a given base name and any valid image extension."""
    
    for ext in valid_extensions:
        candidate_path = os.path.join(directory, base_name + ext)
        if os.path.exists(candidate_path):
            return candidate_path
    return None  # No matching file found

# Function to load and preprocess an image
def load_image(image_path, image_size=(480, 640)):
    transform = transforms.Compose([
        transforms.Resize(image_size),
        transforms.ToTensor(),
    ])
    image = Image.open(image_path).convert("RGB")
    return transform(image).unsqueeze(0).to(device)

# Function to save an image
def save_image(img, filename):
    if isinstance(img, torch.Tensor):
        image = ToPILImage()(img.squeeze(0).cpu())
    elif isinstance(img, Image.Image):
        image = img
    else:
        raise TypeError("save_image expects a torch.Tensor or PIL.Image.Image")
    image.save(filename)
    
#based on photowct2 github repo
def inv_sqrt_cov(cov, inverse=False):
    s, u = torch.linalg.eigh(cov + torch.eye(cov.shape[-1], device=device))
    n_s = torch.sum(s > 1e-5).item()
    s = torch.sqrt(s[:, :n_s])
    if inverse:
        s = 1 / s
    d = torch.diag_embed(s)
    u = u[:, :, :n_s]
    return torch.matmul(u, torch.matmul(d, u.transpose(-2, -1)))

def otsu_threshold(depth_map):
    """
    Computes dynamic depth threshold using Otsu's method.
    Ensures the input depth map is correctly formatted as a single-channel grayscale image.
    """
    depth_np = depth_map.squeeze().cpu().numpy() 

    if depth_np.ndim > 2:
        depth_np = np.mean(depth_np, axis=0)  

    depth_np = (depth_np * 255).astype(np.uint8) 

    # Apply Otsu’s threshold
    _, threshold = cv2.threshold(depth_np, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    threshold_value = float(np.mean(threshold)) / 255.0  

    return threshold_value  
    
def compute_adaptive_k(depth_map, mode='v3'):
    """
    Computes an adaptive scaling factor (-k) for depth-aware fusion.
    Ensures a smooth and natural transition in blending.
    
    Args:
        depth_map: Tensor (B, 1, H, W) - Normalized depth map.
        mode: str, one of ['v1', 'v2', 'v3'] - Chooses which adaptive k formula to use.
    
    Returns:
        adaptive_k: Tensor (B, 1, H, W) - Adaptive scaling factor.
    """
    if mode == 'v3':
        # --- Version 3: Uses a 5x5 kernel and tanh on normalized std ---
        depth_map_smooth = F.avg_pool2d(depth_map, kernel_size=5, stride=1, padding=1)
        depth_mean = torch.mean(depth_map_smooth, dim=[2, 3], keepdim=True)
        depth_std = torch.std(depth_map_smooth, dim=[2, 3], keepdim=True) + 1e-6
        normalized_std = depth_std / (depth_mean + 1e-3)
        contrast_factor = torch.tanh(2.8 * normalized_std)
        adaptive_k = 1.2 + 3.5 * contrast_factor
        return torch.clamp(adaptive_k, min=1.0, max=6.0)
    
    elif mode in ['v1', 'v2']:
        # --- Versions v1 and v2: Use a 3x3 kernel and log-normalized contrast ---
        depth_map_smooth = F.avg_pool2d(depth_map, kernel_size=3, stride=1, padding=1)
        depth_mean = torch.mean(depth_map_smooth, dim=[2, 3], keepdim=True)
        depth_std = torch.std(depth_map_smooth, dim=[2, 3], keepdim=True) + 1e-6
        log_contrast_raw = torch.log1p(depth_std / (depth_mean + 1e-6))
        log_contrast = torch.clamp(log_contrast_raw, min=0.01, max=1.0)
        multiplier = 2.5 if mode == 'v1' else 3.5
        adaptive_k = 1.2 + multiplier * log_contrast
        adaptive_k = torch.clamp(adaptive_k, min=1.0, max=5.0)
        # For v1, apply an extra 3x3 smoothing step
        if mode == 'v1':
            adaptive_k = F.avg_pool2d(adaptive_k, kernel_size=3, stride=1, padding=1)
            #adaptive_k = log_contrast_raw
        return adaptive_k
    
    else:
        raise ValueError("Unknown mode for compute_adaptive_k. Choose from 'v1', 'v2', or 'v3'.")  

def stylize_wct_core_depth(c_feat, s_feat, depth_map, style_strength=1.0, use_depth=True):
    """
    Depth-aware Whitening and Coloring Transform (WCT) with optional depth blending
    Args:
        c_feat: Content feature (B, C, H, W)
        s_feat: Style feature (B, C, H, W)
        depth_map: Depth map for content image (B, 1, H, W)
        opt: Transformation type ('wct')
        style_strength: Style blending factor
        use_depth: Boolean flag to apply depth-based blending if True.
    Returns:
        Stylized feature (B, C, H, W)
    """
    n_batch, n_channel, cont_h, cont_w = c_feat.shape

    # Compute Mean
    m_c = torch.mean(c_feat, dim=[2, 3], keepdim=True)
    m_s = torch.mean(s_feat, dim=[2, 3], keepdim=True)

    # Center the feature maps
    c_feat_centered = c_feat - m_c
    s_feat_centered = s_feat - m_s

    # Reshape for covariance computation (B, C, H*W)
    c_feat_flattened = c_feat_centered.view(n_batch, n_channel, -1)
    s_feat_flattened = s_feat_centered.view(n_batch, n_channel, -1)

    # Compute Covariance Matrices Efficiently
    c_cov = torch.bmm(c_feat_flattened, c_feat_flattened.transpose(1, 2)) / c_feat_flattened.shape[-1]
    s_cov = torch.bmm(s_feat_flattened, s_feat_flattened.transpose(1, 2)) / s_feat_flattened.shape[-1]

    # Compute Whitening Transform
    inv_sqrt_c_cov = inv_sqrt_cov(c_cov, inverse=True)

    # Compute Coloring Transform
    transform_matrix = torch.bmm(inv_sqrt_cov(s_cov), inv_sqrt_c_cov)

    # Apply Transformation
    feat_flattened = torch.bmm(transform_matrix, c_feat_flattened) + m_s.view(n_batch, n_channel, 1)
    feat = feat_flattened.view(n_batch, n_channel, cont_h, cont_w)
    feat = feat.clamp_(0.0, 1.0)

    if use_depth:
        # Resize `depth_map` to match `c_feat` spatial resolution
        depth_map = F.interpolate(depth_map, size=(cont_h, cont_w), mode='bilinear', align_corners=False)

        # Expand `depth_map` to match `c_feat`'s channel dimension
        depth_map = depth_map.expand(-1, n_channel, -1, -1)

        # Normalize Depth Map to range [0,1]
        depth_map = torch.clamp(depth_map, 0.0, 1.0)

        # Ensure `depth_map` is single-channel before Otsu's thresholding
        depth_map_single = depth_map[:, 0, :, :].unsqueeze(1) 

        depth_threshold_value = otsu_threshold(depth_map_single) 
        depth_threshold = torch.tensor(depth_threshold_value, dtype=torch.float32, device=depth_map.device)
        depth_threshold = depth_threshold.view(1, 1, 1, 1)  

        # Compute adaptive_k with the resized depth map
        adaptive_k = compute_adaptive_k(depth_map)
        print(adaptive_k)

        depth_weight = torch.sigmoid(-adaptive_k * (depth_map - depth_threshold))

        # Blend with original content features based on depth
        blended_feat = depth_weight * feat + (1 - depth_weight) * c_feat
        final_feat = style_strength * blended_feat + (1 - style_strength) * c_feat
    else:
        # If depth blending is not applied, blend the transformed features with content
        final_feat = style_strength * feat + (1 - style_strength) * c_feat

    return final_feat   
    
def multi_scale_stylize_depth(content_image, style_image, depth_map, scales=[1.0, 0.75, 0.5, 0.25], style_strength=1.0, use_depth=True):
    stylized_images = []
    for scale in scales:
        content_scaled = F.interpolate(content_image, scale_factor=scale, mode='bilinear', align_corners=False)
        style_scaled = F.interpolate(style_image, scale_factor=scale, mode='bilinear', align_corners=False)
        depth_scaled = F.interpolate(depth_map, scale_factor=scale, mode='bilinear', align_corners=False)

        stylized = stylize_wct_core_depth(content_scaled, style_scaled, depth_scaled, style_strength=style_strength, use_depth=use_depth)
        
        stylized_images.append(F.interpolate(stylized, size=content_image.shape[2:], mode='bilinear', align_corners=False))
    
    return torch.mean(torch.stack(stylized_images), dim=0) 

# Main fusion function with content and style depth maps
def fuse_images(content_dir, style_images_dir, content_depths_dir, style_depths_dir, output_dir, 
                encoder_weights, decoder_weights, style_strength=1.0, 
                use_multi_scale=True, use_attention_fusion=False, use_depth=True):
    # Initialize models
    model = UStyleEncDec(encoder_weights, decoder_weights).to(device)
    model.eval()

    post_processing = GIFSmoothing(r=20, eps=(0.008 * 255) ** 2)

    # Load content and style images
    content_files = sorted([f for f in os.listdir(content_dir) if f.lower().endswith(('.jpg', '.png', '.jpeg', '.bmp', '.tiff'))])
    style_files = sorted([f for f in os.listdir(style_images_dir) if f.lower().endswith(('.jpg', '.png', '.jpeg', '.bmp', '.tiff'))])

    # NEW: Outer loop over each content image
    for content_file in tqdm(content_files, desc="Processing Content Images"):
        # Load content image
        content_image = load_image(os.path.join(content_dir, content_file))
        content_base, content_ext = os.path.splitext(content_file)  

        # Load corresponding content depth map
        content_depth_path = find_matching_file(content_base, content_depths_dir)
        if content_depth_path is None:
            raise ValueError(f"Content depth map not found for {content_file} in {content_depths_dir}")
        content_depth_map = cv2.imread(content_depth_path, cv2.IMREAD_GRAYSCALE)
        if content_depth_map is None:
            raise ValueError(f"Failed to load content depth map: {content_depth_path}")
        content_depth_map = torch.tensor(content_depth_map, dtype=torch.float32) / 255.0
        content_depth_map = content_depth_map.unsqueeze(0).unsqueeze(0).to(device)  # Shape (1, 1, H, W)

        # NEW: Inner loop over each style image for the current content image
        for style_file in style_files:
            style_image = load_image(os.path.join(style_images_dir, style_file))
            style_base, _ = os.path.splitext(style_file)  # NEW: Get style base name

            # Load corresponding style depth map
            style_depth_path = find_matching_file(style_base, style_depths_dir)
            if style_depth_path is None:
                raise ValueError(f"Style depth map not found for {style_file} in {style_depths_dir}")
            style_depth_map = cv2.imread(style_depth_path, cv2.IMREAD_GRAYSCALE)
            if style_depth_map is None:
                raise ValueError(f"Failed to load style depth map: {style_depth_path}")

            # Load style image with OpenCV and convert from BGR to RGB
            style_img_cv = cv2.imread(os.path.join(style_images_dir, style_file))
            if style_img_cv is None:
                raise ValueError(f"Style image not found: {os.path.join(style_images_dir, style_file)}")
            style_img_cv = cv2.cvtColor(style_img_cv, cv2.COLOR_BGR2RGB)

            # Compute the waterbody color using the external function
            waterbody_color = estimate_waterbody(style_img_cv, style_depth_map)
            waterbody_img = np.ones_like(style_img_cv, dtype=np.float32) * waterbody_color.reshape(1, 1, 3)
            waterbody_img_uint8 = np.clip(waterbody_img * 255.0, 0, 255).astype(np.uint8)
            waterbody_pil = Image.fromarray(waterbody_img_uint8)
            style_image = transforms.ToTensor()(waterbody_pil).unsqueeze(0).to(device)

            with torch.no_grad():
                # Extract features for content and style images
                c_bottleneck, c_skips = model.encoder(content_image)
                c1, c2, c3, c4 = c_skips

                s_bottleneck, s_skips = model.encoder(style_image)
                s1, s2, s3, s4 = s_skips

            if use_multi_scale:
                fused_bottleneck = multi_scale_stylize_depth(c_bottleneck, s_bottleneck, content_depth_map, style_strength=style_strength, use_depth=use_depth)
            else:
                fused_bottleneck = stylize_wct_core_depth(c_bottleneck, s_bottleneck, content_depth_map, style_strength=style_strength, use_depth=use_depth)

            x = model.decoder.up1(fused_bottleneck, output_size=c4.shape[2:])
            if use_attention_fusion:
                x = model.attention_fusion1(x, stylize_wct_core_depth(c4, s4, content_depth_map, style_strength=style_strength, use_depth=use_depth))
            else:
                x = torch.cat([x, stylize_wct_core_depth(c4, s4, content_depth_map, style_strength=style_strength, use_depth=use_depth)], dim=1)
            x = model.decoder.conv1(x)

            x = model.decoder.up2(x, output_size=c3.shape[2:])
            if use_attention_fusion:
                x = model.attention_fusion2(x, stylize_wct_core_depth(c3, s3, content_depth_map, style_strength=style_strength, use_depth=use_depth))
            else:
                x = torch.cat([x, stylize_wct_core_depth(c3, s3, content_depth_map, style_strength=style_strength, use_depth=use_depth)], dim=1)
            x = model.decoder.conv2(x)

            x = model.decoder.up3(x, output_size=c2.shape[2:])
            if use_attention_fusion:
                x = model.attention_fusion3(x, stylize_wct_core_depth(c2, s2, content_depth_map, style_strength=style_strength, use_depth=use_depth))
            else:
                x = torch.cat([x, stylize_wct_core_depth(c2, s2, content_depth_map, style_strength=style_strength, use_depth=use_depth)], dim=1)
            x = model.decoder.conv3(x)

            x = model.decoder.up4(x, output_size=c1.shape[2:])
            if use_attention_fusion:
                x = model.attention_fusion4(x, stylize_wct_core_depth(c1, s1, content_depth_map, style_strength=style_strength, use_depth=use_depth))
            else:
                x = torch.cat([x, stylize_wct_core_depth(c1, s1, content_depth_map, style_strength=style_strength, use_depth=use_depth)], dim=1)
            x = model.decoder.conv4(x)

            x = model.decoder.up5(x, output_size=content_image.shape[2:])
            x = model.decoder.conv5(x)
            x = post_processing.process(x.detach().cpu(), content_image.detach().cpu())

            # Construct output filename using both content and style base names
            output_name = f"{content_base}_stylized_with_{style_base}{content_ext}"
            output_path = os.path.join(output_dir, output_name)
            save_image(x, output_path)
            
            # Free GPU memory after processing each style image
            del style_image, style_depth_map, style_img_cv, waterbody_img, waterbody_pil, x
            torch.cuda.empty_cache()

    # Free memory from content image and depth map after finishing all styles for one content image
    del content_image, content_depth_map
    torch.cuda.empty_cache()
  
# Example usage
if __name__ == "__main__":
    # Define absolute paths for input and output directories
    content_dir = os.path.join(base_dir, "inputs/all/content/images")
    content_depths_dir = os.path.join(base_dir, "inputs/all/content/depths")  
    style_images_dir = os.path.join(base_dir, "inputs/all/style/images")
    style_depths_dir = os.path.join(base_dir, "inputs/all/style/depths")  
    output_dir = os.path.join(base_dir, "outputs_all")

    print("Content directory:", content_dir)
    print("Content Depth directory:", content_depths_dir)  
    print("Style directory:", style_images_dir)
    print("Style Depth directory:", style_depths_dir)  
    print("Output directory:", output_dir)

    # Create output directory
    os.makedirs(output_dir, exist_ok=True)

    # Paths to pre-trained encoder and decoder weights
    encoder_weights = "checkpoints/trained_model.pth"
    decoder_weights = "checkpoints/trained_model.pth"
    
    # Run fusion with optional enhancements
    fuse_images(
        content_dir, style_images_dir, content_depths_dir, style_depths_dir, output_dir, 
        encoder_weights, decoder_weights,
        style_strength=1.0,  # Control the strength of style transfer
        use_multi_scale=True,  # Enable multi-scale stylization
        use_attention_fusion=False, # Enable/disable attention fusion
        use_depth=True
    )
