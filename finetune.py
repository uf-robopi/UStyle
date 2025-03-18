"""
# Run command
#python3 finetune.py --checkpoint ./finetune_checkpoints/finetuned_model_final_finetuned.pth --load_checkpoint
"""
import argparse
import os
import torch
import torch.nn.functional as F
import torch.nn as nn
from torch.nn.parallel import DataParallel
from torch.utils.data import DataLoader
from model.model import UStyleEncoder, UStyleDecoder
from utils.data import ImageReconstructionDataset
from utils.loss import ReconstructionLoss, FeatureReconstructionLoss, PerceptualLoss, SSIMLoss, ColorConsistencyLoss, FrequencyDomainLoss, CLIPLoss

# Function to load a checkpoint
def load_checkpoint(encoder, decoder, optimizer, save_dir, block):
    checkpoint_path = os.path.join(save_dir, f'block_{block}_checkpoint_finetuned.pth')
    if os.path.exists(checkpoint_path):
        checkpoint = torch.load(checkpoint_path)
        if num_gpus > 1:
            encoder.module.load_state_dict(checkpoint['encoder'])
            decoder.module.load_state_dict(checkpoint['decoder'])
        else:
            encoder.load_state_dict(checkpoint['encoder'])
            decoder.load_state_dict(checkpoint['decoder'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        start_epoch = checkpoint['epoch'] + 1
        print(f"Resuming fine-tuning from epoch {start_epoch}")
        return start_epoch
    else:
        return 0

# Block-wise training function
def train_block(encoder, decoder, dataloader, device, block, epochs, save_dir, save_best=False):
    # Initialize original losses
    rec_loss = ReconstructionLoss()
    feat_loss = FeatureReconstructionLoss()
    percept_loss = PerceptualLoss()
    ssim_loss = SSIMLoss().to(device)
    colorconsist_loss = ColorConsistencyLoss().to(device) 
    fft_loss = FrequencyDomainLoss().to(device) 
    clip_loss_fn = CLIPLoss(device)

    # Loss weights
    loss_weights = {
        'rec': 10.0,
        'feat': 1.0 if block > 0 else 0,
        'percept': 1.0 if block > 0 else 0,
        'ssim': 1.0,
        'color': 0.001,
        'fft': 0.00001,
        'clip': 0.2,  
    }

    # Freeze/unfreeze logic
    for p in encoder.parameters():
        p.requires_grad = False
    
    # Handle both DataParallel and non-DataParallel cases
    if isinstance(encoder, DataParallel):
        encoder_blocks = encoder.module.get_encoder_blocks()  
    else:
        encoder_blocks = encoder.get_encoder_blocks()  
    
    for l in range(block + 1):
        for p in encoder_blocks[l].parameters():
            p.requires_grad = True
        
    if isinstance(decoder, DataParallel):
        decoder_blocks = [
            [decoder.module.up1, decoder.module.conv1],
            [decoder.module.up2, decoder.module.conv2],
            [decoder.module.up3, decoder.module.conv3],
            [decoder.module.up4, decoder.module.conv4],
            [decoder.module.up5, decoder.module.conv5],  
        ]
    else:
        decoder_blocks = [
            [decoder.up1, decoder.conv1],
            [decoder.up2, decoder.conv2],
            [decoder.up3, decoder.conv3],
            [decoder.up4, decoder.conv4],
            [decoder.up5, decoder.conv5],  
        ]

    for b in range(block + 1):
        for p in decoder_blocks[b]:
            p.requires_grad = True

    optimizer = torch.optim.Adam(
        list(encoder.parameters()) + list(decoder.parameters()),
        lr=1e-4
    )

    # If tracking best loss, initialize best_loss.
    best_loss = float('inf') if save_best else None

    # Load checkpoint if it exists
    start_epoch = load_checkpoint(encoder, decoder, optimizer, save_dir, block)

    # Initialize checkpoint_epoch with default values
    checkpoint_epoch = {
        'encoder': encoder.module.state_dict() if isinstance(encoder, DataParallel) else encoder.state_dict(),
        'decoder': decoder.module.state_dict() if isinstance(decoder, DataParallel) else decoder.state_dict(),
        'optimizer': optimizer.state_dict(),
        'epoch': start_epoch - 1 if start_epoch > 0 else 0,  
        'block': block,
        'loss': float('inf')  
    }
    
    
    for epoch in range(start_epoch, epochs):
        # Initialize variables to accumulate loss and count batches for averaging.
        epoch_loss = 0.0
        batch_count = 0
    
        for batch_idx, (inputs, _) in enumerate(dataloader):
            inputs = inputs.to(device)
            
            # Forward pass
            optimizer.zero_grad()
            bottleneck, all_skips = encoder(inputs)
            outputs = decoder(bottleneck, all_skips)
            
            # Compute individual losses
            loss_rec   = loss_weights['rec']   * rec_loss(inputs, outputs)
            loss_ssim  = loss_weights['ssim']  * ssim_loss(outputs, inputs)
            loss_color = loss_weights['color'] * colorconsist_loss(inputs, outputs)
            loss_fft   = loss_weights['fft']   * fft_loss(inputs, outputs)
            loss_clip  = loss_weights['clip']  * clip_loss_fn(inputs, outputs)
            
            # Sum base losses
            loss = loss_rec + loss_ssim + loss_color + loss_fft + loss_clip
            
            # Initialize default values for additional losses (for block == 0)
            loss_feat = 0.0
            loss_percept = 0.0
    
            if block > 0:
                enc_skip_idx = 3 - block
                if num_gpus > 1:
                    dec_feat_idx = block - 1  # Adjust index to match the feats list (0-based indexing)
                else:
                    dec_feat_idx = block
                                        
                if isinstance(decoder, DataParallel):
                    if hasattr(decoder.module, 'feats') and len(decoder.module.feats) > dec_feat_idx:
                        # Only use the correct indexed feature map
                        dec_feats_resized = decoder.module.feats[dec_feat_idx].to(device)
                
                        # Resize spatial dimensions
                        if all_skips[enc_skip_idx].shape[2:] != dec_feats_resized.shape[2:]:
                            dec_feats_resized = F.interpolate(
                                dec_feats_resized, 
                                size=all_skips[enc_skip_idx].shape[2:], 
                                mode='bilinear', align_corners=False
                            )
                
                        # Match channel count
                        if dec_feats_resized.shape[1] != all_skips[enc_skip_idx].shape[1]:
                            channel_mapper = nn.Conv2d(dec_feats_resized.shape[1], all_skips[enc_skip_idx].shape[1], kernel_size=1).to(device)
                            dec_feats_resized = channel_mapper(dec_feats_resized)
                        
                        # Ensure batch sizes match
                        if dec_feats_resized.shape[0] != all_skips[enc_skip_idx].shape[0]:
                            repeat_factor = all_skips[enc_skip_idx].shape[0] // dec_feats_resized.shape[0]
                            dec_feats_resized = dec_feats_resized.repeat(repeat_factor, 1, 1, 1)
                        
                        # Compute loss
                        loss_feat = loss_weights['feat'] * feat_loss(all_skips[enc_skip_idx], dec_feats_resized)
    
                    else:
                        loss_feat = torch.tensor(0.0, device=device) 
    
                else:
                    if hasattr(decoder, 'feats') and len(decoder.feats) > dec_feat_idx:
                        loss_feat = loss_weights['feat'] * feat_loss(all_skips[enc_skip_idx], decoder.feats[dec_feat_idx])
                    else:
                        loss_feat = torch.tensor(0.0, device=device) 
                
                with torch.no_grad():
                    recon_feats, _ = encoder(outputs)
                loss_percept = loss_weights['percept'] * percept_loss(bottleneck, recon_feats)
                
                loss += loss_feat + loss_percept
                
            # Backprop
            loss.backward()
            optimizer.step()
            
            # Accumulate loss and batch count for epoch averaging.
            epoch_loss += loss.item()
            batch_count += 1
    
            # Display individual loss values along with the total loss.
            if batch_idx % 10 == 0:
                print(f'Epoch [{epoch + 1}/{epochs}], Batch [{batch_idx}/{len(dataloader)}], '
                      f'Total Loss: {loss.item():.6f}, Rec: {loss_rec.item():.6f}, '
                      f'SSIM: {loss_ssim.item():.6f}, Color: {loss_color.item():.6f}, FFT: {loss_fft.item():.6f}, '
                      f'Feat: {loss_feat.item() if not isinstance(loss_feat, float) else loss_feat:.6f}, '
                      f'Percept: {loss_percept.item() if not isinstance(loss_percept, float) else loss_percept:.6f},'
                      f'Clip: {loss_clip.item():.6f} ')

    
        # Compute average epoch loss.
        avg_epoch_loss = epoch_loss / batch_count
        print(f"Epoch {epoch+1} average loss: {avg_epoch_loss:.4f}")
    
        # Update checkpoint_epoch with current state
        checkpoint_epoch = {
            'encoder': encoder.module.state_dict() if isinstance(encoder, DataParallel) else encoder.state_dict(),
            'decoder': decoder.module.state_dict() if isinstance(decoder, DataParallel) else decoder.state_dict(),
            'optimizer': optimizer.state_dict(),
            'epoch': epoch,
            'block': block,
            'loss': avg_epoch_loss
        }
        #torch.save(checkpoint_epoch, os.path.join(save_dir, f'block_{block}_checkpoint_epoch_{epoch+1}.pth'))
        
        # If tracking best model for final block, check and update best loss.
        if save_best and avg_epoch_loss < best_loss:
            best_loss = avg_epoch_loss
            torch.save(checkpoint_epoch, os.path.join(save_dir, 'final_model_best_finetuned.pth'))
            print(f"--> Best model updated at epoch {epoch+1} with average loss {avg_epoch_loss:.4f}")
    
    # After all epochs, save final checkpoint.
    torch.save(checkpoint_epoch, os.path.join(save_dir, f'block_{block}_checkpoint_finetuned.pth'))
    print(f"Final checkpoint for block {block} saved as 'block_{block}_checkpoint_finetuned.pth'.")

# -----------------------------------------------------------------------------
# Main fine-tuning code (using block-wise training strategy)
# -----------------------------------------------------------------------------
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--finetune_dataset', type=str, nargs='+',
                    default=[
                        './USOD10K/USOD10K_TR/RGB/',
                        './USOD10K/USOD10K_TE/RGB/'
                    ],
                    help="Path(s) to training dataset")
    parser.add_argument('--batch_size', type=int, default=4)
    parser.add_argument('--epochs', type=int, nargs=5, default=[10, 10, 10, 10, 10])
    parser.add_argument('--checkpoint', type=str, default='./finetune_checkpoints/finetuned_model_final_finetuned.pth',
                        help="Path to pre-trained checkpoint for fine-tuning")
    parser.add_argument('--load_checkpoint', action='store_true',
                    help="Whether to load a checkpoint for fine-tuning")
    parser.add_argument('--save_dir', type=str, default='./finetune_checkpoints')
    args = parser.parse_args()
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    num_gpus = torch.cuda.device_count()
    
    # Initialize models
    encoder = UStyleEncoder()
    decoder = UStyleDecoder()
    
    if num_gpus > 1:
        print(f"Using {num_gpus} GPUs!")
        encoder = DataParallel(encoder)
        decoder = DataParallel(decoder)

    encoder = encoder.to(device)
    decoder = decoder.to(device)
    
    if args.load_checkpoint:
        checkpoint = torch.load(args.checkpoint, map_location=device)
        encoder_state = checkpoint.get('encoder', checkpoint)
        decoder_state = checkpoint.get('decoder', checkpoint)
        if num_gpus > 1:
            encoder.module.load_state_dict(encoder_state)
            decoder.module.load_state_dict(decoder_state)
        else:
            encoder.load_state_dict(encoder_state)
            decoder.load_state_dict(decoder_state)
        print(f"Loaded checkpoint from {args.checkpoint}")
    else:
        print("Skipping checkpoint loading.")
    
    # Set models to training mode (for fine-tuning)
    encoder.train()
    decoder.train()
    
    # Initialize the fine-tuning dataset and dataloader
    dataset = ImageReconstructionDataset(args.finetune_dataset)
    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size *max(1, num_gpus),
        shuffle=True,
        num_workers=0,
        pin_memory=True,
        persistent_workers=False
    )
    
    # Create save directory for fine-tuning checkpoints if it doesn't exist
    os.makedirs(args.save_dir, exist_ok=True)
    
    # Fine-tune the model in a block-wise manner
    for block in range(5):
        print(f"Fine-tuning block {block}")
        # Only the final block (block 4) will track the best model.
        save_best_flag = (block == 4)
        train_block(encoder, decoder, dataloader, device, block, args.epochs[block], args.save_dir, save_best=save_best_flag)
    
    # Save an overall final state after fine-tuning
    final_state = {
        'encoder': encoder.module.state_dict() if num_gpus > 1 else encoder.state_dict(),
        'decoder': decoder.module.state_dict() if num_gpus > 1 else decoder.state_dict()
    }
    torch.save(final_state, os.path.join(args.save_dir, 'finetuned_model_final_finetuned.pth'))
    print("Final fine-tuned state saved as 'finetuned_model_final_finetuned.pth'.")
    
