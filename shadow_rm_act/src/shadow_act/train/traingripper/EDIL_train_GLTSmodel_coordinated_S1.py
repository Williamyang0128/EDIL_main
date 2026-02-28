import torch
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from EDIL_GLTSmodel_coordinated import CoordinatedMultiModalGripperNet, GripperDataset, get_data_transforms
import torch.nn as nn
import os
import argparse
from tqdm import tqdm
import numpy as np
import glob
import wandb

def train_model(args):
    # Initialize wandb
    wandb.init(project=args.wandb_project, entity=args.wandb_entity, name=args.run_name, config=args)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    print("Training a coordinated dual-arm model with separate weighted loss for each arm.")

    # Data Loading Section
    print("Loading and preparing paired data for both arms...")
    # Use glob to find all episode directories
    # Assumes data structure: data_dir/episode_*/{actions_left.csv, cam_left.mp4, ...}
    episode_dirs = sorted(glob.glob(os.path.join(args.data_dir, 'episode_*')))
    
    left_csv_files, left_video_files = [], []
    right_csv_files, right_video_files = [], []
    
    for episode_dir in episode_dirs:
        dir_name = os.path.basename(episode_dir)
        # Construct file paths assuming standard naming convention
        l_csv_path = os.path.join(episode_dir, f"{dir_name}_actions_left.csv")
        l_video_path = os.path.join(episode_dir, f"{dir_name}_cam_left.mp4")
        r_csv_path = os.path.join(episode_dir, f"{dir_name}_actions_right.csv")
        r_video_path = os.path.join(episode_dir, f"{dir_name}_cam_right.mp4")
        
        # Verify all required files exist for this episode
        if all(os.path.exists(p) for p in [l_csv_path, l_video_path, r_csv_path, r_video_path]):
            left_csv_files.append(l_csv_path)
            left_video_files.append(l_video_path)
            right_csv_files.append(r_csv_path)
            right_video_files.append(r_video_path)
            
    if not left_csv_files: 
        raise ValueError(f"No paired data found in {args.data_dir}. Please check your data directory structure.")
        
    print(f"Found {len(left_csv_files)} paired episodes for dual-arm training.")
    
    transform = get_data_transforms()
    full_dataset = GripperDataset(
        left_csv_files=left_csv_files, 
        left_video_files=left_video_files,
        right_csv_files=right_csv_files, 
        right_video_files=right_video_files,
        sequence_length=args.sequence_length, 
        transform=transform
    )
    
    # Split dataset into train and validation sets (80/20 split)
    train_size = int(0.8 * len(full_dataset))
    val_size = len(full_dataset) - train_size
    train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size])
    
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=4, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=4, pin_memory=True)
    
    # Initialize Model
    # coordination_influence controls the impact of the cross-attention mechanism
    model = CoordinatedMultiModalGripperNet(coordination_influence=args.coordination_influence).to(device)
    
    # Track model gradients and parameters with wandb
    wandb.watch(model, log='all', log_freq=100)
    
    criterion = nn.MSELoss(reduction='none')
    optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=args.learning_rate)
    
    print("Starting training...")
    best_val_loss = np.inf
    
    for epoch in range(args.num_epochs):
        # --- Training Loop ---
        model.train()
        running_loss = 0.0
        train_pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{args.num_epochs} [Train]")
        
        for img_l, seq_l, lbl_l, img_r, seq_r, lbl_r in train_pbar:
            img_l, seq_l, lbl_l = img_l.to(device), seq_l.to(device), lbl_l.to(device)
            img_r, seq_r, lbl_r = img_r.to(device), seq_r.to(device), lbl_r.to(device)
            
            optimizer.zero_grad()
            
            # Forward pass
            outputs_l, outputs_r = model(img_l, seq_l, img_r, seq_r)
            
            # Calculate separate losses
            unweighted_loss_l = criterion(outputs_l, lbl_l)
            unweighted_loss_r = criterion(outputs_r, lbl_r)

            # Create weight masks
            weight_l = torch.ones_like(lbl_l)
            weight_r = torch.ones_like(lbl_r)
            
            # Apply higher weight to positive actions (if configured)
            weight_l[lbl_l > 0] = args.positive_weight_left
            weight_r[lbl_r > 0] = args.positive_weight_right

            # Apply weights and mean reduction
            loss_l = (unweighted_loss_l * weight_l).mean()
            loss_r = (unweighted_loss_r * weight_r).mean()
            
            loss = loss_l + loss_r
            
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item() * img_l.size(0)
            train_pbar.set_postfix({'total_loss': loss.item(), 'loss_L': loss_l.item(), 'loss_R': loss_r.item()})
            
            # Log step metrics
            wandb.log({
                'train_total_loss_step': loss.item(),
                'train_loss_L_step': loss_l.item(),
                'train_loss_R_step': loss_r.item()
            })

        # --- Validation Loop ---
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            val_pbar = tqdm(val_loader, desc=f"Epoch {epoch+1}/{args.num_epochs} [Val]")
            for img_l, seq_l, lbl_l, img_r, seq_r, lbl_r in val_pbar:
                img_l, seq_l, lbl_l = img_l.to(device), seq_l.to(device), lbl_l.to(device)
                img_r, seq_r, lbl_r = img_r.to(device), seq_r.to(device), lbl_r.to(device)
                
                outputs_l, outputs_r = model(img_l, seq_l, img_r, seq_r)
                
                unweighted_loss_l = criterion(outputs_l, lbl_l)
                unweighted_loss_r = criterion(outputs_r, lbl_r)
                
                weight_l = torch.ones_like(lbl_l)
                weight_r = torch.ones_like(lbl_r)
                weight_l[lbl_l > 0] = args.positive_weight_left
                weight_r[lbl_r > 0] = args.positive_weight_right
                
                loss_l = (unweighted_loss_l * weight_l).mean()
                loss_r = (unweighted_loss_r * weight_r).mean()
                
                loss = loss_l + loss_r
                val_loss += loss.item() * img_l.size(0)
        
        epoch_train_loss = running_loss / len(train_dataset)
        epoch_val_loss = val_loss / len(val_dataset)
        
        tqdm.write(f"Epoch {epoch+1}/{args.num_epochs} -> Train Loss: {epoch_train_loss:.6f} | Val Loss: {epoch_val_loss:.6f}")
        
        # Log epoch metrics
        wandb.log({
            'epoch': epoch + 1,
            'train_loss_epoch': epoch_train_loss,
            'val_loss_epoch': epoch_val_loss
        })

        # Save best model
        if epoch_val_loss < best_val_loss:
            best_val_loss = epoch_val_loss
            if not os.path.exists(args.save_dir): 
                os.makedirs(args.save_dir)
            torch.save(model.state_dict(), os.path.join(args.save_dir, 'best_model_coordinated.pth'))
            tqdm.write(f"New best model saved with validation loss: {epoch_val_loss:.6f}")

    print("Training finished.")
    wandb.finish()

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Train the Coordinated multi-modal model with separate weighted loss for each arm.")
    
    # Set default paths relative to script location
    script_dir = os.path.dirname(os.path.abspath(__file__))
    default_data_dir = os.path.join(script_dir, 'prepare_result_data')
    default_save_dir = os.path.join(script_dir, 'checkpoints')

    # Path Arguments
    parser.add_argument('--data_dir', type=str, default=default_data_dir, 
                        help='Path to the data directory containing episode folders.')
    parser.add_argument('--save_dir', type=str, default=default_save_dir, 
                        help='Directory to save the trained model checkpoints.')
    
    # Training Hyperparameters
    parser.add_argument('--positive_weight_left', type=float, default=20.0, 
                        help='Weight for positive gripper actions (open/close events) for the LEFT arm.')
    parser.add_argument('--positive_weight_right', type=float, default=1.0, 
                        help='Weight for positive gripper actions for the RIGHT arm.')
    parser.add_argument('--coordination_influence', type=float, default=0.05, 
                        help='Influence factor for the cross-attention coordination module.')
    parser.add_argument('--sequence_length', type=int, default=10, 
                        help='Length of the action sequence history.')
    parser.add_argument('--learning_rate', type=float, default=1e-4, 
                        help='Learning rate for the optimizer.')
    parser.add_argument('--batch_size', type=int, default=64, 
                        help='Batch size for training.')
    parser.add_argument('--num_epochs', type=int, default=50, 
                        help='Number of epochs to train for.')
    
    # WandB Configuration
    parser.add_argument('--wandb_project', type=str, default='EDIL_Gripper_Policy', 
                        help='Wandb project name.')
    parser.add_argument('--wandb_entity', type=str, default=None, 
                        help='Wandb entity (username or team name).')
    parser.add_argument('--run_name', type=str, default='Coordinated_Run', 
                        help='A specific name for this training run.')

    args = parser.parse_args()
    
    print(f"Model will be saved to: {args.save_dir}")
    print(f"Configuration: Left Weight={args.positive_weight_left}, Right Weight={args.positive_weight_right}, Coord Influence={args.coordination_influence}")
    
    train_model(args)
