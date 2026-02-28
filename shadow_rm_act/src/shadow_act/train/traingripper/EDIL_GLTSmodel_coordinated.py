import torch
import torch.nn as nn
import torchvision.models as models
from torch.utils.data import Dataset, DataLoader, ConcatDataset
import pandas as pd
import cv2
import numpy as np
from torchvision import transforms
import os
import math

# 1. Data Loading Section: Dual-arm data loading mode
class GripperEpisodeDataset(Dataset):
    def __init__(self, left_csv_file, left_video_file, right_csv_file, right_video_file, sequence_length=10, transform=None):
        self.sequence_length = sequence_length
        self.transform = transform

        # --- Left Arm Data ---
        self.left_video_file = left_video_file
        self.left_action_data = pd.read_csv(left_csv_file)
        self.left_video_capture = None
        
        # Verify video integrity
        temp_cap_left = cv2.VideoCapture(self.left_video_file)
        if not temp_cap_left.isOpened(): 
            raise IOError(f"Cannot open left video file {self.left_video_file}")
        self.left_num_frames = int(temp_cap_left.get(cv2.CAP_PROP_FRAME_COUNT))
        temp_cap_left.release()
        
        self.left_num_samples = min(self.left_num_frames, len(self.left_action_data)) - self.sequence_length + 1

        # --- Right Arm Data ---
        self.right_video_file = right_video_file
        self.right_action_data = pd.read_csv(right_csv_file)
        self.right_video_capture = None
        
        temp_cap_right = cv2.VideoCapture(self.right_video_file)
        if not temp_cap_right.isOpened(): 
            raise IOError(f"Cannot open right video file {self.right_video_file}")
        self.right_num_frames = int(temp_cap_right.get(cv2.CAP_PROP_FRAME_COUNT))
        temp_cap_right.release()
        
        self.right_num_samples = min(self.right_num_frames, len(self.right_action_data)) - self.sequence_length + 1
            
        self.num_samples = min(self.left_num_samples, self.right_num_samples)

    def __len__(self):
        return self.num_samples
    
    def _get_video_capture(self, arm):
        if arm == 'left':
            if self.left_video_capture is None:
                self.left_video_capture = cv2.VideoCapture(self.left_video_file)
            return self.left_video_capture
        else: # 'right'
            if self.right_video_capture is None:
                self.right_video_capture = cv2.VideoCapture(self.right_video_file)
            return self.right_video_capture

    def __getitem__(self, idx):
        # Left Arm
        left_cap = self._get_video_capture('left')
        left_action_seq = self.left_action_data.iloc[idx:idx + self.sequence_length, :-1].values
        left_action_tensor = torch.tensor(left_action_seq, dtype=torch.float32)
        
        frame_idx_left = idx + self.sequence_length - 1
        left_cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx_left)
        ret_left, frame_left = left_cap.read()
        
        if not ret_left: # Fallback for last frame
            left_cap.set(cv2.CAP_PROP_POS_FRAMES, self.left_num_frames - 1)
            ret_left, frame_left = left_cap.read()
            
        frame_left = cv2.cvtColor(frame_left, cv2.COLOR_BGR2RGB)
        if self.transform: 
            frame_left = self.transform(frame_left)
            
        label_left = self.left_action_data.iloc[idx + self.sequence_length - 1, -1]
        label_tensor_left = torch.tensor(float(label_left), dtype=torch.float32)

        # Right Arm
        right_cap = self._get_video_capture('right')
        right_action_seq = self.right_action_data.iloc[idx:idx + self.sequence_length, :-1].values
        right_action_tensor = torch.tensor(right_action_seq, dtype=torch.float32)
        
        frame_idx_right = idx + self.sequence_length - 1
        right_cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx_right)
        ret_right, frame_right = right_cap.read()
        
        if not ret_right: # Fallback for last frame
            right_cap.set(cv2.CAP_PROP_POS_FRAMES, self.right_num_frames - 1)
            ret_right, frame_right = right_cap.read()
            
        frame_right = cv2.cvtColor(frame_right, cv2.COLOR_BGR2RGB)
        if self.transform: 
            frame_right = self.transform(frame_right)
            
        label_right = self.right_action_data.iloc[idx + self.sequence_length - 1, -1]
        label_tensor_right = torch.tensor(float(label_right), dtype=torch.float32)

        return frame_left, left_action_tensor, label_tensor_left, frame_right, right_action_tensor, label_tensor_right

    def __del__(self):
        if self.left_video_capture: self.left_video_capture.release()
        if self.right_video_capture: self.right_video_capture.release()


class GripperDataset(ConcatDataset):
    def __init__(self, left_csv_files, left_video_files, right_csv_files, right_video_files, sequence_length=10, transform=None):
        datasets = []
        for l_csv, l_vid, r_csv, r_vid in zip(left_csv_files, left_video_files, right_csv_files, right_video_files):
            try:
                dataset = GripperEpisodeDataset(l_csv, l_vid, r_csv, r_vid, sequence_length, transform)
                if len(dataset) > 0: datasets.append(dataset)
            except Exception as e:
                print(f"Warning: Could not load episode from {os.path.basename(l_csv)}: {e}, skipping.")
        super().__init__(datasets)

# 2. Hybrid Visual Feature Extraction
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super().__init__()
        # Create constant 'pe' matrix
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, x):
        # x: [batch_size, seq_len, d_model]
        x = x + self.pe[:x.size(1), :]
        return x

class RobustVisionModule(nn.Module):
    def __init__(self, hidden_dim=128, transformer_layers=2, nhead=4):
        super().__init__()
        # ResNet Backbone (up to layer4)
        resnet = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
        self.backbone = nn.Sequential(*list(resnet.children())[:-2])
        
        # Freeze early layers
        for param in self.backbone.parameters():
            param.requires_grad = False
        # Finetune layer4
        for param in self.backbone[-1].parameters():
            param.requires_grad = True
            
        # 1x1 Conv to reduce dimension to hidden_dim (D)
        self.conv1x1 = nn.Conv2d(512, hidden_dim, kernel_size=1)
        
        # Transformer Encoder
        self.pos_encoder = PositionalEncoding(hidden_dim)
        encoder_layer = nn.TransformerEncoderLayer(d_model=hidden_dim, nhead=nhead, batch_first=True)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=transformer_layers)
        
        # Final projection head
        self.norm_final = nn.LayerNorm(hidden_dim)
        self.head = nn.Linear(hidden_dim, hidden_dim)

    def forward(self, x):
        # x: [B, 3, H, W]
        # Backbone features: [B, 512, H', W']
        features = self.backbone(x)
        
        # 1x1 Conv: [B, D, H', W']
        features = self.conv1x1(features)
        
        # Serialize: [B, D, N] where N = H' * W'
        b, d, h, w = features.shape
        features = features.flatten(2) 
        
        # Transpose for Transformer: [B, N, D]
        features = features.transpose(1, 2)
        
        # Positional Encoding (Eq 12)
        features = self.pos_encoder(features)
        
        # Transformer Encoder (Eq 13, 14)
        # Output: [B, N, D]
        context_features = self.transformer_encoder(features)
        
        # Global Average Pooling (Eq 15)
        # f_t_pooled: [B, D]
        pooled_features = context_features.mean(dim=1)
        
        # Final Head
        # f_t_vision: [B, D]
        out = self.head(self.norm_final(pooled_features))
        
        return out

# 3. Core Coordination Module (Eq 18-20)
class GatedCrossAttention(nn.Module):
    def __init__(self, embed_dim, num_heads=4, influence_factor=1.0):
        super().__init__()
        self.embed_dim = embed_dim
        self.influence_factor = influence_factor
        
        # Multi-head attention (Internal W_Q, W_K, W_V)
        self.multihead_attn = nn.MultiheadAttention(embed_dim, num_heads, batch_first=True, dropout=0.1)
        
        # Gate mechanism (Eq 20.0)
        self.gate_layer = nn.Sequential(nn.Linear(embed_dim, 1), nn.Sigmoid())
        
        # LayerNorm (for Eq 20)
        self.norm_out = nn.LayerNorm(embed_dim)
        
    def forward(self, query_state, context_states):
        """
        query_state: [B, D] - f_Intra (The arm asking for info)
        context_states: [B, 2, D] - S_R/S_L (The other arm's modalities: [seq, vision])
        """
        # Prepare Query: [B, 1, D]
        query = query_state.unsqueeze(1)
        
        # Key/Value: context_states is already [B, 2, D]
        key = context_states
        value = context_states
        
        # MHSA (Eq 20 inner part)
        attn_output, _ = self.multihead_attn(query=query, key=key, value=value)
        # attn_output: [B, 1, D]
        
        attn_output = self.norm_out(attn_output.squeeze(1)) # [B, D]
        
        # Gate (Eq 20.0)
        gate = self.gate_layer(query_state) # [B, 1]
        
        # Final Coordination Vector (Eq 20)
        # influence_factor allows external control/ablation
        gated_context = self.influence_factor * gate * attn_output
        
        return gated_context

# 4. Final Main Model
class CoordinatedMultiModalGripperNet(nn.Module):
    def __init__(self, num_joints=6, hidden_dim=128, num_layers=2, n_classes=1, num_heads=4, coordination_influence=1.0):
        super().__init__()
        
        print(f"Model initialized with coordination influence factor: {coordination_influence}")
        
        self.hidden_dim = hidden_dim
        
        # --- Vision Modules ---
        self.vision_module_left = RobustVisionModule(hidden_dim, nhead=num_heads)
        self.vision_module_right = RobustVisionModule(hidden_dim, nhead=num_heads)
        
        # --- Sequence Modules ---
        # Eq 7: Embedding layer
        self.joint_embedding_left = nn.Linear(num_joints, hidden_dim)
        self.joint_embedding_right = nn.Linear(num_joints, hidden_dim)
        
        # GRUs
        self.gru_left = nn.GRU(input_size=hidden_dim, hidden_size=hidden_dim, num_layers=num_layers, batch_first=True)
        self.gru_right = nn.GRU(input_size=hidden_dim, hidden_size=hidden_dim, num_layers=num_layers, batch_first=True)
        
        # --- Intra-arm Fusion ---
        # Eq 16/17: Linear + ReLU
        self.fusion_layer_left = nn.Sequential(nn.Linear(hidden_dim * 2, hidden_dim), nn.ReLU())
        self.fusion_layer_right = nn.Sequential(nn.Linear(hidden_dim * 2, hidden_dim), nn.ReLU())
        
        # --- Coordination Modules ---
        # Pass influence_factor to the coordination module
        self.coord_L_asks_R = GatedCrossAttention(hidden_dim, num_heads, influence_factor=coordination_influence)
        self.coord_R_asks_L = GatedCrossAttention(hidden_dim, num_heads, influence_factor=coordination_influence)

        # --- Decision Heads ---
        # Added Sigmoid at the end to output range [0, 1]
        self.decision_head_left = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2), 
            nn.ReLU(), 
            nn.Linear(hidden_dim // 2, n_classes),
            nn.Sigmoid()
        )
        self.decision_head_right = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2), 
            nn.ReLU(), 
            nn.Linear(hidden_dim // 2, n_classes),
            nn.Sigmoid()
        )

    def forward(self, image_left, sequence_left, image_right, sequence_right):
        # --- 1. Feature Extraction ---
        
        # Vision Features: [B, D]
        f_vision_L = self.vision_module_left(image_left)
        f_vision_R = self.vision_module_right(image_right)
        
        # Sequence Features
        # Embed joints first (Eq 7): [B, T, D]
        seq_emb_L = self.joint_embedding_left(sequence_left)
        seq_emb_R = self.joint_embedding_right(sequence_right)
        
        # GRU: output [B, T, D], h_n [L, B, D]
        # We need the hidden state of the final layer at the final timestep
        _, h_n_L = self.gru_left(seq_emb_L)
        _, h_n_R = self.gru_right(seq_emb_R)
        
        f_seq_L = h_n_L[-1] # [B, D]
        f_seq_R = h_n_R[-1] # [B, D]
        
        # --- 2. Intra-arm Fusion (Eq 16, 17) ---
        f_intra_L = self.fusion_layer_left(torch.cat((f_seq_L, f_vision_L), dim=1))
        f_intra_R = self.fusion_layer_right(torch.cat((f_seq_R, f_vision_R), dim=1))
        
        # --- 3. Dynamic Coordination (Eq 18-20) ---
        
        # Construct Source Contexts S_R and S_L (Eq 18/19 Inputs)
        # S_R = [f_seq_R, f_vision_R] -> [B, 2, D]
        S_R = torch.stack([f_seq_R, f_vision_R], dim=1)
        S_L = torch.stack([f_seq_L, f_vision_L], dim=1)
        
        # Cross Attention
        # Left asks Right
        c_R = self.coord_L_asks_R(query_state=f_intra_L, context_states=S_R)
        
        # Right asks Left
        c_L = self.coord_R_asks_L(query_state=f_intra_R, context_states=S_L)
        
        # --- 4. Execution Feature (Eq 22) ---
        # Integrate learned coordination context into the arm's state
        f_exec_L = f_intra_L + c_R
        f_exec_R = f_intra_R + c_L
        
        # --- 5. Decision ---
        output_left = self.decision_head_left(f_exec_L)
        output_right = self.decision_head_right(f_exec_R)
        
        return output_left.squeeze(1), output_right.squeeze(1)

# 5. Data Augmentation
def get_data_transforms():
    return transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((224, 224)),
        transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3),
        transforms.RandomAffine(degrees=5, translate=(0.05, 0.05), scale=(0.95, 1.05)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])


if __name__ == '__main__':
    # Usage Example
    
    # Configure paths
    # Please replace the following paths with your actual data paths
    left_csv_path = "<PATH_TO_LEFT_ACTION_CSV>"
    left_video_path = "<PATH_TO_LEFT_VIDEO_MP4>"
    right_csv_path = "<PATH_TO_RIGHT_ACTION_CSV>"
    right_video_path = "<PATH_TO_RIGHT_VIDEO_MP4>"
    
    transform = get_data_transforms()
    
    # Check if paths are valid (only if user has set them)
    if all(not p.startswith("<PATH") and os.path.exists(p) for p in [left_csv_path, left_video_path, right_csv_path, right_video_path]):
        dataset = GripperDataset(
            left_csv_files=[left_csv_path], 
            left_video_files=[left_video_path],
            right_csv_files=[right_csv_path], 
            right_video_files=[right_video_path],
            transform=transform
        )
        
        if len(dataset) > 0:
            dataloader = DataLoader(dataset, batch_size=4, shuffle=True)
            
            # Get a batch for testing
            images_l, seq_l, label_l, images_r, seq_r, label_r = next(iter(dataloader))
            print("Left Images batch shape:", images_l.shape)
            print("Left Sequences batch shape:", seq_l.shape)

            # Test Model
            model = CoordinatedMultiModalGripperNet()
            output_l, output_r = model(images_l, seq_l, images_r, seq_r)
            print("\n--- Model Test ---")
            print("Left Output shape:", output_l.shape)
            print("Right Output shape:", output_r.shape)
            print("Output range example:", output_l[0].item()) # Should be between 0 and 1

        else:
            print("Dataset is empty.")
    else:
        print("Example paths are not configured. Please set paths in __main__ to test.")
