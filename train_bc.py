import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import os

from bc_dataset import ReacherHDF5Dataset
from bc_model import StateBCPolicy, VisualBCPolicy

def train(args):
    # Select Device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # 1. Initialize Datasets & DataLoaders
    train_dataset = ReacherHDF5Dataset(args.dataset, mode=args.mode, split='train', max_samples=args.max_samples)
    val_dataset = ReacherHDF5Dataset(args.dataset, mode=args.mode, split='val', max_samples=args.max_samples)
    
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=2)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=2)
    
    # 2. Initialize Model
    if args.mode == 'state':
        model = StateBCPolicy().to(device)
    else:
        model = VisualBCPolicy().to(device)
        
    print(f"Model Initialized. Total Parameters: {sum(p.numel() for p in model.parameters())}")
    
    # 3. Define Optimizer and Loss
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    criterion = nn.MSELoss() # Behavioral Cloning uses Mean Squared Error against true actions
    
    # 4. Training Loop
    os.makedirs('checkpoints', exist_ok=True)
    
    for epoch in range(1, args.epochs + 1):
        # --- TRAIN PHASE ---
        model.train()
        train_loss = 0.0
        
        for batch_idx, batch_data in enumerate(train_loader):
            optimizer.zero_grad()
            
            if args.mode == 'state':
                obs, actions = batch_data
                obs, actions = obs.to(device), actions.to(device)
                pred_actions = model(obs)
                
            elif args.mode == 'visual':
                imgs, targets, actions = batch_data
                imgs, targets, actions = imgs.to(device), targets.to(device), actions.to(device)
                pred_actions = model(imgs, targets)
                
            loss = criterion(pred_actions, actions)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            
        avg_train_loss = train_loss / len(train_loader) if len(train_loader) > 0 else 0
        
        # --- VALIDATION PHASE ---
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for batch_data in val_loader:
                if args.mode == 'state':
                    obs, actions = batch_data
                    obs, actions = obs.to(device), actions.to(device)
                    pred_actions = model(obs)
                elif args.mode == 'visual':
                    imgs, targets, actions = batch_data
                    imgs, targets, actions = imgs.to(device), targets.to(device), actions.to(device)
                    pred_actions = model(imgs, targets)
                    
                loss = criterion(pred_actions, actions)
                val_loss += loss.item()
                
        avg_val_loss = val_loss / len(val_loader) if len(val_loader) > 0 else 0
        
        print(f"Epoch {epoch}/{args.epochs} | Train MSE: {avg_train_loss:.4f} | Val MSE: {avg_val_loss:.4f}")
        
        # Save Checkpoint
        save_path = f"checkpoints/{args.mode}_policy_epoch_{epoch}.pth"
        torch.save(model.state_dict(), save_path)
        print(f"Saved Checkpoint: {save_path}\n")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train Behavior Cloning Policy for Reacher")
    parser.add_argument('--dataset', type=str, default='/data/5pourghe/le-wm-v2/le-wm-storage/reacher.h5', help='Path to dataset')
    parser.add_argument('--mode', type=str, choices=['state', 'visual'], required=True, help='Train state or visual policy')
    parser.add_argument('--epochs', type=int, default=5, help='Number of training epochs')
    parser.add_argument('--batch_size', type=int, default=64, help='Batch size')
    parser.add_argument('--lr', type=float, default=1e-3, help='Learning rate')
    parser.add_argument('--max_samples', type=int, default=None, help='Limit dataset size for fast testing')
    
    args = parser.parse_args()
    train(args)
