import argparse
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from bc_dataset import ReacherHDF5Dataset
from bc_model import StateBCPolicy, VisualBCPolicy

def evaluate(args):
    # Select Device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Evaluating on device: {device}")
    
    # 1. Initialize Test Dataset
    # We strictly use split='test' to ensure the AI has never seen this data
    test_dataset = ReacherHDF5Dataset(args.dataset, mode=args.mode, split='test', max_samples=args.max_samples)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=2)
    
    # 2. Initialize Model Architecture
    if args.mode == 'state':
        model = StateBCPolicy().to(device)
    else:
        model = VisualBCPolicy().to(device)
        
    # 3. Load Saved Checkpoint Weights
    print(f"Loading checkpoint from: {args.checkpoint}")
    model.load_state_dict(torch.load(args.checkpoint, map_location=device))
    
    # Set model to evaluation mode (disables dropout/batchnorm randomness)
    model.eval()
    
    criterion = nn.MSELoss()
    total_test_loss = 0.0
    
    print("Running evaluation on hidden Test Set...")
    with torch.no_grad(): # Disable gradients for memory efficiency
        for batch_data in test_loader:
            if args.mode == 'state':
                obs, actions = batch_data
                obs, actions = obs.to(device), actions.to(device)
                pred_actions = model(obs)
            elif args.mode == 'visual':
                imgs, targets, actions = batch_data
                imgs, targets, actions = imgs.to(device), targets.to(device), actions.to(device)
                pred_actions = model(imgs, targets)
                
            loss = criterion(pred_actions, actions)
            total_test_loss += loss.item()
            
    avg_test_loss = total_test_loss / len(test_loader) if len(test_loader) > 0 else 0
    
    print("="*40)
    print(f" FINAL TEST MSE ({args.mode.upper()} MODE): {avg_test_loss:.4f}")
    print("="*40)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate Behavior Cloning Policy")
    parser.add_argument('--dataset', type=str, default='/data/5pourghe/le-wm-v2/le-wm-storage/reacher.h5', help='Path to dataset')
    parser.add_argument('--mode', type=str, choices=['state', 'visual'], required=True, help='Evaluate state or visual policy')
    parser.add_argument('--checkpoint', type=str, required=True, help='Path to the .pth model checkpoint file')
    parser.add_argument('--batch_size', type=int, default=64, help='Batch size')
    parser.add_argument('--max_samples', type=int, default=None, help='Limit test dataset size')
    
    args = parser.parse_args()
    evaluate(args)
