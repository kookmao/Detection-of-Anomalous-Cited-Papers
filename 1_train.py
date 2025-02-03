import os
import torch
from typing import Optional, Dict, Any
import numpy as np
import argparse
from pathlib import Path


if torch.cuda.is_available():
    torch.cuda.empty_cache()
    for device in range(torch.cuda.device_count()):
        torch.cuda.set_device(device)
        # Set larger chunk sizes
        torch.cuda.memory.set_per_process_memory_fraction(0.95, device)
        # Enable active memory management
        torch.cuda.set_per_process_memory_fraction(0.95, device)
    # Force garbage collection
    torch.cuda.empty_cache()

from codes.DynamicDatasetLoader import DynamicDatasetLoader
from codes.Component import MyConfig
from codes.DynADModel import DynADModel
from codes.Settings import Settings



# Configure CUDA memory allocation
os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'max_split_size_mb:128,garbage_collection_threshold:0.8'

def create_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    # Dataset params
    parser.add_argument('--dataset', type=str, 
                       choices=['uci', 'digg', 'btc_alpha', 'btc_otc',
                               'year_1992','year_1993','five_year'], 
                       default='uci')
    parser.add_argument('--anomaly_per', type=float,
                       choices=[0.01, 0.05, 0.1, 0.2], 
                       default=0.1)
    parser.add_argument('--train_per', type=float, 
                       default=0.5)

    # Model params
    parser.add_argument('--neighbor_num', type=int, default=4)
    parser.add_argument('--window_size', type=int, default=2)
    parser.add_argument('--embedding_dim', type=int, default=32)
    parser.add_argument('--num_hidden_layers', type=int, default=2)
    parser.add_argument('--num_attention_heads', type=int, default=2)
    parser.add_argument('--batch_size', type=int, default=16)

    # Training params
    parser.add_argument('--max_epoch', type=int, default=100)
    parser.add_argument('--lr', type=float, default=0.0002)
    parser.add_argument('--weight_decay', type=float, default=2e-3)
    parser.add_argument('--seed', type=int, default=1)
    parser.add_argument('--print_freq', type=int, default=10)
    
    # Checkpointing
    parser.add_argument('--checkpoint_dir', type=str, default='checkpoints')
    parser.add_argument('--resume', type=str, default='')
    
    return parser

def save_checkpoint(state: Dict[str, Any], is_best: bool, 
                   checkpoint_dir: str, filename: str = 'checkpoint.pt') -> None:
    filepath = os.path.join(checkpoint_dir, filename)
    torch.save(state, filepath)
    if is_best:
        best_fpath = os.path.join(checkpoint_dir, 'model_best.pt')
        torch.save(state, best_fpath)

def setup_environment(args):
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(args.seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = True
        
        # Print initial memory state
        print("\nInitial GPU Memory Configuration:")
        for i in range(torch.cuda.device_count()):
            props = torch.cuda.get_device_properties(i)
            print(f"GPU {i}: {props.name}")
            print(f"Total Memory: {props.total_memory/1024**2:.0f}MB")
            print(f"Allocated: {torch.cuda.memory_allocated(i)/1024**2:.0f}MB")
            print(f"Cached: {torch.cuda.memory_reserved(i)/1024**2:.0f}MB")
            print("---")

def main(args: argparse.Namespace) -> None:
    setup_environment(args)
    
    # Initialize dataset with proper memory pinning
    data_obj = DynamicDatasetLoader()
    data_obj.dataset_name = args.dataset
    data_obj.k = args.neighbor_num
    data_obj.window_size = args.window_size
    data_obj.anomaly_per = args.anomaly_per
    data_obj.train_per = args.train_per
    data_obj.load_all_tag = False
    data_obj.compute_s = True
    
    # Initialize model config
    config = MyConfig(
        k=args.neighbor_num,
        window_size=args.window_size,
        hidden_size=args.embedding_dim,
        intermediate_size=args.embedding_dim,
        num_attention_heads=args.num_attention_heads,
        num_hidden_layers=args.num_hidden_layers,
        weight_decay=args.weight_decay,
        batch_size=args.batch_size
    )
    
    # Initialize model and move to GPU
    model = DynADModel(config, args)
    if torch.cuda.is_available():
        model = model.cuda()
    model.spy_tag = True
    model.max_epoch = args.max_epoch
    model.lr = args.lr
    
    # Resume from checkpoint if specified
    start_epoch = 0
    best_auc = 0
    if args.resume:
        if os.path.isfile(args.resume):
            checkpoint = torch.load(args.resume, map_location='cuda')
            start_epoch = checkpoint['epoch']
            best_auc = checkpoint['best_auc']
            model.load_state_dict(checkpoint['state_dict'])
    
    # Initialize settings and run
    setting_obj = Settings()
    setting_obj.prepare(data_obj, model)
    
    try:
        setting_obj.run()
        
        # Save final checkpoint
        if args.checkpoint_dir:
            save_checkpoint({
                'epoch': args.max_epoch,
                'state_dict': model.state_dict(),
                'best_auc': model.best_auc if hasattr(model, 'best_auc') else 0,
                'config': vars(args)
            }, False, args.checkpoint_dir)
            
    except Exception as e:
        print(f"Error during training: {str(e)}")
        raise
    finally:
        # Clean up
        torch.cuda.empty_cache()

if __name__ == '__main__':
    parser = create_parser()
    args = parser.parse_args()
    main(args)