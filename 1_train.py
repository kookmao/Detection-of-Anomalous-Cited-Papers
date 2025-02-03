import os
import sys
import argparse
import numpy as np
import torch
from codes.DynamicDatasetLoader import DynamicDatasetLoader
from codes.Component import MyConfig
from codes.DynADModel import DynADModel
from codes.Settings import Settings

def verify_cuda_availability():
    """Verify CUDA status and memory"""
    if torch.cuda.is_available():
        device = torch.device("cuda")
        device_props = torch.cuda.get_device_properties(device)
        print(f"Using GPU: {torch.cuda.get_device_name(0)}")
        print(f"Available memory: {device_props.total_memory / 1e9:.2f} GB")
        torch.cuda.empty_cache()
    else:
        print("CUDA not available, using CPU")
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")

def parse_arguments():
    """Parse and validate command line arguments"""
    parser = argparse.ArgumentParser(description='Dynamic Anomaly Detection Training')
    
    # Dataset parameters
    parser.add_argument('--dataset', type=str, 
                       choices=['uci', 'digg', 'btc_alpha', 'btc_otc', 'year_1992', 'year_1993', 'five_year'],
                       default='uci')
    parser.add_argument('--anomaly_per', type=float, 
                       choices=[0.01, 0.05, 0.1, 0.2],
                       default=0.1)
    parser.add_argument('--train_per', type=float, default=0.5)
    
    # Model architecture
    parser.add_argument('--neighbor_num', type=int, default=4)
    parser.add_argument('--window_size', type=int, default=2)
    parser.add_argument('--embedding_dim', type=int, default=32)
    parser.add_argument('--num_hidden_layers', type=int, default=2)
    parser.add_argument('--num_attention_heads', type=int, default=2)
    parser.add_argument('--batch_size', type=int, default=64)
    
    # Training parameters
    parser.add_argument('--max_epoch', type=int, default=100)
    parser.add_argument('--lr', type=float, default=0.0002)
    parser.add_argument('--weight_decay', type=float, default=2e-3)
    parser.add_argument('--seed', type=int, default=1)
    parser.add_argument('--print_feq', type=int, default=10)
    
    args = parser.parse_args()
    
    # Validate arguments
    if args.window_size < 2:
        raise ValueError("window_size must be at least 2")
    if args.batch_size < 1:
        raise ValueError("batch_size must be positive")
    if args.lr <= 0:
        raise ValueError("Learning rate must be positive")
    
    return args

def setup_environment(args):
    """Configure training environment"""
    # Set random seeds
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(args.seed)
        torch.cuda.manual_seed_all(args.seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

def initialize_components(args, device):
    """Initialize model components with validation"""
    # Initialize dataset
    data_obj = DynamicDatasetLoader()
    data_obj.dataset_name = args.dataset
    data_obj.k = args.neighbor_num
    data_obj.window_size = args.window_size
    data_obj.anomaly_per = args.anomaly_per
    data_obj.train_per = args.train_per
    data_obj.load_all_tag = False
    data_obj.compute_s = True
    
    # Initialize model configuration
    my_config = MyConfig(
        k=args.neighbor_num,
        window_size=args.window_size,
        hidden_size=args.embedding_dim,
        intermediate_size=args.embedding_dim,
        num_attention_heads=args.num_attention_heads,
        num_hidden_layers=args.num_hidden_layers,
        weight_decay=args.weight_decay,
        batch_size=args.batch_size
    )
    
    # Initialize model
    method_obj = DynADModel(my_config, args)
    method_obj.spy_tag = True
    method_obj.max_epoch = args.max_epoch
    method_obj.lr = args.lr
    
    return data_obj, method_obj

def main():
    """Main training execution"""
    # Parse arguments and setup
    args = parse_arguments()
    device = verify_cuda_availability()
    setup_environment(args)
    
    print('\n$$$$ Starting Training $$$$')
    
    try:
        # Initialize components
        data_obj, method_obj = initialize_components(args, device)
        
        # Initialize settings
        setting_obj = Settings()
        setting_obj.prepare(data_obj, method_obj)
        
        # Execute training
        setting_obj.run()
        
        print('\n$$$$ Training Complete $$$$')
        
    except Exception as e:
        print(f"\nError during training: {str(e)}")
        raise
    finally:
        # Cleanup
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

if __name__ == '__main__':
    main()