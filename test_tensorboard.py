#!/usr/bin/env python3
"""
Simple test script to verify TensorBoard integration works correctly.
This script creates dummy data and logs it to TensorBoard to test the visualizer.
"""

import torch
import numpy as np
from collections import OrderedDict
import os
import sys
import wandb

# Add the project root to the path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from options.train_options import TrainOptions
from util.visualizer import Visualizer
from util import util

def test_tensorboard_logging():
    """Test function to verify TensorBoard logging works"""
    print("Testing TensorBoard integration...")
    
    # Create dummy options
    opt = TrainOptions().parse()
    opt.name = "tensorboard_test"
    opt.checkpoints_dir = "./test_checkpoints"
    opt.isTrain = True
    opt.no_html = True
    opt.display_id = 1
    opt.tensorboard_log_dir = "./test_logs"
    
    # Initialize wandb if project name is provided
    if opt.wandb_project:
        wandb.init(project=opt.wandb_project, name=opt.name, config=vars(opt))
    
    # Create visualizer
    visualizer = Visualizer(opt)
    
    # Create dummy visuals (fake images)
    visuals = OrderedDict()
    visuals['real_A'] = torch.randn(1, 3, 256, 256)  # Random RGB image, batch size 1
    visuals['fake_B'] = torch.randn(1, 3, 256, 256)
    visuals['real_B'] = torch.randn(1, 3, 256, 256)
    
    # Create dummy losses
    losses = OrderedDict()
    losses['G_GAN'] = 0.5
    losses['D_real'] = 0.3
    losses['D_fake'] = 0.4
    losses['G'] = 0.8
    losses['NCE'] = 0.2
    
    # Test logging for a few epochs
    for epoch in range(1, 4):
        print(f"Logging epoch {epoch}...")
        
        # Log images
        visualizer.display_current_results(visuals, epoch, True)
        
        # Log images to wandb (concatenated if possible)
        concat_keys = ['real_A', 'fake_B', 'real_B', 'idt_B']
        images_to_concat = []
        for k in concat_keys:
            if k in visuals:
                image_numpy = util.tensor2im(visuals[k])
                images_to_concat.append(image_numpy)
        if opt.wandb_project and len(images_to_concat) > 0:
            concat_image = np.concatenate(images_to_concat, axis=1) if len(images_to_concat) > 1 else images_to_concat[0]  # HWC format for wandb
            wandb.log({"results": [wandb.Image(concat_image, caption=f"Epoch {epoch}")]}, step=epoch)
        
        # Log losses
        for iter_count in range(0, 100, 20):
            counter_ratio = iter_count / 100.0
            # Add some noise to losses to simulate training dynamics
            noisy_losses = OrderedDict()
            for k, v in losses.items():
                noisy_losses[k] = v + np.random.normal(0, 0.1)
            
            visualizer.plot_current_losses(epoch, counter_ratio, noisy_losses)
            visualizer.print_current_losses(epoch, iter_count, noisy_losses, 0.1, 0.05)
            if opt.wandb_project:
                wandb.log({**noisy_losses, "epoch": epoch, "iter": iter_count}, step=epoch * 100 + iter_count)
    
    # Close visualizer
    visualizer.close()
    if opt.wandb_project:
        wandb.finish()
    
    print("\nTensorBoard test completed!")
    print(f"Logs saved to: {opt.tensorboard_log_dir}/{opt.name}")
    print(f"To view results, run: tensorboard --logdir={opt.tensorboard_log_dir}")
    print("Then open http://localhost:6006 in your browser")

if __name__ == "__main__":
    test_tensorboard_logging()
