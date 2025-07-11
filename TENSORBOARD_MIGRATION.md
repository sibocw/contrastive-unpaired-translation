# TensorBoard Migration Summary

This document summarizes the changes made to replace Visdom with TensorBoard for visualization in the Contrastive Unpaired Translation (CUT) project.

## Files Modified

### 1. Dependencies
- **requirements.txt**: Replaced `visdom>=0.1.8.8` with `tensorboard>=2.0.0`
- **environment.yml**: Replaced `visdom==0.1.8` with `tensorboard>=2.0.0`

### 2. Configuration
- **options/train_options.py**:
  - Updated comments to reference TensorBoard instead of Visdom
  - Changed `--display_server` to `--tensorboard_log_dir` (default: "./logs")
  - Changed default `--display_port` from 8097 to 6006 (TensorBoard default)
  - Removed Visdom-specific server options

- **options/base_options.py**:
  - Updated help text to reference TensorBoard instead of Visdom

### 3. Core Visualization
- **util/visualizer.py**: Complete rewrite to use TensorBoard
  - Replaced Visdom imports with `torch.utils.tensorboard.SummaryWriter`
  - Updated `__init__()` to create TensorBoard writer instead of Visdom connection
  - Modified `display_current_results()` to log images using `writer.add_image()`
  - Updated `plot_current_losses()` to log scalars using `writer.add_scalar()`
  - Enhanced `print_current_losses()` to also log timing information to TensorBoard
  - Added `close()` method to properly close TensorBoard writer
  - Removed all Visdom-specific code (connection handling, server creation, etc.)

### 4. Training Scripts
- **train.py**: 
  - Updated comment to reference TensorBoard
  - Added call to `visualizer.close()` at the end of training

- **test.py**: Updated comment to reference TensorBoard

### 5. Model Files
- **models/base_model.py**: Updated comments to reference TensorBoard instead of Visdom

### 6. Documentation
- **README.md**: 
  - Updated dependency list to mention TensorBoard instead of Visdom
  - Changed server startup instructions from `python -m visdom.server` to `tensorboard --logdir=./logs`
  - Updated URL from http://localhost:8097 to http://localhost:6006

## Key Changes in Functionality

### What's Logged to TensorBoard:
1. **Images**: All visual results (real_A, fake_B, real_B, etc.) are logged as images
2. **Loss Curves**: All training losses are logged as scalar plots
3. **Timing Information**: Computation time and data loading time are tracked
4. **Organized Structure**: Losses are grouped under "losses/" namespace for better organization

### TensorBoard Advantages:
1. **Better Integration**: Native PyTorch integration, no external server needed
2. **Wandb Compatibility**: TensorBoard logs can be easily imported into Weights & Biases
3. **Performance**: More efficient and lightweight than Visdom
4. **Features**: Better plot organization, histogram support, and advanced visualization options

## Usage Instructions

### Starting TensorBoard:
```bash
# Start TensorBoard server
tensorboard --logdir=./logs

# Access via browser
# Open http://localhost:6006
```

### Training with TensorBoard:
```bash
# The training command remains the same
python train.py --dataroot ./datasets/grumpifycat --name grumpycat_CUT --CUT_mode CUT

# TensorBoard logs will be automatically saved to ./logs/grumpycat_CUT/
```

### Viewing Results:
1. Start TensorBoard: `tensorboard --logdir=./logs`
2. Open browser to http://localhost:6006
3. View:
   - **Scalars tab**: Loss curves and timing information
   - **Images tab**: Generated images from each epoch

## Testing

A test script `test_tensorboard.py` has been created to verify the TensorBoard integration works correctly. Run it to test the logging functionality:

```bash
python test_tensorboard.py
```

## Migration Notes

- All previous Visdom functionality has been preserved but implemented using TensorBoard
- HTML saving functionality remains unchanged
- Console logging remains unchanged
- The same data is logged, just using a different backend
- TensorBoard logs are more structured and easier to analyze programmatically

## Future Wandb Integration

With TensorBoard logging in place, integrating with Weights & Biases becomes straightforward:

```python
import wandb

# Initialize wandb
wandb.init(project="contrastive-unpaired-translation")

# Log to both TensorBoard and wandb
wandb.log({"loss/G_GAN": loss_value, "epoch": epoch})
```

Or use wandb's TensorBoard sync:
```bash
wandb sync ./logs/experiment_name
```
