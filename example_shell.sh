python -u train.py \
    --dataroot /scratch/sibwang/contrastive-unpaired-translation/bulk_data/style_transfer_cut/datasets/spotlight202506_to_aymanns2022 \
    --name spotlight202506_to_aymanns2022 \
    --CUT_mode CUT \
    --checkpoints_dir /scratch/sibwang/contrastive-unpaired-translation/bulk_data/style_transfer_cut/checkpoints/test_trial \
    --tensorboard_log_dir /scratch/sibwang/contrastive-unpaired-translation/bulk_data/style_transfer_cut/logs/test_trial \
    --n_epochs 40 \
    --n_epochs_decay 40 \
    --save_epoch_freq 1 \
    --batch_size 1