from pathlib import Path

"""
--ngf (# of gen filters in the last conv layer, default=64)         32 64
--netG (generator architecture, default=resnet_9blocks)             resnet_6blocks resnet_9blocks
--batch_size (batch size, default=1)                                1 4
--lambda_GAN (weight for GAN loss, default=1.0)                     1.0 3.0
"""

template_path = Path("/home/sibwang/contrastive-unpaired-translation/scripts_on_cluster/20250711_parameter_sweep/template.run")
data_output_dir = Path("/scratch/sibwang/contrastive-unpaired-translation/bulk_data/style_transfer_cut/20250711_parameter_sweep/")
script_output_dir = Path("/home/sibwang/contrastive-unpaired-translation/scripts_on_cluster/20250711_parameter_sweep/batch_scripts/")
checkpoint_dir = data_output_dir / "checkpoints"
log_dir = data_output_dir / "logs"
wandb_dir = data_output_dir / "wandb"
checkpoint_dir.mkdir(parents=True, exist_ok=True)
log_dir.mkdir(parents=True, exist_ok=True)
wandb_dir.mkdir(parents=True, exist_ok=True)
script_output_dir.mkdir(parents=True, exist_ok=True)

if list(data_output_dir.glob("*.run")):
    raise RuntimeError(
        "Batch scripts already found in the output directory. Empty them manually to "
        "be explicit about which machine-generated scipts to run."
    )

with open(template_path) as f:
    template_str = f.read()

counter = 0
for ngf in [32, 64]:
    for netG in ["resnet_6blocks", "resnet_9blocks"]:
        for batch_size in [1, 4]:
            for lambda_GAN in [1.0, 3.0]:
                run_name = (
                    f"ngf{ngf}_netG{netG}_batsize{batch_size}_lambGAN{lambda_GAN}"
                )
                script_str = template_str \
                    .replace("<<<NGF>>>", str(ngf)) \
                    .replace("<<<NETG>>>", netG) \
                    .replace("<<<BATCH_SIZE>>>", str(batch_size)) \
                    .replace("<<<LAMBDA_GAN>>>", str(lambda_GAN)) \
                    .replace("<<<CHECKPOINT_DIR>>>", str(checkpoint_dir / run_name)) \
                    .replace("<<<LOG_DIR>>>", str(log_dir / run_name)) \
                    .replace("<<<WANDB_DIR>>>", str(wandb_dir / run_name)) \
                    .replace("<<<OUTPUT_FILE>>>", str(script_output_dir.parent / f"outputs/{run_name}.out")) \
                    .replace("<<<NAME>>>", run_name)

                filename = f"{run_name}.run"
                script_path = script_output_dir / filename
                with open(script_path, "w") as f:
                    f.write(script_str)
                print(f"Script written to {script_path}")
                counter += 1

print(f"{counter} scripts written to {data_output_dir}")