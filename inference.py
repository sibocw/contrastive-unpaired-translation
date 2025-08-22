import torch
import torchvision
from pathlib import Path
from PIL import Image
from models.cut_model import CUTModel
from options.option_stats import OptionsWrapper


class Options:
    def __init__(
        self,
        input_nc: int,
        output_nc: int,
        ngf: int,
        netG: str,
        image_side_length: int,
        nce_layers: list[int],
    ):
        # Variables listed as "don't-care" are not actually used in the inference
        # pipeline, but they are needed for initializing the model instances
        self.isTrain = False
        self.gpu_ids = [0]
        self.checkpoints_dir = ""  # don't-care
        self.name = ""  # don't-care
        self.preprocess = None  # don't-care
        self.nce_layers = ",".join([str(x) for x in nce_layers])  # don't-care
        self.nce_idt = True  # don't-care
        self.input_nc = input_nc
        self.output_nc = output_nc
        self.ngf = ngf
        self.netG = netG
        self.normG = "instance"  # don't-care
        self.no_dropout = True  # don't-care
        self.init_type = None  # don't-care
        self.init_gain = None  # don't-care
        self.no_antialias = False  # don't-care
        self.no_antialias_up = False  # don't-care
        self.load_size = image_side_length
        self.crop_size = image_side_length
        self.stylegan2_G_num_downsampling = 1
        self.netF = "mlp_sample"  # don't-care
        self.netF_nc = None


class InferencePipeline:
    def __init__(
        self,
        netG_ckpt_path: Path,
        *,
        input_nc: int,
        output_nc: int,
        ngf: int,
        netG: str,
        image_side_length: int,
        nce_layers: list[int],
        device: str | torch.device = "cuda",
    ):
        _opt = Options(input_nc, output_nc, ngf, netG, image_side_length, nce_layers)
        self.opt = OptionsWrapper(_opt)
        self.model = CUTModel(self.opt)
        self.is_model_initialized = False
        self.netG_ckpt_path = netG_ckpt_path
        self.device = device
        self.input_transforms = torchvision.transforms.Compose(
            [
                torchvision.transforms.ToTensor(),
                torchvision.transforms.Resize((image_side_length, image_side_length)),
                torchvision.transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
            ]
        )

    def infer(self, input_images):
        if not self.is_model_initialized:
            self._initialize_model(input_images.to(self.device))

        input_device = input_images.device
        with torch.no_grad():
            output_images = self.model.netG(input_images.to(self.device))
            output_images = output_images.to(input_device)
            output_images = (output_images + 1.0) / 2.0
            return output_images

    def _initialize_model(self, input_images):
        # Data-dependent initialization
        # When forward is called, the forward method of the PatchSampleF
        # layers will call create_mlp based on feature shape
        self.model.netG(input_images)

        # Load model weights
        net = getattr(self.model, "netG")
        if isinstance(net, torch.nn.DataParallel):
            net = net.module
        state_dict = torch.load(self.netG_ckpt_path, map_location=str(self.device))
        if hasattr(state_dict, "_metadata"):
            del state_dict._metadata
        net.load_state_dict(state_dict)

        # Print network architecture
        self.model.print_networks(verbose=True)

        # Parallelize (only useful from multi-GPU case)
        parallel_net = torch.nn.DataParallel(net, device_ids=self.opt.gpu_ids)
        setattr(self.model, "netG", parallel_net)


if __name__ == "__main__":
    netG_ckpt_path = Path(
        "/home/sibwang/Projects/contrastive-unpaired-translation/from_scitas/ngf32_netGstylegan2_batsize4_lambGAN1.0/latest_net_G.pth"
    )
    inference_pipeline = InferencePipeline(
        netG_ckpt_path,
        input_nc=3,
        output_nc=3,
        ngf=32,
        netG="stylegan2",
        image_side_length=256,
        nce_layers=[0, 4, 8, 12, 16],
    )

    input_image_dir = Path(
        "/home/sibwang/Projects/biomechpose/bulk_data/style_transfer_cut/datasets/spotlight202506_to_aymanns2022/testA"
    )
    output_image_dir = Path("./output_images")
    output_image_dir.mkdir(exist_ok=True)
    num_images_to_process = 10
    image_paths = sorted(list(input_image_dir.glob("*.jpg")))
    images = []
    for path in image_paths[:num_images_to_process]:
        img = Image.open(path).convert("RGB")
        img = inference_pipeline.input_transforms(img)
        images.append(img)
    images = torch.stack(images)

    print("Input images shape:", images.shape)
    output_images = inference_pipeline.infer(images)
    print("Output images shape:", output_images.shape)

    for i in range(output_images.shape[0]):
        img = output_images[i]
        img = torchvision.transforms.ToPILImage()(img)
        img.save(output_image_dir / f"output_{i:05d}.jpg")

    inference_pipeline.opt.print_summary()
