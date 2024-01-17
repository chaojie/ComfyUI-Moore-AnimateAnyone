import os
import random
from datetime import datetime

import gradio as gr
import numpy as np
import torch
from diffusers import AutoencoderKL, DDIMScheduler
from einops import repeat
from omegaconf import OmegaConf
from PIL import Image
import torchvision
from torchvision import transforms
from transformers import CLIPVisionModelWithProjection
from einops import rearrange
import av

import sys
current_path = os.path.abspath(os.path.dirname(__file__))
sys.path.append(current_path)

from src.models.pose_guider import PoseGuider
from src.models.unet_2d_condition import UNet2DConditionModel
from src.models.unet_3d import UNet3DConditionModel
from src.pipelines.pipeline_pose2vid_long import Pose2VideoPipeline
from src.utils.util import get_fps, read_frames, save_videos_grid

import folder_paths
comfy_path = os.path.dirname(folder_paths.__file__)
pretrained_weights_path=f'{comfy_path}/custom_nodes/ComfyUI-Moore-AnimateAnyone/pretrained_weights'
pretrained_weights=os.listdir(pretrained_weights_path)
inference_config_path=f'{comfy_path}/custom_nodes/ComfyUI-Moore-AnimateAnyone/configs/inference/inference_v2.yaml'
infer_config = OmegaConf.load(inference_config_path)

class LoadAnimateAnyone_VAE:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "path": (pretrained_weights, {"default": "sd-vae-ft-mse"}),
            },
        }

    RETURN_TYPES = ("VAE",)
    FUNCTION = "run_inference"
    CATEGORY = "AnimateAnyone"

    def run_inference(self,path):
        global pretrained_weights_path
        path = os.path.join(pretrained_weights_path, path)
        vae = AutoencoderKL.from_pretrained(
                path
            ).to("cuda", dtype=torch.float16)
        return (vae,)

class LoadAnimateAnyone_Image_Encoder:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "path": (pretrained_weights, {"default": "image_encoder"}),
            },
        }

    RETURN_TYPES = ("CLIPVision",)
    FUNCTION = "run_inference"
    CATEGORY = "AnimateAnyone"

    def run_inference(self,path):
        global pretrained_weights_path
        path = os.path.join(pretrained_weights_path, path)
        image_enc = CLIPVisionModelWithProjection.from_pretrained(
                path
            ).to(dtype=torch.float16, device="cuda")
        return (image_enc,)

class LoadAnimateAnyone_Reference_Unet:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "pretrained_base_model_path": (pretrained_weights, {"default": "stable-diffusion-v1-5"}),
                "reference_unet_path": (pretrained_weights, {"default": "reference_unet.pth"}),
            },
        }

    RETURN_TYPES = ("UNet2DConditionModel",)
    FUNCTION = "run_inference"
    CATEGORY = "AnimateAnyone"

    def run_inference(self,pretrained_base_model_path,reference_unet_path):
        global pretrained_weights_path
        pretrained_base_model_path = os.path.join(pretrained_weights_path, pretrained_base_model_path)
        reference_unet_path = os.path.join(pretrained_weights_path, reference_unet_path)
        reference_unet = UNet2DConditionModel.from_pretrained(
                pretrained_base_model_path,
                subfolder="unet",
            ).to(dtype=torch.float16, device="cuda")
        reference_unet.load_state_dict(
            torch.load(reference_unet_path, map_location="cpu"),
        )
        return (reference_unet,)

class LoadAnimateAnyone_Denoising_Unet:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "pretrained_base_model_path": (pretrained_weights, {"default": "stable-diffusion-v1-5"}),
                "motion_module_path": (pretrained_weights, {"default": "motion_module.pth"}),
                "denoising_unet_path": (pretrained_weights, {"default": "denoising_unet.pth"}),
            },
        }

    RETURN_TYPES = ("UNet3DConditionModel",)
    FUNCTION = "run_inference"
    CATEGORY = "AnimateAnyone"

    def run_inference(self,pretrained_base_model_path,motion_module_path,denoising_unet_path):
        global pretrained_weights_path, infer_config
        pretrained_base_model_path = os.path.join(pretrained_weights_path, pretrained_base_model_path)
        motion_module_path = os.path.join(pretrained_weights_path, motion_module_path)
        denoising_unet_path = os.path.join(pretrained_weights_path, denoising_unet_path)
        denoising_unet = UNet3DConditionModel.from_pretrained_2d(
                pretrained_base_model_path,
                motion_module_path,
                subfolder="unet",
                unet_additional_kwargs=infer_config.unet_additional_kwargs,
            ).to(dtype=torch.float16, device="cuda")
        denoising_unet.load_state_dict(
                torch.load(denoising_unet_path, map_location="cpu"),
                strict=False,
            )
        return (denoising_unet,)

class LoadAnimateAnyone_Pose_Guider:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "pose_guider_path": (pretrained_weights, {"default": "pose_guider.pth"}),
            },
        }

    RETURN_TYPES = ("PoseGuider",)
    FUNCTION = "run_inference"
    CATEGORY = "AnimateAnyone"

    def run_inference(self,pose_guider_path):
        global pretrained_weights_path, infer_config
        pose_guider_path = os.path.join(pretrained_weights_path, pose_guider_path)
        pose_guider = PoseGuider(320, block_out_channels=(16, 32, 96, 256)).to(
                dtype=torch.float16, device="cuda"
            )
        pose_guider.load_state_dict(
                torch.load(pose_guider_path, map_location="cpu"),
            )
        return (pose_guider,)


class AnimateAnyonePipelineLoader:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "vae": ("VAE",),
                "image_enc": ("CLIPVision",),
                "reference_unet": ("UNet2DConditionModel",),
                "denoising_unet": ("UNet3DConditionModel",),
                "pose_guider": ("PoseGuider",),
            }
        }
        
    RETURN_TYPES = ("Pose2VideoPipeline",)
    RETURN_NAMES = ("pipe",)
    FUNCTION = "load_checkpoint"
    CATEGORY = "AnimateAnyone"

    def load_checkpoint(self, vae, image_enc, reference_unet, denoising_unet, pose_guider):
        global infer_config
        sched_kwargs = OmegaConf.to_container(infer_config.noise_scheduler_kwargs)
        scheduler = DDIMScheduler(**sched_kwargs)
        pipe = Pose2VideoPipeline(
            vae=vae,
            image_encoder=image_enc,
            reference_unet=reference_unet,
            denoising_unet=denoising_unet,
            pose_guider=pose_guider,
            scheduler=scheduler,
        )
        pipe = pipe.to("cuda", dtype=torch.float16)
        
        return (pipe,)

class AnimateAnyoneSimple:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "pipe": ("Pose2VideoPipeline",),
                "ref_image": ("IMAGE",),
                "pose_images": ("IMAGE",),
                "width": ("INT", {"default": 512}),
                "height": ("INT", {"default": 768}),
                "length": ("INT", {"default": 24}),
                "steps": ("INT", {"default": 25}),
                "cfg": ("FLOAT", {"default": 3.5}),
                "seed": ("INT", {"default": 123}),
            },
        }

    RETURN_TYPES = ("IMAGE","IMAGE",)
    RETURN_NAMES = ("result","result_compare",)
    FUNCTION = "run_inference"
    CATEGORY = "motionctrl"

    def run_inference(self,pipe,ref_image,pose_images,width,height,length,steps,cfg,seed):
        generator = torch.manual_seed(seed)

        ref_image = 255.0 * ref_image[0].cpu().numpy()
        ref_image = Image.fromarray(np.clip(ref_image, 0, 255).astype(np.uint8))

        pose_list = []
        pose_tensor_list = []
        pose_transform = transforms.Compose(
            [transforms.Resize((height, width)), transforms.ToTensor()]
        )
        for pose_image_pil in pose_images[:length]:
            pose_image_pil = 255.0 * pose_image_pil.cpu().numpy()
            pose_image_pil = Image.fromarray(np.clip(pose_image_pil, 0, 255).astype(np.uint8))
            
            pose_list.append(pose_image_pil)
            pose_tensor_list.append(pose_transform(pose_image_pil))
        #pose_list[0].save('pose_image_pil.png')
        src_fps=8

        video = pipe(
            ref_image,
            pose_list,
            width=width,
            height=height,
            video_length=length,
            num_inference_steps=steps,
            guidance_scale=cfg,
            generator=generator,
        ).videos
        
        ref_image_tensor = pose_transform(ref_image)  # (c, h, w)
        ref_image_tensor = ref_image_tensor.unsqueeze(1).unsqueeze(0)  # (1, c, 1, h, w)
        ref_image_tensor = repeat(
            ref_image_tensor, "b c f h w -> b c (repeat f) h w", repeat=length
        )
        pose_tensor = torch.stack(pose_tensor_list, dim=0)  # (f, c, h, w)
        pose_tensor = pose_tensor.transpose(0, 1)
        pose_tensor = pose_tensor.unsqueeze(0)
        video_compare = torch.cat([ref_image_tensor, pose_tensor, video], dim=0)

        video = rearrange(video, "b c t h w -> t b c h w")
        height, width = video.shape[-2:]
        outframes = []

        for x in video:
            x = torchvision.utils.make_grid(x, nrow=1)  # (c h w)
            x = x.transpose(0, 1).transpose(1, 2).squeeze(-1)  # (h w c)
            x = (x * 255).numpy().astype(np.uint8)
            image = Image.fromarray(x)
            image_tensor_out = torch.tensor(np.array(image).astype(np.float32) / 255.0)  # Convert back to CxHxW
            image_tensor_out = torch.unsqueeze(image_tensor_out, 0)
            outframes.append(image_tensor_out)

        video_compare = rearrange(video_compare, "b c t h w -> t b c h w")
        height, width = video_compare.shape[-2:]
        outs = []

        for x in video_compare:
            x = torchvision.utils.make_grid(x, nrow=1)  # (c h w)
            x = x.transpose(0, 1).transpose(1, 2).squeeze(-1)  # (h w c)
            x = (x * 255).numpy().astype(np.uint8)
            image = Image.fromarray(x)
            image_tensor_out = torch.tensor(np.array(image).astype(np.float32) / 255.0)  # Convert back to CxHxW
            image_tensor_out = torch.unsqueeze(image_tensor_out, 0)
            outs.append(image_tensor_out)

        torch.cuda.empty_cache()

        return (torch.cat(tuple(outframes), dim=0),torch.cat(tuple(outs), dim=0),)


NODE_CLASS_MAPPINGS = {
    "Moore-AnimateAnyone Simple":AnimateAnyoneSimple,
    "Moore-AnimateAnyone Pipeline Loader":AnimateAnyonePipelineLoader,
    "Moore-AnimateAnyone Pose Guider":LoadAnimateAnyone_Pose_Guider,
    "Moore-AnimateAnyone Denoising Unet":LoadAnimateAnyone_Denoising_Unet,
    "Moore-AnimateAnyone Reference Unet":LoadAnimateAnyone_Reference_Unet,
    "Moore-AnimateAnyone Image Encoder":LoadAnimateAnyone_Image_Encoder,
    "Moore-AnimateAnyone VAE":LoadAnimateAnyone_VAE,
}
