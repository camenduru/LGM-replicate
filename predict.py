import os
from cog import BasePredictor, Input, Path
from typing import List
import sys
sys.path.append('/content/LGM-hf')
os.chdir('/content/LGM-hf')

import tyro
import imageio
import numpy as np
import tqdm
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms.functional as TF
from safetensors.torch import load_file
import rembg
from PIL import Image

import kiui
from kiui.op import recenter
from kiui.cam import orbit_camera

from core.options import AllConfigs, Options
from core.models import LGM
from mvdream.pipeline_mvdream import MVDreamPipeline

IMAGENET_DEFAULT_MEAN = (0.485, 0.456, 0.406)
IMAGENET_DEFAULT_STD = (0.229, 0.224, 0.225)
GRADIO_VIDEO_PATH = 'gradio_output.mp4'
GRADIO_PLY_PATH = 'gradio_output.ply'

# inference function
def inference(input_image, prompt, prompt_neg='', input_elevation=0, input_num_steps=30, input_seed=42, opt=None, device=None, model=None, proj_matrix=None, pipe_text=None, pipe_image=None, bg_remover=None):
    # seed
    kiui.seed_everything(input_seed)

    os.makedirs(opt.workspace, exist_ok=True)
    output_video_path = os.path.join(opt.workspace, GRADIO_VIDEO_PATH)
    output_ply_path = os.path.join(opt.workspace, GRADIO_PLY_PATH)

    # text-conditioned
    if input_image is None:
        mv_image_uint8 = pipe_text(prompt, negative_prompt=prompt_neg, num_inference_steps=input_num_steps, guidance_scale=7.5, elevation=input_elevation)
        mv_image_uint8 = (mv_image_uint8 * 255).astype(np.uint8)
        # bg removal
        mv_image = []
        for i in range(4):
            image = rembg.remove(mv_image_uint8[i], session=bg_remover) # [H, W, 4]
            # to white bg
            image = image.astype(np.float32) / 255
            image = recenter(image, image[..., 0] > 0, border_ratio=0.2)
            image = image[..., :3] * image[..., -1:] + (1 - image[..., -1:])
            mv_image.append(image)
    # image-conditioned (may also input text, but no text usually works too)
    else:
        input_image = Image.open(input_image)
        input_image = np.array(input_image) # uint8
        # bg removal
        carved_image = rembg.remove(input_image, session=bg_remover) # [H, W, 4]
        mask = carved_image[..., -1] > 0
        image = recenter(carved_image, mask, border_ratio=0.2)
        image = image.astype(np.float32) / 255.0
        image = image[..., :3] * image[..., 3:4] + (1 - image[..., 3:4])
        mv_image = pipe_image(prompt, image, negative_prompt=prompt_neg, num_inference_steps=input_num_steps, guidance_scale=5.0,  elevation=input_elevation)
        
    mv_image_grid = np.concatenate([
        np.concatenate([mv_image[1], mv_image[2]], axis=1),
        np.concatenate([mv_image[3], mv_image[0]], axis=1),
    ], axis=0)

    # generate gaussians
    input_image = np.stack([mv_image[1], mv_image[2], mv_image[3], mv_image[0]], axis=0) # [4, 256, 256, 3], float32
    input_image = torch.from_numpy(input_image).permute(0, 3, 1, 2).float().to(device) # [4, 3, 256, 256]
    input_image = F.interpolate(input_image, size=(opt.input_size, opt.input_size), mode='bilinear', align_corners=False)
    input_image = TF.normalize(input_image, IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD)

    rays_embeddings = model.prepare_default_rays(device, elevation=input_elevation)
    input_image = torch.cat([input_image, rays_embeddings], dim=1).unsqueeze(0) # [1, 4, 9, H, W]

    with torch.no_grad():
        with torch.autocast(device_type='cuda', dtype=torch.float16):
            # generate gaussians
            gaussians = model.forward_gaussians(input_image)
        
        # save gaussians
        model.gs.save_ply(gaussians, output_ply_path)
        
        # render 360 video 
        images = []
        elevation = 0
        if opt.fancy_video:
            azimuth = np.arange(0, 720, 4, dtype=np.int32)
            for azi in tqdm.tqdm(azimuth):
                
                cam_poses = torch.from_numpy(orbit_camera(elevation, azi, radius=opt.cam_radius, opengl=True)).unsqueeze(0).to(device)

                cam_poses[:, :3, 1:3] *= -1 # invert up & forward direction
                
                # cameras needed by gaussian rasterizer
                cam_view = torch.inverse(cam_poses).transpose(1, 2) # [V, 4, 4]
                cam_view_proj = cam_view @ proj_matrix # [V, 4, 4]
                cam_pos = - cam_poses[:, :3, 3] # [V, 3]

                scale = min(azi / 360, 1)

                image = model.gs.render(gaussians, cam_view.unsqueeze(0), cam_view_proj.unsqueeze(0), cam_pos.unsqueeze(0), scale_modifier=scale)['image']
                images.append((image.squeeze(1).permute(0,2,3,1).contiguous().float().cpu().numpy() * 255).astype(np.uint8))
        else:
            azimuth = np.arange(0, 360, 2, dtype=np.int32)
            for azi in tqdm.tqdm(azimuth):
                
                cam_poses = torch.from_numpy(orbit_camera(elevation, azi, radius=opt.cam_radius, opengl=True)).unsqueeze(0).to(device)

                cam_poses[:, :3, 1:3] *= -1 # invert up & forward direction
                
                # cameras needed by gaussian rasterizer
                cam_view = torch.inverse(cam_poses).transpose(1, 2) # [V, 4, 4]
                cam_view_proj = cam_view @ proj_matrix # [V, 4, 4]
                cam_pos = - cam_poses[:, :3, 3] # [V, 3]

                image = model.gs.render(gaussians, cam_view.unsqueeze(0), cam_view_proj.unsqueeze(0), cam_pos.unsqueeze(0), scale_modifier=1)['image']
                images.append((image.squeeze(1).permute(0,2,3,1).contiguous().float().cpu().numpy() * 255).astype(np.uint8))

        images = np.concatenate(images, axis=0)
        imageio.mimwrite(output_video_path, images, fps=30)

    return mv_image_grid, output_video_path, output_ply_path


class Predictor(BasePredictor):
    def setup(self) -> None:        
        ckpt_path = "/content/LGM-hf/model/model.safetensors"

        # self.opt = tyro.cli(AllConfigs)
        self.opt = Options(
            input_size=256,
            up_channels=(1024, 1024, 512, 256, 128), # one more decoder
            up_attention=(True, True, True, False, False),
            splat_size=128,
            output_size=512, # render & supervise Gaussians at a higher resolution.
            batch_size=8,
            num_views=8,
            gradient_accumulation_steps=1,
            mixed_precision='bf16',
            resume=ckpt_path,
        )

        # self.model
        self.model = LGM(self.opt)

        # resume pretrained checkpoint
        if self.opt.resume is not None:
            if self.opt.resume.endswith('safetensors'):
                ckpt = load_file(self.opt.resume, device='cpu')
            else:
                ckpt = torch.load(self.opt.resume, map_location='cpu')
            self.model.load_state_dict(ckpt, strict=False)
            print(f'[INFO] Loaded checkpoint from {self.opt.resume}')
        else:
            print(f'[WARN] self.model randomly initialized, are you sure?')

        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = self.model.half().to(self.device)
        self.model.eval()

        tan_half_fov = np.tan(0.5 * np.deg2rad(self.opt.fovy))
        self.proj_matrix = torch.zeros(4, 4, dtype=torch.float32, device=self.device)
        self.proj_matrix[0, 0] = 1 / tan_half_fov
        self.proj_matrix[1, 1] = 1 / tan_half_fov
        self.proj_matrix[2, 2] = (self.opt.zfar + self.opt.znear) / (self.opt.zfar - self.opt.znear)
        self.proj_matrix[3, 2] = - (self.opt.zfar * self.opt.znear) / (self.opt.zfar - self.opt.znear)
        self.proj_matrix[2, 3] = 1

        # load dreams
        self.pipe_text = MVDreamPipeline.from_pretrained(
            'ashawkey/mvdream-sd2.1-diffusers', # remote weights
            torch_dtype=torch.float16,
            trust_remote_code=True,
            # local_files_only=True,
        )
        self.pipe_text = self.pipe_text.to(self.device)

        self.pipe_image = MVDreamPipeline.from_pretrained(
            "ashawkey/imagedream-ipmv-diffusers", # remote weights
            torch_dtype=torch.float16,
            trust_remote_code=True,
            # local_files_only=True,
        )
        self.pipe_image = self.pipe_image.to(self.device)

        # load rembg
        self.bg_remover = rembg.new_session()
    def predict(
        self,
        input_image: Path = Input(description="Input Image"),
        prompt: str = Input(default="a songbird"),
        negative_prompt: str = Input(default="ugly, blurry, pixelated obscure, unnatural colors, poor lighting, dull, unclear, cropped, lowres, low quality, artifacts, duplicate"),
        seed: int = Input(default=42),
    ) -> List[Path]:
        output_video = inference(input_image, prompt, negative_prompt, 0, 30, seed, self.opt, self.device, self.model, self.proj_matrix, self.pipe_text, self.pipe_image, self.bg_remover)
        return [Path(output_video[1]), Path(output_video[2])]
