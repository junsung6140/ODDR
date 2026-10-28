"""ODD / ODDR models.

ODD (One-step Deshadow Diffusion) adapts the Stable Diffusion 2.1-base UNet with LoRA
(ODD-LoRA) and adds convolutional skip connections to the VAE (ODD-Conv). The UNet is used
as a time- and text-free one-step network at t = 999.

ODDR adds a second LoRA (ODDR-LoRA, adapter name "stage2") on top of the frozen ODD and
fine-tunes it with ShadowReward.

Checkpoint format (`.pkl`, saved by `ODDTrainModel.save_model`):
    state_dict_unet : ODD-LoRA ("default_encoder", "default_decoder", "default_others"),
                      UNet conv_in, and for ODDR also ODDR-LoRA ("stage2")
    state_dict_vae  : ODD-Conv weights (decoder.skip_conv_*)
    unet_lora_{encoder,decoder,others}_modules, rank_unet : LoRA configuration

This file is derived from OSEDiff (https://github.com/cswry/OSEDiff, Apache-2.0).
"""
import types

import numpy as np
import torch
import torch.nn as nn
from diffusers import DDPMScheduler
from diffusers.models.attention import BasicTransformerBlock
from diffusers.models.resnet import ResnetBlock2D
from diffusers.models.transformer_2d import Transformer2DModel
from diffusers.models.unet_2d_blocks import (CrossAttnDownBlock2D, CrossAttnUpBlock2D, DownBlock2D,
                                             UNetMidBlock2DCrossAttn, UpBlock2D)
from peft import LoraConfig

from models.autoencoder_kl import AutoencoderKL
from models.unet_2d_condition import UNet2DConditionModel
from models.unet_forward import (MyCrossAttnDownBlock2D_SD_forward, MyCrossAttnUpBlock2D_SD_forward,
                                 MyDownBlock2D_SD_forward, MyResnetBlock2D_SD_forward,
                                 MyTransformer2DModel_SD_forward, MyUNet2DConditionModel_SD_forward,
                                 MyUNetMidBlock2DCrossAttn_SD_forward, MyUpBlock2D_SD_forward)

ODD_ADAPTERS = ["default_encoder", "default_decoder", "default_others"]
ODDR_ADAPTER = "stage2"


def initialize_vae(args):
    vae = AutoencoderKL.from_pretrained_orig(args.pretrained_model_name_or_path, subfolder="vae")
    vae.requires_grad_(False)
    vae.train()

    # Module list kept only for checkpoint format compatibility (no LoRA is added to the VAE).
    l_target_modules_encoder = []
    l_grep = ["conv1", "conv2", "conv_in", "conv_shortcut", "conv", "conv_out", "to_k", "to_q", "to_v", "to_out.0"]
    for n, p in vae.named_parameters():
        if "bias" in n or "norm" in n or "skip" in n:
            continue
        for pattern in l_grep:
            if pattern in n and ("decoder" in n):
                l_target_modules_encoder.append(n.replace(".weight", ""))
            elif ('quant_conv' in n) and ('post_quant_conv' not in n):
                l_target_modules_encoder.append(n.replace(".weight", ""))
    return vae, l_target_modules_encoder


def initialize_unet(args, with_lora=True):
    """Load the SD UNet and turn it into a time- and text-free network.

    The time embedding projections and the cross-attention layers are removed, and the block
    forward functions are replaced (see models/unet_forward.py).
    """
    unet = UNet2DConditionModel.from_pretrained(args.pretrained_model_name_or_path, subfolder="unet")

    def ResnetBlock2D_remove_time_emb_proj(module):
        if isinstance(module, ResnetBlock2D):
            del module.time_emb_proj
    unet.apply(ResnetBlock2D_remove_time_emb_proj)

    def BasicTransformerBlock_remove_cross_attn(module):
        if isinstance(module, BasicTransformerBlock):
            del module.attn2, module.norm2
    unet.apply(BasicTransformerBlock_remove_cross_attn)

    def set_inplace_to_true(module):
        if isinstance(module, nn.Dropout) or isinstance(module, nn.SiLU):
            module.inplace = True
    unet.apply(set_inplace_to_true)

    def replace_forward_methods(module):
        if isinstance(module, CrossAttnDownBlock2D):
            module.forward = types.MethodType(MyCrossAttnDownBlock2D_SD_forward, module)
        elif isinstance(module, DownBlock2D):
            module.forward = types.MethodType(MyDownBlock2D_SD_forward, module)
        elif isinstance(module, UNetMidBlock2DCrossAttn):
            module.forward = types.MethodType(MyUNetMidBlock2DCrossAttn_SD_forward, module)
        elif isinstance(module, UpBlock2D):
            module.forward = types.MethodType(MyUpBlock2D_SD_forward, module)
        elif isinstance(module, CrossAttnUpBlock2D):
            module.forward = types.MethodType(MyCrossAttnUpBlock2D_SD_forward, module)
        elif isinstance(module, ResnetBlock2D):
            module.forward = types.MethodType(MyResnetBlock2D_SD_forward, module)
        elif isinstance(module, Transformer2DModel):
            module.forward = types.MethodType(MyTransformer2DModel_SD_forward, module)
    unet.apply(replace_forward_methods)
    unet.forward = types.MethodType(MyUNet2DConditionModel_SD_forward, unet)
    unet.body = nn.Sequential(
        *unet.down_blocks,
        unet.mid_block,
        *unet.up_blocks,
        unet.conv_norm_out,
        unet.conv_act,
        unet.conv_out,
    )
    unet.requires_grad_(False)

    # LoRA target modules (ODD-LoRA): split into encoder / decoder / other parts of the UNet.
    l_target_modules_encoder, l_target_modules_decoder, l_modules_others = [], [], []
    if with_lora:
        unet.train()
        l_grep = ["to_k", "to_q", "to_v", "to_out.0", "conv", "conv1", "conv2", "conv_in", "conv_shortcut",
                  "conv_out", "proj_out", "proj_in", "ff.net.2", "ff.net.0.proj"]
        for n, p in unet.named_parameters():
            if "bias" in n or "norm" in n:
                continue
            for pattern in l_grep:
                if pattern in n and ("down_blocks" in n or "conv_in" in n):
                    l_target_modules_encoder.append(n.replace(".weight", ""))
                    break
                elif pattern in n and ("up_blocks" in n or "conv_out" in n):
                    l_target_modules_decoder.append(n.replace(".weight", ""))
                    break
                elif pattern in n:
                    l_modules_others.append(n.replace(".weight", ""))
                    break

        lora_conf_encoder = LoraConfig(r=args.lora_rank, init_lora_weights="gaussian", target_modules=l_target_modules_encoder)
        lora_conf_decoder = LoraConfig(r=args.lora_rank, init_lora_weights="gaussian", target_modules=l_target_modules_decoder)
        lora_conf_others = LoraConfig(r=args.lora_rank, init_lora_weights="gaussian", target_modules=l_modules_others)
        unet.add_adapter(lora_conf_encoder, adapter_name="default_encoder")
        unet.add_adapter(lora_conf_decoder, adapter_name="default_decoder")
        unet.add_adapter(lora_conf_others, adapter_name="default_others")

    return unet, l_target_modules_encoder, l_target_modules_decoder, l_modules_others


def _copy_strict(module, state_dict, select, name):
    """Copy `state_dict` into the parameters of `module` whose names satisfy `select`.

    Raises an error if a selected parameter is missing from the checkpoint or if the
    checkpoint has keys that do not match any selected parameter.
    """
    # state_dict() keeps every alias of a shared parameter (e.g. `unet.body` reuses the
    # UNet blocks), which matches how the checkpoints were saved.
    params = {n: p for n, p in module.state_dict(keep_vars=True).items() if select(n)}
    missing = sorted(set(params) - set(state_dict))
    unexpected = sorted(set(state_dict) - set(params))
    if missing or unexpected:
        raise RuntimeError(f"[{name}] checkpoint mismatch: {len(missing)} missing keys {missing[:5]}, "
                           f"{len(unexpected)} unexpected keys {unexpected[:5]}")
    for n, p in params.items():
        p.data.copy_(state_dict[n])


class ODDTrainModel(torch.nn.Module):
    """Trainable one-step model used for ODD training (stage 1) and ODDR fine-tuning (stage 2)."""

    def __init__(self, args):
        super().__init__()
        self.noise_scheduler = DDPMScheduler.from_pretrained(args.pretrained_model_name_or_path, subfolder="scheduler")
        self.noise_scheduler.set_timesteps(1, device="cuda")
        self.noise_scheduler.alphas_cumprod = self.noise_scheduler.alphas_cumprod.cuda()
        self.args = args

        self.vae, self.lora_vae_modules_encoder = initialize_vae(self.args)
        self.unet, self.lora_unet_modules_encoder, self.lora_unet_modules_decoder, self.lora_unet_others = \
            initialize_unet(self.args)
        self.lora_rank_vae = self.args.lora_rank
        self.lora_rank_unet = self.args.lora_rank

        self.unet.to("cuda")
        self.vae.to("cuda")
        self.timesteps = torch.tensor([999], device="cuda").long()

    def load_ckpt(self, model):
        """Load an ODD checkpoint (ODD-LoRA, conv_in and ODD-Conv)."""
        _copy_strict(self.unet, model["state_dict_unet"], lambda n: "lora" in n or "conv_in" in n, "unet")
        self.unet.set_adapter(ODD_ADAPTERS)
        _copy_strict(self.vae, model["state_dict_vae"], lambda n: "skip" in n, "vae")

    def set_train_odd(self):
        """Stage 1: train ODD-LoRA, UNet conv_in and ODD-Conv."""
        self.unet.train()
        self.vae.train()
        for n, _p in self.unet.named_parameters():
            if "lora" in n:
                _p.requires_grad = True
        self.unet.conv_in.requires_grad_(True)
        self.vae.decoder.skip_conv_1.requires_grad_(True)
        self.vae.decoder.skip_conv_2.requires_grad_(True)
        self.vae.decoder.skip_conv_3.requires_grad_(True)
        self.vae.decoder.skip_conv_4.requires_grad_(True)

    def add_oddr_lora(self):
        """Attach ODDR-LoRA (same target modules as ODD-LoRA). lora_B is zero-initialized."""
        lora_conf_stage2 = LoraConfig(
            r=self.lora_rank_unet,
            init_lora_weights="gaussian",
            target_modules=self.lora_unet_modules_decoder + self.lora_unet_modules_encoder + self.lora_unet_others,
        )
        self.unet.add_adapter(lora_conf_stage2, adapter_name=ODDR_ADAPTER)
        self.unet.set_adapter([ODDR_ADAPTER])

    def set_train_oddr(self):
        """Stage 2: only ODDR-LoRA is trainable; ODD-LoRA, conv_in and ODD-Conv stay frozen."""
        self.add_oddr_lora()
        self.unet.train()
        for n, p in self.unet.named_parameters():
            p.requires_grad = ODDR_ADAPTER in n

    def forward(self, c_t, batch=None, args=None):
        with torch.no_grad():
            encoded_control = self.vae.encode(c_t).latent_dist.sample() * self.vae.config.scaling_factor
            residuals = self.vae.encoder.residuals

        model_pred = self.unet(encoded_control)
        x_denoised = self.noise_scheduler.step(model_pred, self.timesteps, encoded_control, return_dict=True).prev_sample
        output_image = (self.vae.decode(x_denoised / self.vae.config.scaling_factor, residuals=residuals).sample).clamp(-1, 1)
        return output_image

    def save_model(self, outf):
        sd = {}
        sd["vae_lora_encoder_modules"] = self.lora_vae_modules_encoder
        sd["unet_lora_encoder_modules"], sd["unet_lora_decoder_modules"], sd["unet_lora_others_modules"] = \
            self.lora_unet_modules_encoder, self.lora_unet_modules_decoder, self.lora_unet_others
        sd["rank_vae"] = self.lora_rank_vae
        sd["rank_unet"] = self.lora_rank_unet
        sd["state_dict_unet"] = {k: v for k, v in self.unet.state_dict().items() if "lora" in k or "conv_in" in k}
        sd["state_dict_vae"] = {k: v for k, v in self.vae.state_dict().items() if "skip" in k}
        torch.save(sd, outf)


class ODDInference(torch.nn.Module):
    """Inference model. Loads ODD.pkl (ODD-LoRA only) or ODDR.pkl (ODD-LoRA + ODDR-LoRA)."""

    def __init__(self, args):
        super().__init__()
        self.args = args
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.noise_scheduler = DDPMScheduler.from_pretrained(args.pretrained_model_name_or_path, subfolder="scheduler")
        self.noise_scheduler.set_timesteps(1, device="cuda")
        self.vae = AutoencoderKL.from_pretrained_orig(self.args.pretrained_model_name_or_path, subfolder="vae")
        self.unet, _, _, _ = initialize_unet(self.args, with_lora=False)

        self.weight_dtype = torch.float32
        if args.mixed_precision == "fp16":
            self.weight_dtype = torch.float16

        ckpt = torch.load(args.ckpt_path, map_location="cpu")
        self.load_ckpt(ckpt)

        self.unet.to("cuda", dtype=self.weight_dtype)
        self.vae.to("cuda", dtype=self.weight_dtype)
        self.timesteps = torch.tensor([999], device="cuda").long()
        self.noise_scheduler.alphas_cumprod = self.noise_scheduler.alphas_cumprod.cuda()

    def load_ckpt(self, model):
        target_modules = model["unet_lora_encoder_modules"] + model["unet_lora_decoder_modules"] + model["unet_lora_others_modules"]
        lora_conf_encoder = LoraConfig(r=model["rank_unet"], init_lora_weights="gaussian", target_modules=model["unet_lora_encoder_modules"])
        lora_conf_decoder = LoraConfig(r=model["rank_unet"], init_lora_weights="gaussian", target_modules=model["unet_lora_decoder_modules"])
        lora_conf_others = LoraConfig(r=model["rank_unet"], init_lora_weights="gaussian", target_modules=model["unet_lora_others_modules"])
        self.unet.add_adapter(lora_conf_encoder, adapter_name="default_encoder")
        self.unet.add_adapter(lora_conf_decoder, adapter_name="default_decoder")
        self.unet.add_adapter(lora_conf_others, adapter_name="default_others")
        adapters = list(ODD_ADAPTERS)

        self.has_oddr_lora = any(ODDR_ADAPTER in k for k in model["state_dict_unet"])
        if self.has_oddr_lora:
            lora_conf_stage2 = LoraConfig(r=model["rank_unet"], init_lora_weights="gaussian", target_modules=target_modules)
            self.unet.add_adapter(lora_conf_stage2, adapter_name=ODDR_ADAPTER)
            adapters.append(ODDR_ADAPTER)

        _copy_strict(self.unet, model["state_dict_unet"], lambda n: "lora" in n or "conv_in" in n, "unet")
        self.unet.set_adapter(adapters)
        _copy_strict(self.vae, model["state_dict_vae"], lambda n: "skip" in n, "vae")

    @torch.no_grad()
    def forward(self, lq):
        """lq: [1, 3, H, W] in [-1, 1], H and W multiples of 64. Returns the deshadowed image in [-1, 1]."""
        lq = lq.to(self.weight_dtype)
        lq_latent = self.vae.encode(lq).latent_dist.sample() * self.vae.config.scaling_factor
        residuals = self.vae.encoder.residuals

        # Tiled latent inference for large inputs
        _, _, h, w = lq_latent.size()
        tile_size, tile_overlap = (self.args.latent_tiled_size, self.args.latent_tiled_overlap)
        if h * w <= tile_size * tile_size:
            model_pred = self.unet(lq_latent)
        else:
            tile_size = min(tile_size, min(h, w))
            tile_weights = self._gaussian_weights(tile_size, tile_size, 1)

            grid_rows = 0
            cur_x = 0
            while cur_x < lq_latent.size(-1):
                cur_x = max(grid_rows * tile_size - tile_overlap * grid_rows, 0) + tile_size
                grid_rows += 1

            grid_cols = 0
            cur_y = 0
            while cur_y < lq_latent.size(-2):
                cur_y = max(grid_cols * tile_size - tile_overlap * grid_cols, 0) + tile_size
                grid_cols += 1

            input_list = []
            noise_preds = []
            for row in range(grid_rows):
                for col in range(grid_cols):
                    if col < grid_cols - 1 or row < grid_rows - 1:
                        # extract tile from input image
                        ofs_x = max(row * tile_size - tile_overlap * row, 0)
                        ofs_y = max(col * tile_size - tile_overlap * col, 0)
                    if row == grid_rows - 1:
                        ofs_x = w - tile_size
                    if col == grid_cols - 1:
                        ofs_y = h - tile_size

                    input_start_x = ofs_x
                    input_end_x = ofs_x + tile_size
                    input_start_y = ofs_y
                    input_end_y = ofs_y + tile_size

                    input_tile = lq_latent[:, :, input_start_y:input_end_y, input_start_x:input_end_x]
                    input_list.append(input_tile)

                    if len(input_list) == 1 or col == grid_cols - 1:
                        input_list_t = torch.cat(input_list, dim=0)
                        model_out = self.unet(input_list_t)
                        input_list = []
                    noise_preds.append(model_out)

            # Stitch predictions of all tiles
            noise_pred = torch.zeros(lq_latent.shape, device=lq_latent.device)
            contributors = torch.zeros(lq_latent.shape, device=lq_latent.device)
            for row in range(grid_rows):
                for col in range(grid_cols):
                    if col < grid_cols - 1 or row < grid_rows - 1:
                        ofs_x = max(row * tile_size - tile_overlap * row, 0)
                        ofs_y = max(col * tile_size - tile_overlap * col, 0)
                    if row == grid_rows - 1:
                        ofs_x = w - tile_size
                    if col == grid_cols - 1:
                        ofs_y = h - tile_size

                    input_start_x = ofs_x
                    input_end_x = ofs_x + tile_size
                    input_start_y = ofs_y
                    input_end_y = ofs_y + tile_size

                    noise_pred[:, :, input_start_y:input_end_y, input_start_x:input_end_x] += noise_preds[row * grid_cols + col] * tile_weights
                    contributors[:, :, input_start_y:input_end_y, input_start_x:input_end_x] += tile_weights
            # Average overlapping areas
            noise_pred /= contributors
            model_pred = noise_pred

        x_denoised = self.noise_scheduler.step(model_pred, self.timesteps, lq_latent, return_dict=True).prev_sample
        output_image = (self.vae.decode(x_denoised.to(self.weight_dtype) / self.vae.config.scaling_factor,
                                        residuals=residuals).sample).clamp(-1, 1)
        return output_image

    def _gaussian_weights(self, tile_width, tile_height, nbatches):
        """Generates a gaussian mask of weights for tile contributions."""
        from numpy import pi, exp, sqrt

        latent_width = tile_width
        latent_height = tile_height

        var = 0.01
        midpoint = (latent_width - 1) / 2  # -1 because index goes from 0 to latent_width - 1
        x_probs = [exp(-(x - midpoint) * (x - midpoint) / (latent_width * latent_width) / (2 * var)) / sqrt(2 * pi * var) for x in range(latent_width)]
        midpoint = latent_height / 2
        y_probs = [exp(-(y - midpoint) * (y - midpoint) / (latent_height * latent_height) / (2 * var)) / sqrt(2 * pi * var) for y in range(latent_height)]

        weights = np.outer(y_probs, x_probs)
        return torch.tile(torch.tensor(weights, device=self.device), (nbatches, self.unet.config.in_channels, 1, 1))
