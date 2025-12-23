import math
import torch
import torch.nn as nn
from diffusers import DiffusionPipeline
from diffusers.utils import load_image
from PIL import Image

from token_mapper import TokenMapper

class CrossAttnConcatProcessor(torch.nn.Module):
    def __init__(self, base_processor, cond_ctx=None):
        super().__init__()
        self.base = base_processor
        # cond_ctx is expected to be (B, L, D)
        self.register_buffer("cond_ctx", cond_ctx, persistent=False)

    def set_cond_ctx(self, cond_ctx):
        self.cond_ctx = cond_ctx

    def forward(self, attn, hidden_states, encoder_hidden_states=None, attention_mask=None, temb=None, **kwargs):
        if encoder_hidden_states is None:
            encoder_hidden_states = hidden_states

        b = hidden_states.shape[0]
        cond = self.cond_ctx
        
        # Expand condition to batch size if needed
        if cond is not None:
             if cond.shape[0] != b:
                 cond = cond.expand(b, -1, -1)
             
             # Concatenate along sequence dimension
             encoder_hidden_states = torch.cat(
                 [encoder_hidden_states, cond.to(dtype=encoder_hidden_states.dtype, device=encoder_hidden_states.device)], 
                 dim=1
             )

        return self.base(attn, hidden_states, encoder_hidden_states=encoder_hidden_states, attention_mask=attention_mask, temb=temb, **kwargs)


class StreamerGenerator(nn.Module):
    def __init__(
        self,
        token_mapper,
        device="cuda",
        num_frames=24,
        fps=6,
        motion_bucket_id=127,
        noise_aug_strength=0.1
    ):
        super().__init__()
        self.device = device
        self.num_frames = num_frames
        self.fps = fps
        self.motion_bucket_id = motion_bucket_id
        self.noise_aug_strength = noise_aug_strength
        
        # Load SVD
        self.pipe = DiffusionPipeline.from_pretrained(
            "stabilityai/stable-video-diffusion-img2vid-xt",
            torch_dtype=torch.float16,
            variant="fp16",
        )
        self.pipe.to(device)
        
        self.token_mapper = token_mapper
        self.token_mapper.to(device)

        # Install Hooks
        self._install_hooks()

    def _install_hooks(self):
        orig = self.pipe.unet.attn_processors
        wrapped = {}
        for name, proc in orig.items():
            if "attn2" in name:
                wrapped[name] = CrossAttnConcatProcessor(proc)
            else:
                wrapped[name] = proc
        self.pipe.unet.set_attn_processor(wrapped)
        self.wrapped_processors = wrapped

    def set_context(self, cond_ctx):
        """Updates the context token buffer in all attention processors"""
        for proc in self.wrapped_processors.values():
            if isinstance(proc, CrossAttnConcatProcessor):
                proc.set_cond_ctx(cond_ctx)

    def forward(self, encoded_tokens, image):
        """
        Inference Forward Pass.
        Generates video frames based on encoded_tokens and a conditioning image.
        
        Args:
            encoded_tokens: (B, N, D) Tensor
            image: PIL Image or Tensor (the conditioning image)
        """
        encoded_tokens = encoded_tokens.to(self.device)
        
        # Map tokens
        cond_ctx = self.token_mapper(encoded_tokens)
        
        # Update hooks with new context
        self.set_context(cond_ctx)

        # Generate using stored parameters
        with torch.no_grad():
            result = self.pipe(
                image=image,
                num_frames=self.num_frames,
                fps=self.fps,
                motion_bucket_id=self.motion_bucket_id,
                noise_aug_strength=self.noise_aug_strength,
            )
        
        return result.frames[0]
