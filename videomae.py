import numpy as np
import torch
import torch.nn as nn
from transformers import AutoImageProcessor, VideoMAEModel

class GameplayEncoder(nn.Module):
    def __init__(self, model_id="MCG-NJU/videomae-base", device="cuda"):
        super().__init__()
        self.device = device
        self.image_processor = AutoImageProcessor.from_pretrained(model_id)
        self.model = VideoMAEModel.from_pretrained(model_id).to(device)
        self.model.eval()
        self.chunk_size = 16

    def forward(self, video):
        """
        video: 
            - (T, H, W, 3) numpy array or tensor (0-255) -> will be processed
            - (B, T, C, H, W) tensor (normalized) -> will be chunked directly
            - (T, C, H, W) tensor (normalized) -> will be chunked directly
        Returns: (1, N_total, 768) tensor
        """
        processed_input = None
        
        # Check if already processed (Float tensor, C=3 in dim 1 or 2)
        if isinstance(video, torch.Tensor) and video.dtype != torch.uint8:
            # Assume it's pixel_values
            if video.ndim == 5: # (B, T, C, H, W)
                processed_input = video 
            elif video.ndim == 4: # (T, C, H, W)
                processed_input = video.unsqueeze(0) # Make it (1, T, C, H, W)
            
            # Ensure on device
            if processed_input is not None:
                processed_input = processed_input.to(self.device)

        if processed_input is None:
            # Raw video path logic (usually B=1 or list of videos)
            # For simplicity, if we get list of arrays or tensor (T,H,W,C), we assume B=1
            if isinstance(video, torch.Tensor):
                video = video.cpu().numpy().astype(np.uint8)
            
            # Process entire video
            inputs = self.image_processor(list(video), return_tensors="pt")
            processed_input = inputs["pixel_values"].to(self.device) # (1, T, 3, 224, 224) or (B, ...)

        # Now we have processed_input: (B, T, 3, 224, 224)
        B, T, C, H, W = processed_input.shape
        all_hidden_states = []

        # Process in chunks of 16 frames
        for start_idx in range(0, T, self.chunk_size):
            end_idx = min(start_idx + self.chunk_size, T)
            
            chunk = processed_input[:, start_idx:end_idx] # (B, N, 3, 224, 224)
            
            # Pad if shorter than chunk_size
            if chunk.shape[1] < self.chunk_size:
                pad_len = self.chunk_size - chunk.shape[1]
                # Repeat the last frame to fill
                last_frame = chunk[:, -1:] # (B, 1, 3, 224, 224)
                padding = last_frame.repeat(1, pad_len, 1, 1, 1)
                chunk = torch.cat([chunk, padding], dim=1)

            # chunk is (B, 16, 3, 224, 224)

            with torch.no_grad():
                # Model expects (batch, num_frames, num_channels, height, width)
                outputs = self.model(pixel_values=chunk)
                all_hidden_states.append(outputs.last_hidden_state) # (B, 1568, 768)

        if not all_hidden_states:
             # Should not happen with padding, but safe fallback
             raise ValueError(f"Video too short even after padding logic.")

        full_sequence = torch.cat(all_hidden_states, dim=1) # (B, N_total, 768)
        return full_sequence
