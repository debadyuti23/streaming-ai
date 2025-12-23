import argparse
import os
import glob
import math
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader, random_split
from diffusers.utils.export_utils import export_to_video
from diffusers.utils.loading_utils import load_image
import torch
import torch.nn.functional as F
import numpy as np
import cv2
from diffusers import AutoencoderKLTemporalDecoder
from transformers import AutoImageProcessor
from diffusers.schedulers import EulerDiscreteScheduler, DDPMScheduler

from prepare_data import prepare_video_data
from split_gameplay_stream import process_video
from split_video_clips import split_video

# New model classes
from upsample import upsample_video_ffmpeg
from videomae import GameplayEncoder
from videogen import StreamerGenerator
from token_mapper import TokenMapper

def collate_video_batch(batch):
    """
    Custom collate function to handle video stacking by truncating to the minimum length in the batch.
    batch: List of tuples (gp_video, st_video, gp_path)
    gp_video: numpy (T, H, W, 3)
    st_video: numpy (T, H, W, 3)
    """
    gp_videos = []
    st_videos = []
    paths = []
    
    # 1. Find min frame count in this batch to ensure stackability
    min_frames_gp = min([item[0].shape[0] for item in batch])
    min_frames_st = min([item[1].shape[0] for item in batch])
    
    # Common min length (if we want synced trimming, though networks might handle them separately)
    # Actually, train_one_step handles subsampling of streamer video separately based on FPS.
    # But for stacking, we just need them to be stackable per type.
    
    for gp, st, path in batch:
        # Truncate
        gp_trunc = gp[:min_frames_gp]
        st_trunc = st[:min_frames_st]
        
        gp_videos.append(gp_trunc)
        st_videos.append(st_trunc)
        paths.append(path)
        
    # Stack
    gp_batch = np.stack(gp_videos) # (B, T_min, H, W, 3)
    st_batch = np.stack(st_videos) # (B, T_min, H, W, 3)
    
    return torch.from_numpy(gp_batch), torch.from_numpy(st_batch), paths

def resize_video_frames(video, target_size):
    """
    Resize video frames to target size (W, H).
    video: (T, H, W, 3) numpy array
    target_size: (W, H) tuple
    """
    resized_frames = []
    for frame in video:
        # cv2.resize expects (W, H)
        resized = cv2.resize(frame, target_size, interpolation=cv2.INTER_LINEAR)
        resized_frames.append(resized)
    return np.stack(resized_frames)

class VideoDataset(Dataset):
    def __init__(self, gameplay_dir, streamer_dir, image_processor_name="MCG-NJU/videomae-base"):
        self.gameplay_dir = gameplay_dir
        self.streamer_dir = streamer_dir
        self.image_processor = AutoImageProcessor.from_pretrained(image_processor_name)
        
        self.gameplay_clips = sorted(glob.glob(os.path.join(gameplay_dir, "*.mp4")))
        
        self.valid_pairs = []
        for gp_path in self.gameplay_clips:
            filename = os.path.basename(gp_path)
            st_filename = filename.replace("_gameplay_", "_streamer_")
            st_path = os.path.join(streamer_dir, st_filename)
            if os.path.exists(st_path):
                self.valid_pairs.append((gp_path, st_path))
            else:
                pass
        
        if not self.valid_pairs:
            print("No valid pairs found. Ensure split_video_clips has run.")

    def __len__(self):
        return len(self.valid_pairs)

    def __getitem__(self, idx):
        gp_path, st_path = self.valid_pairs[idx]
        try:
            gp_video = prepare_video_data(gp_path) # (T, H, W, 3)
            st_video = prepare_video_data(st_path) # (T, H, W, 3)
            
            # 1. Process Gameplay using VideoMAE processor (Correct Resize/Norm)
            # processor expects list of numpy arrays, returns dict with tensors
            gp_inputs = self.image_processor(list(gp_video), return_tensors="pt")
            gp_video = gp_inputs["pixel_values"][0] # (T, 3, 224, 224) Tensor
            gp_video = gp_video.numpy() # Convert to numpy for collate_fn stack
            
            # 2. Process Streamer (Target) - Manual Resize to SVD resolution
            # Streamer -> 512x512 (Standard SVD training resolution)
            st_video = resize_video_frames(st_video, (512, 512)) # (T, 512, 512, 3) Numpy
            
            return gp_video, st_video, gp_path
        except Exception as e:
            print(f"Error loading {gp_path}: {e}")
            # Return safe defaults
            # gp: (16, 3, 224, 224) float32
            # st: (16, 512, 512, 3) uint8
            return np.zeros((16, 3, 224, 224), dtype=np.float32), np.zeros((16, 512, 512, 3), dtype=np.uint8), gp_path

def train_one_step(
    gameplay_video,
    streamer_video_full,
    mae_encoder,
    generator,
    noise_scheduler,
    optimizer,
    device,
    target_fps=6,
    source_fps=60,
    num_frames=24
):
    # Handle batch dim
    if isinstance(gameplay_video, torch.Tensor):
        gameplay_video = gameplay_video.to(device)
    else:
        gameplay_video = torch.from_numpy(gameplay_video).to(device)
    
    # Simple unbatching for B=1 safety
    # Removed to support proper batching
    # if gameplay_video.ndim == 5: 
    #    gameplay_video = gameplay_video[0] 
    #    streamer_video_full = streamer_video_full[0]

    with torch.no_grad():
        gameplay_tokens = mae_encoder(gameplay_video) # (B, N, 768)
        gameplay_tokens = gameplay_tokens.to(device)

    cond_ctx = generator.token_mapper(gameplay_tokens) # (B, L, 1024)
    cond_ctx = cond_ctx.to(dtype=generator.pipe.unet.dtype)
    
    generator.set_context(cond_ctx)

    stride = max(1, int(source_fps / target_fps))
    T_full = streamer_video_full.shape[1] # T is dim 1
    indices = np.arange(0, T_full, stride)
    if len(indices) > num_frames:
        indices = indices[:num_frames]
    
    # Handle batch slicing
    if isinstance(streamer_video_full, torch.Tensor):
        streamer_video_sampled = streamer_video_full[:, indices]
    else:
        streamer_video_sampled = streamer_video_full[:, indices]
    
    # (B, T, H, W, C) -> (B, C, T, H, W)
    if not isinstance(streamer_video_sampled, torch.Tensor):
        video_tensor = torch.from_numpy(streamer_video_sampled).permute(0, 4, 1, 2, 3).float()
    else:
        video_tensor = streamer_video_sampled.permute(0, 4, 1, 2, 3).float()
        
    video_tensor = (video_tensor / 127.5) - 1.0
    # video_tensor is already (B, C, T, H, W), no unsqueeze needed
    video_tensor = video_tensor.to(device) 
    video_tensor = video_tensor.to(dtype=generator.pipe.vae.dtype)

    with torch.no_grad():
        b, c, t, h, w = video_tensor.shape
        video_tensor_flattened = video_tensor.permute(0, 2, 1, 3, 4).reshape(b * t, c, h, w)
        latents_flattened = generator.pipe.vae.encode(video_tensor_flattened).latent_dist.sample()
        
        _, c_lat, h_lat, w_lat = latents_flattened.shape
        latents = latents_flattened.reshape(b, t, c_lat, h_lat, w_lat).permute(0, 2, 1, 3, 4)
        latents = latents * generator.pipe.vae.config.scaling_factor
        latents = latents.to(dtype=generator.pipe.unet.dtype)

    bsz = latents.shape[0]
    noise = torch.randn_like(latents)
    timesteps = torch.randint(0, noise_scheduler.config.num_train_timesteps, (bsz,), device=device).long()
    noisy_latents = noise_scheduler.add_noise(latents, noise, timesteps)

    with torch.no_grad():
        cond_image_tensor = video_tensor[:, :, 0, :, :] 
        cond_image_latents = generator.pipe.vae.encode(cond_image_tensor).latent_dist.mode()
        cond_image_latents = cond_image_latents.unsqueeze(2).repeat(1, 1, t, 1, 1)
        cond_image_latents = cond_image_latents.to(dtype=generator.pipe.unet.dtype)

    unet_input = torch.cat([noisy_latents, cond_image_latents], dim=1)
    unet_input = unet_input.permute(0, 2, 1, 3, 4)
    unet_input.requires_grad_(True)

    image_tensor = video_tensor[:, :, 0, :, :]
    with torch.no_grad():
        img_resized = F.interpolate(image_tensor, size=(224, 224), mode='bicubic', align_corners=False)
        image_embeddings = generator.pipe.image_encoder(img_resized).image_embeds
        image_embeddings = image_embeddings.unsqueeze(1).to(dtype=generator.pipe.unet.dtype)
        
        add_time_ids = torch.tensor([[6, 127, 0.02]], device=device, dtype=generator.pipe.unet.dtype)
        add_time_ids = add_time_ids.repeat(bsz, 1)

    model_pred = generator.pipe.unet(
        unet_input,
        timesteps,
        encoder_hidden_states=image_embeddings,
        added_time_ids=add_time_ids,
        return_dict=False,
    )[0]

    model_pred = model_pred.permute(0, 2, 1, 3, 4)

    if generator.pipe.scheduler.config.prediction_type == "v_prediction":
        alphas_cumprod = noise_scheduler.alphas_cumprod.to(device)
        alpha_prod_t = alphas_cumprod[timesteps].reshape(bsz, 1, 1, 1, 1)
        sqrt_alpha_prod = alpha_prod_t ** 0.5
        sqrt_one_minus_alpha_prod = (1 - alpha_prod_t) ** 0.5
        target = sqrt_alpha_prod * noise - sqrt_one_minus_alpha_prod * latents
    else:
        target = noise

    loss = F.mse_loss(model_pred.float(), target.float())

    return loss

def train_one_epoch(
    dataloader,
    mae_encoder,
    generator,
    noise_scheduler,
    optimizer,
    lr_scheduler,
    device
):
    generator.token_mapper.train()
    total_loss = 0.0
    steps_in_epoch = 0
    
    for batch_idx, (gp_vid, st_vid, _) in enumerate(dataloader):
        optimizer.zero_grad()
        
        loss_tensor = train_one_step(
            gp_vid, st_vid, mae_encoder, generator, noise_scheduler, optimizer, device
        )
        
        loss_tensor.backward()
        optimizer.step()
        lr_scheduler.step()
        
        total_loss += loss_tensor.item()
        steps_in_epoch += 1
            
        if batch_idx % 10 == 0:
            print(f"  Step [{batch_idx}/{len(dataloader)}] Loss: {loss_tensor.item():.4f}")
            
    return total_loss / max(1, steps_in_epoch)

def validate_one_epoch(
    dataloader,
    mae_encoder,
    generator,
    noise_scheduler,
    device
):
    generator.token_mapper.eval()
    total_loss = 0.0
    steps = 0
    
    with torch.no_grad():
        for gp_vid, st_vid, _ in dataloader:
            loss_tensor = train_one_step(
                gp_vid, st_vid, mae_encoder, generator, noise_scheduler, None, device
            )
            total_loss += loss_tensor.item()
            steps += 1
            
    return total_loss / max(1, steps)

def prepare_dataset(input_dir, gameplay_dir, streamer_dir):
    gp_shorts_dir = os.path.join(gameplay_dir, "gameplay_shorts")
    st_shorts_dir = os.path.join(streamer_dir, "streamer_shorts")

    # 1. Check if shorts exist and are populated
    has_gp_shorts = os.path.exists(gp_shorts_dir) and len(glob.glob(os.path.join(gp_shorts_dir, "*.mp4"))) > 0
    has_st_shorts = os.path.exists(st_shorts_dir) and len(glob.glob(os.path.join(st_shorts_dir, "*.mp4"))) > 0
    
    if has_gp_shorts and has_st_shorts:
        print("Found existing short clips in gameplay_shorts/ and streamer_shorts/. Skipping all preprocessing.")
        return gp_shorts_dir, st_shorts_dir

    print("Short clips not found or incomplete. Checking for intermediate streams...")
    
    # Ensure directories exist
    os.makedirs(gameplay_dir, exist_ok=True)
    os.makedirs(streamer_dir, exist_ok=True)

    has_gp_streams = len(glob.glob(os.path.join(gameplay_dir, "*.mp4"))) > 0
    has_st_streams = len(glob.glob(os.path.join(streamer_dir, "*.mp4"))) > 0

    # 2. If streams missing, process from raw inputs
    if not (has_gp_streams and has_st_streams):
        print("Intermediate streams not found. Checking for raw input videos...")
        if input_dir and os.path.exists(input_dir):
            raw_videos = glob.glob(os.path.join(input_dir, "*.mp4"))
            if not raw_videos:
                print(f"No .mp4 files found in {input_dir}")
            
            for raw_path in raw_videos:
                filename = os.path.basename(raw_path)
                gp_path = os.path.join(gameplay_dir, filename.replace(".mp4", "_gameplay.mp4"))
                st_path = os.path.join(streamer_dir, filename.replace(".mp4", "_streamer.mp4"))
                
                # Only process if streams don't exist
                if not os.path.exists(gp_path) or not os.path.exists(st_path):
                    print(f"Splitting raw video: {filename}")
                    try:
                        process_video(raw_path, gp_path, st_path, stream_box=(768, 512), mask_mode="black", face_detector=True)
                    except Exception as e:
                        print(f"Failed to process {filename}: {e}")
        else:
             print(f"Input directory {input_dir} not found. Cannot generate streams.")

    # 3. Split streams into shorts
    os.makedirs(gp_shorts_dir, exist_ok=True)
    os.makedirs(st_shorts_dir, exist_ok=True)
    
    print("Splitting streams into short clips...")
    for gp_video in glob.glob(os.path.join(gameplay_dir, "*.mp4")):
        split_video(gp_video, gp_shorts_dir, clip_length=4.0)

    for st_video in glob.glob(os.path.join(streamer_dir, "*.mp4")):
        split_video(st_video, st_shorts_dir, clip_length=4.0)
            
    return gp_shorts_dir, st_shorts_dir

def plot_losses(train_losses, val_losses, save_path="loss_plot.png"):
    plt.figure(figsize=(10, 5))
    plt.plot(train_losses, label="Train Loss")
    val_epochs = np.linspace(0, len(train_losses), len(val_losses))
    plt.plot(val_epochs, val_losses, label="Validation Loss", marker='o')
    plt.xlabel("Steps")
    plt.ylabel("Loss")
    plt.legend()
    plt.title("Training vs Validation Loss")
    plt.savefig(save_path)
    print(f"Saved loss plot to {save_path}")

from diffusers.optimization import get_cosine_schedule_with_warmup

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_dir", type=str, default="data/raw_videos")
    parser.add_argument("--gameplay_dir", type=str, default="data/gameplays")
    parser.add_argument("--streamer_dir", type=str, default="data/streams")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--save_ckpt", type=str, default="token_mapper.pt")
    parser.add_argument("--final_ckpt", type=str, default="token_mapper_final.pt")
    parser.add_argument("--image_path", type=str, default="default_streamer.png")
    parser.add_argument("--output_path", type=str, default="output_test.mp4")
    parser.add_argument("--batch_size", type=int, default=1)
    # parser.add_argument("--gradient_accumulation_steps", type=int, default=4)
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Running on device: {device}")

    # 1. Prepare Dataset
    print("Preparing dataset from", args.input_dir)
    gp_shorts, st_shorts = prepare_dataset(args.input_dir, args.gameplay_dir, args.streamer_dir)
    
    # 2. Create Dataset
    full_dataset = VideoDataset(gp_shorts, st_shorts)
    
    if len(full_dataset) == 0:
        print("No clips found! Exiting.")
        exit()
        
    # Split: 70/30 first (TrainVal / Test)
    train_val_size = int(0.7 * len(full_dataset))
    test_size = len(full_dataset) - train_val_size
    train_val_ds, test_ds = random_split(full_dataset, [train_val_size, test_size], generator=torch.Generator().manual_seed(42))
    
    # Split TrainVal: 80/20 (Train / Val)
    train_size = int(0.8 * len(train_val_ds))
    val_size = len(train_val_ds) - train_size
    train_ds, val_ds = random_split(train_val_ds, [train_size, val_size], generator=torch.Generator().manual_seed(42))
    
    print(f"Data Split: Train={len(train_ds)}, Val={len(val_ds)}, Test={len(test_ds)}")
    
    # 3. DataLoaders
    # Apply custom collate_fn
    train_loader = DataLoader(
        train_ds, 
        batch_size=args.batch_size, 
        shuffle=True, 
        num_workers=4, 
        pin_memory=True,
        collate_fn=collate_video_batch
    )
    val_loader = DataLoader(
        val_ds, 
        batch_size=args.batch_size, 
        shuffle=False,
        num_workers=4, 
        pin_memory=True,
        collate_fn=collate_video_batch
    )
    # Test loader for later if needed
    test_loader = DataLoader(test_ds, batch_size=1, shuffle=False)

    # 4. Init Models
    mae_encoder = GameplayEncoder(device=device)
    token_mapper = TokenMapper(src_dim=768, ctx_dim=1024)
    generator = StreamerGenerator(token_mapper=token_mapper, device=device)
    noise_scheduler = DDPMScheduler.from_config(generator.pipe.scheduler.config)

    # Freeze SVD
    generator.pipe.unet.requires_grad_(False)
    generator.pipe.vae.requires_grad_(False)
    generator.pipe.image_encoder.requires_grad_(False)
    
    # 5. Resume or Start New
    start_epoch = 0
    if os.path.exists(args.final_ckpt):
        print(f"Found final checkpoint {args.final_ckpt}, loading...")
        generator.token_mapper.load_state_dict(torch.load(args.final_ckpt, map_location=device))
        print("Model loaded. Skipping training (remove checkpoint to retrain).")
        start_epoch = args.epochs
    elif os.path.exists(args.save_ckpt):
        print(f"Found checkpoint {args.save_ckpt}, resuming...")
        generator.token_mapper.load_state_dict(torch.load(args.save_ckpt, map_location=device))
    
    optimizer = torch.optim.AdamW(generator.token_mapper.parameters(), lr=args.lr)

    # Scheduler
    num_training_steps = args.epochs * len(train_loader)
    lr_scheduler = get_cosine_schedule_with_warmup(
        optimizer=optimizer,
        num_warmup_steps=int(0.1 * num_training_steps),
        num_training_steps=num_training_steps,
    )

    # 6. Training Loop
    train_losses_history = []
    val_losses_history = []

    if start_epoch < args.epochs:
        print(f"Starting training for {args.epochs} epochs...")
        for epoch in range(start_epoch, args.epochs):
            # TRAIN
            avg_train = train_one_epoch(
                train_loader, mae_encoder, generator, noise_scheduler, optimizer, lr_scheduler, device
            )
            train_losses_history.append(avg_train)
            
            # VALIDATION
            avg_val = validate_one_epoch(
                val_loader, mae_encoder, generator, noise_scheduler, device
            )
            val_losses_history.append(avg_val)
            
            print(f"Epoch {epoch+1} Done. Train Loss: {avg_train:.4f} | Val Loss: {avg_val:.4f}")
            
            if (epoch + 1) % 10 == 0:
                ckpt_name = args.save_ckpt.replace(".pt", f"_epoch{epoch+1}.pt")
                torch.save(generator.token_mapper.state_dict(), ckpt_name)
                print(f"Saved checkpoint: {ckpt_name}")
            
            torch.save(generator.token_mapper.state_dict(), args.save_ckpt)

        torch.save(generator.token_mapper.state_dict(), args.final_ckpt)
        print(f"Saved FINAL model to {args.final_ckpt}")
        plot_losses(train_losses_history, val_losses_history)

    # 7. Inference on Test Set
    print("Running inference on Test Set sample...")
    if len(test_ds) > 0:
        gp_vid, st_vid, gp_path_batch = test_ds[0]
        gp_path = gp_path_batch
        print(f"Generating reaction for Test Clip: {gp_path}")
        
        image = load_image(args.image_path)
        
        if isinstance(gp_vid, np.ndarray):
            gp_vid = torch.from_numpy(gp_vid).to(device)
        else:
            gp_vid = gp_vid.to(device)
            
        if gp_vid.ndim == 4:
            gp_vid = gp_vid.unsqueeze(0)
            
        with torch.no_grad():
            encoded_tokens = mae_encoder(gp_vid)
        
        streamer_frames = generator(encoded_tokens, image)
        
        print("Upsampling to 60fps...")
        intermediate_path = args.output_path.replace(".mp4", "_6fps.mp4")
        export_to_video(streamer_frames, intermediate_path, fps=6)
        upsample_video_ffmpeg(intermediate_path, args.output_path, target_fps=60)
        print(f"Test Generation Saved: {args.output_path}")
