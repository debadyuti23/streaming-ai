import argparse
import os
import torch
from diffusers.utils import export_to_video, load_image

from prepare_data import prepare_video_data
from split_gameplay_stream import process_video
from upsample import upsample_video_ffmpeg

# Models
from videomae import GameplayEncoder
from videogen import StreamerGenerator
from token_mapper import TokenMapper

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_path", type=str, default="input.mp4")
    parser.add_argument("--output_path", type=str, default="output.mp4")
    parser.add_argument("--gameplay_path", type=str)
    parser.add_argument("--streamer_path", type=str)
    parser.add_argument("--stream_box", type=tuple, default=(768, 512))
    parser.add_argument("--mask_mode", type=str, default="black")
    parser.add_argument("--image_path", type=str, default="image.png")
    parser.add_argument("--face_detector", type=bool, action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--token_mapper_ckpt", type=str, default=None)
    # Generation args
    parser.add_argument("--num_frames", type=int, default=24) # 4s @ 6fps
    parser.add_argument("--fps", type=int, default=6)
    parser.add_argument("--motion_bucket_id", type=int, default=127)
    parser.add_argument("--noise_aug_strength", type=float, default=0.1)

    args = parser.parse_args()
    args.streamer_path = args.streamer_path or args.input_path.replace(".mp4", "_streamer.mp4")
    args.gameplay_path = args.gameplay_path or args.input_path.replace(".mp4", "_gameplay.mp4")
    args.stream_box = tuple(args.stream_box)
    
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # 1. Process Video
    if not os.path.exists(args.gameplay_path):
        process_video(args.input_path, args.gameplay_path, args.streamer_path, args.stream_box, args.mask_mode, args.face_detector)

    # 2. Init Models
    print("Loading models...")
    mae_encoder = GameplayEncoder(device=device)
    
    # Load TokenMapper if ckpt exists
    token_mapper = None
    if args.token_mapper_ckpt and os.path.exists(args.token_mapper_ckpt):
        print(f"Loading TokenMapper from {args.token_mapper_ckpt}")
        token_mapper = TokenMapper(src_dim=768, ctx_dim=1024)
        token_mapper.load_state_dict(torch.load(args.token_mapper_ckpt, map_location=device))
    
    # Init generator with config
    generator = StreamerGenerator(
        token_mapper=token_mapper, 
        device=device,
        num_frames=args.num_frames,
        fps=args.fps,
        motion_bucket_id=args.motion_bucket_id,
        noise_aug_strength=args.noise_aug_strength
    )

    # 3. Encode Gameplay
    print("Encoding gameplay...")
    video = prepare_video_data(args.gameplay_path)
    with torch.no_grad():
        encoded_tokens = mae_encoder(video)
    
    # 4. Generate Streamer Video
    print("Generating streamer reaction...")
    
    # Load image outside
    image = load_image(args.image_path)
    
    streamer_frames = generator(
        encoded_tokens,
        image
    )

    # Save intermediate
    intermediate_path = args.output_path.replace(".mp4", "_6fps.mp4")
    export_to_video(streamer_frames, intermediate_path, fps=args.fps)
    print(f"Saved intermediate: {intermediate_path}")

    # 5. Upsample
    print("Upsampling to 60fps...")
    upsample_video_ffmpeg(intermediate_path, args.output_path, target_fps=60)
    print(f"Done! Saved {args.output_path}")
