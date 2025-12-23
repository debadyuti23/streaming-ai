import subprocess
import os
import sys

def upsample_video_ffmpeg(input_path, output_path, target_fps=60):
    """
    Upsamples video using FFmpeg's motion interpolation (minterpolate).
    This creates fake intermediate frames based on motion vectors (Optical Flow).
    
    Note: This is CPU-intensive. For 4s of video it might take ~30-60s.
    """
    if not os.path.exists(input_path):
        raise FileNotFoundError(f"Input video not found: {input_path}")

    print(f"Upsampling {input_path} to {target_fps} FPS using FFmpeg motion interpolation...")

    # minterpolate filter options:
    # mi_mode=mci: Motion Compensated Interpolation (smooths motion)
    # mc_mode=aobmc: Adaptive Overlapped Block Motion Compensation (reduces artifacts)
    # vsbmc=1: Variable-size block motion compensation (better detail)
    # fps={target_fps}: Target frame rate
    filter_str = f"minterpolate='mi_mode=mci:mc_mode=aobmc:vsbmc=1:fps={target_fps}'"

    cmd = [
        "ffmpeg", "-y",                # Overwrite output
        "-v", "error",                 # Quiet mode
        "-i", input_path,              # Input file
        "-vf", filter_str,             # Video filter
        "-c:v", "libx264",             # Codec
        "-crf", "18",                  # High quality
        "-preset", "slow",             # Better compression
        "-pix_fmt", "yuv420p",         # Standard pixel format
        output_path
    ]
    
    try:
        subprocess.run(cmd, check=True)
        print(f"Success! Saved upsampled video to {output_path}")
    except subprocess.CalledProcessError as e:
        print(f"Error running FFmpeg: {e}")
        print("Ensure ffmpeg is installed and accessible in your PATH.")
        sys.exit(1)

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=str, default="output.mp4")
    parser.add_argument("--output", type=str, default="output_60fps.mp4")
    parser.add_argument("--fps", type=int, default=60)
    args = parser.parse_args()
    
    upsample_video_ffmpeg(args.input, args.output, args.fps)

