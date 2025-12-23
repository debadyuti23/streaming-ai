This is the prototype of what an ai-streamer could be. It uses trained network pipeline which accepts a gameplay video and still image of the streamer, generates appropriate reaction (eg: "game over"--> sad, "jumpscare" --> fear)

## Arhcitecture

VideoEncoder model (frozen): encoding ```gameplay-video``` into ```encoded-tokens```
Token Mapper (trainable): map ```encoded-tokens``` into ```reaction-tokens```
ReactionDecoder (SVD) (frozen): ```reaction-tokens``` as conditional and ```ai-streamer-image``` as input image to produce reaction video through diffusion


## How to train

```bash
python train.py --input_dir </path/to/raw/videos> --gameplay_dir </path/to/gameplay/only> --streamer_dir </path/to/streamer/only> --epochs 10 --image_path </path/to/ai/streamer/image>
```

All arguments list:

```
input_dir: Path to input videos
gameplay_dir: Path where only gameplay-only videos contain
streamer_dir: Path where only streamer reaction-only videos contain
epochs: Number of epochs to train
lr: Learning rate (default=1e-4)
save_ckpt: load the pretrained token_mapper
final_ckpt: save the trained token_mapper
image_path: Path to the still image of AI-streamer (feel free use to use your own image)
output_path: Path to AI-streamer reaction video
batch_size: Batch size for training
```

## How to test

```bash
python run.py --gameplay_path </path/to/gameplay/only/video> --image_path </path/to/ai/streamer/image> --output_path </path/to/ai/streamer/output>
```

## Model choice

1. VideoEncoder: VideoMAE [huggingface](https://huggingface.co/docs/transformers/en/model_doc/videomae)
2. Token Mapper: Standard NN + Transformer
3. ReactionDecoder: Stable-diffusion SVD [huggingface](https://huggingface.co/docs/diffusers/en/using-diffusers/svd)
