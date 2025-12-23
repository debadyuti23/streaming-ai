import av
import numpy as np

def read_video_pyav(container, indices):
    '''
    Decode the video with PyAV decoder.
    Args:
        container (`av.container.input.InputContainer`): PyAV container.
        indices (`list[int]`): List of frame indices to decode.
    Returns:
        result (np.ndarray): np array of decoded frames of shape (num_frames, height, width, 3).
    '''
    frames = []
    container.seek(0)
    start_index = indices[0]
    end_index = indices[-1]
    for i, frame in enumerate(container.decode(video=0)):
        if i > end_index:
            break
        if i >= start_index and i in indices:
            frames.append(frame)
    return np.stack([x.to_ndarray(format="rgb24") for x in frames])


def sample_frame_indices(clip_len, frame_sample_rate, seg_len):
    '''
    Sample a given number of frame indices from the video.
    Args:
        clip_len (`int`): Total number of frames to sample.
        frame_sample_rate (`int`): Sample every n-th frame.
        seg_len (`int`): Maximum allowed index of sample's last frame.
    Returns:
        indices (`list[int]`): List of sampled frame indices
    '''
    converted_len = int(clip_len * frame_sample_rate)
    end_idx = np.random.randint(converted_len, seg_len)
    start_idx = end_idx - converted_len
    indices = np.linspace(start_idx, end_idx, num=clip_len)
    indices = np.clip(indices, start_idx, end_idx - 1).astype(np.int64)
    return indices

def prepare_video_data(file_path):
    """
    Reads the entire video from file_path and returns it as a numpy array.
    Returns:
        video (np.ndarray): Shape (T, H, W, 3) in RGB format.
    """
    container = av.open(file_path)
    frames = []
    # Decode all frames in the first video stream
    for frame in container.decode(video=0):
        frames.append(frame.to_ndarray(format="rgb24"))
    
    if not frames:
        raise RuntimeError(f"No frames found in {file_path}")

    # Ensure contiguous memory layout for DataLoader safety
    video = np.ascontiguousarray(np.stack(frames))
    return video
