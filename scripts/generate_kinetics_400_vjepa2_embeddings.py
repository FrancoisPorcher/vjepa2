import os
import pandas as pd
from transformers import AutoVideoProcessor, AutoModel
import numpy as np
import torch
from decord import VideoReader
from typing import List, Dict, Tuple, Optional
from tqdm import tqdm
from torch.utils.data import Dataset, DataLoader

def make_embed_path(video_path: str, embeddings_root: str) -> str:
    """
    Create a mirrored path under embeddings_root with .pt extension.
    If 'Kinetics400/' exists in the path, mirror from there; otherwise use basename.
    """
    norm = os.path.normpath(video_path)
    key = "Kinetics400" + os.sep
    if key in norm:
        rel = norm.split(key, 1)[1]
    else:
        rel = os.path.basename(norm)
    rel_no_ext = os.path.splitext(rel)[0]
    out_path = os.path.join(embeddings_root, rel_no_ext + ".pt")
    return out_path

def ensure_parent_dir(path: str):
    os.makedirs(os.path.dirname(path), exist_ok=True)

def load_64_frames_with_padding(vr: VideoReader) -> Tuple[np.ndarray, bool]:
    """
    Returns exactly 64 frames as uint8 [T,H,W,C] and a 'padded' flag.
    Desired indices are 0..126 step 2. If the video is shorter,
    we fetch what exists and pad by repeating the last available frame.
    """
    desired_idx = np.arange(0, 128, 2)  # 64 indices: 0,2,...,126
    n = len(vr)
    if n == 0:
        raise RuntimeError("Video has zero frames.")
    valid_idx = desired_idx[desired_idx < n]
    if valid_idx.size == 0:
        valid_idx = np.array([0], dtype=np.int64)

    frames = vr.get_batch(valid_idx).asnumpy()  # [T,H,W,C] uint8
    t = frames.shape[0]
    padded = False
    if t < 64:
        last = frames[-1:]                          # [1,H,W,C]
        pad = np.repeat(last, 64 - t, axis=0)       # [64-t,H,W,C]
        frames = np.concatenate([frames, pad], axis=0)
        padded = True
    elif t > 64:
        frames = frames[:64]  # shouldn't happen with our indices, but safe-guard
    return frames, padded

class K400Dataset(Dataset):
    def __init__(self, df: pd.DataFrame, embeddings_root="/private/home/francoisporcher/kinetics_400_embeddings_vjepa2_vitl_fpc64_256"):
        self.df = df
        self.embeddings_root = embeddings_root

    def __len__(self):
        return self.df.shape[0]

    def __getitem__(self, k):
        video_path = self.df.loc[k, "video_path"]
        index = int(self.df.loc[k, "index"])
        failed = False
        error_msg = ""
        n_frames: Optional[int] = None
        padded = False

        try:
            vr = VideoReader(video_path)
            n_frames = len(vr)
            video, padded = load_64_frames_with_padding(vr)     # [64,H,W,C] uint8
            video_tensor = torch.from_numpy(video).permute(0, 3, 1, 2)  # [64,C,H,W]
        except Exception as e:
            # Create a dummy tensor so collate doesn't break, but we will skip processing later.
            failed = True
            error_msg = str(e)
            # Provide a minimal dummy frame (processor will resize if used, but we skip on failed)
            video_tensor = torch.zeros((64, 3, 64, 64), dtype=torch.uint8)

        out_path = make_embed_path(video_path, self.embeddings_root)
        return {
            "video": video_tensor,          # [64,C,H,W] uint8 tensor
            "index": index,
            "video_path": video_path,
            "out_path": out_path,
            "failed": failed,
            "error": error_msg,
            "n_frames": n_frames,
            "padded": padded,
        }

def collate_videos(batch):
    """Pass list of [T,C,H,W] tensors to the processor; keep metadata."""
    videos = [b["video"] for b in batch]
    meta = [{
        "video_path": b["video_path"],
        "index": b["index"],
        "out_path": b["out_path"],
        "failed": b["failed"],
        "error": b["error"],
        "n_frames": b["n_frames"],
        "padded": b["padded"],
    } for b in batch]
    return videos, meta

def main():
    kinetics_400_csv_path = "/checkpoint/abardes/datasets/Kinetics400/annotations/k400_train_paths.csv"
    embeddings_root = "/private/home/francoisporcher/kinetics_400_embeddings_vjepa2_vitl_fpc64_256"
    hf_repo = "facebook/vjepa2-vitl-fpc64-256"

    os.makedirs(embeddings_root, exist_ok=True)

    # Read CSV (two columns: video_path, index)
    df_400 = pd.read_csv(kinetics_400_csv_path, header=None, names=["video_path", "index"], sep=" ")

    dataset = K400Dataset(df=df_400, embeddings_root=embeddings_root)
    dataloader = DataLoader(
        dataset,
        batch_size=1,
        shuffle=False,
        num_workers=4,
        drop_last=False,
        pin_memory=True,
        collate_fn=collate_videos,
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = AutoModel.from_pretrained(hf_repo).to(device).eval()
    processor = AutoVideoProcessor.from_pretrained(hf_repo)

    # Manifest rows accumulate here
    manifest_rows = []

    # Wrap the entire inference loop once
    with torch.inference_mode():
        for videos, metas in tqdm(dataloader, desc="Embedding videos"):
            # batch_size=1 by construction
            meta = metas[0]
            video_path = meta["video_path"]
            out_path = meta["out_path"]

            if meta["failed"]:
                # Log failure; skip compute/save
                manifest_rows.append({
                    "video_path": video_path,
                    "index": meta["index"],
                    "out_path": out_path,
                    "padded": meta["padded"],
                    "failed": True,
                    "error": meta["error"],
                    "n_frames": meta["n_frames"],
                    "feature_shape": None,
                })
                continue

            try:
                inputs = processor(videos, return_tensors="pt")["pixel_values_videos"]  # [B,T,C,H,W]
                inputs = inputs.to(device, non_blocking=True)

                feats = model.get_vision_features(inputs)  # model-specific shape
                features = feats.squeeze(0).detach().cpu()

                ensure_parent_dir(out_path)
                torch.save(features, out_path)

                manifest_rows.append({
                    "video_path": video_path,
                    "index": meta["index"],
                    "out_path": out_path,
                    "padded": meta["padded"],
                    "failed": False,
                    "error": "",
                    "n_frames": meta["n_frames"],
                    "feature_shape": tuple(features.shape),
                })

            except Exception as e:
                # Inference/IO failure
                manifest_rows.append({
                    "video_path": video_path,
                    "index": meta["index"],
                    "out_path": out_path,
                    "padded": meta["padded"],
                    "failed": True,
                    "error": str(e),
                    "n_frames": meta["n_frames"],
                    "feature_shape": None,
                })

    # Save manifest at the end
    manifest_df = pd.DataFrame(manifest_rows)
    manifest_path = os.path.join(embeddings_root, "manifest.csv")
    ensure_parent_dir(manifest_path)
    manifest_df.to_csv(manifest_path, index=False)
    print(f"Saved manifest with {len(manifest_df)} rows to {manifest_path}")

if __name__ == "__main__":
    main()
