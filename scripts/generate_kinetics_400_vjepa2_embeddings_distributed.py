# vjepa2_ddp_extract.py
import os
import argparse
import pandas as pd
import numpy as np
import torch
import torch.distributed as dist
from torch.utils.data import Dataset, DataLoader, DistributedSampler
from typing import List, Dict, Tuple, Optional
from transformers import AutoVideoProcessor, AutoModel
from decord import VideoReader
from tqdm import tqdm

# ----------------------- Helpers -----------------------
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

# ----------------------- Dataset -----------------------
class K400Dataset(Dataset):
    def __init__(self, df: pd.DataFrame, embeddings_root: str):
        self.df = df.reset_index(drop=True)
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
            failed = True
            error_msg = str(e)
            video_tensor = torch.zeros((64, 3, 64, 64), dtype=torch.uint8)  # placeholder

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
    """Return list of [T,C,H,W] tensors and per-item metadata."""
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

# ----------------------- Distributed utils -----------------------
def init_distributed(backend: str = "nccl"):
    if dist.is_available() and not dist.is_initialized():
        dist.init_process_group(backend=backend, init_method="env://")
    rank = dist.get_rank() if dist.is_initialized() else 0
    world_size = dist.get_world_size() if dist.is_initialized() else 1
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    return rank, world_size, local_rank

def is_main_process():
    return (not dist.is_initialized()) or dist.get_rank() == 0

# ----------------------- Main -----------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv_path", type=str,
                        default="/checkpoint/abardes/datasets/Kinetics400/annotations/k400_train_paths.csv")
    parser.add_argument("--embeddings_root", type=str,
                        default="/private/home/francoisporcher/kinetics_400_embeddings_vjepa2_vitl_fpc64_256")
    parser.add_argument("--hf_repo", type=str, default="facebook/vjepa2-vitl-fpc64-256")
    parser.add_argument("--batch_size", type=int, default=1)  # increase if memory allows
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--pin_memory", action="store_true")
    parser.add_argument("--backend", type=str, default="nccl")
    args = parser.parse_args()

    rank, world_size, local_rank = init_distributed(backend=args.backend)
    
    print("rank", rank)
    print("world size", world_size)
    print("local rank", local_rank)

    # Device setup
    use_cuda = torch.cuda.is_available()
    if use_cuda:
        torch.cuda.set_device(local_rank)
        device = torch.device(f"cuda:{local_rank}")
        torch.backends.cudnn.benchmark = True
    else:
        device = torch.device("cpu")
    print("Using device", device)

    if is_main_process():
        os.makedirs(args.embeddings_root, exist_ok=True)

    # Read CSV (two columns: video_path, index)
    df_400 = pd.read_csv(args.csv_path, header=None, names=["video_path", "index"], sep=" ")

    dataset = K400Dataset(df=df_400, embeddings_root=args.embeddings_root)
    sampler = DistributedSampler(
        dataset,
        num_replicas=world_size,
        rank=rank,
        shuffle=False,
        drop_last=False,
    ) if world_size > 1 else None

    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False if sampler is not None else False,
        sampler=sampler,
        num_workers=args.num_workers,
        drop_last=False,
        pin_memory=args.pin_memory,
        collate_fn=collate_videos,
        persistent_workers=(args.num_workers > 0),
    )
    
    if is_main_process():
        print(
            f"[rank {rank}] dataset={len(dataset)} "
            f"world_size={world_size} "
            f"sampler_len={len(sampler) if sampler is not None else len(dataset)} "
            f"dataloader_len(batches)={len(dataloader)} "
            f"batch_size={args.batch_size}"
        )
        
    print(f"[rank {rank}] sampler_len={len(sampler) if sampler is not None else len(dataset)} "
        f"dataloader_len={len(dataloader)} batch_size={args.batch_size}")


    # Model & processor
    model = AutoModel.from_pretrained(args.hf_repo).to(device).eval()
    processor = AutoVideoProcessor.from_pretrained(args.hf_repo)

    # Per-rank manifest rows
    manifest_rows: List[Dict] = []

    # Inference loop
    if sampler is not None:
        sampler.set_epoch(0)  # needed by PyTorch even if shuffle=False

    progress = tqdm(
        dataloader,
        total=len(dataloader),                    # <-- explicit
        disable=not is_main_process(),
        desc=f"Rank {rank} embedding"
    )
    with torch.inference_mode():
        for videos, metas in progress:
            # Split items that failed during __getitem__
            compute_indices = [i for i, m in enumerate(metas) if not m["failed"]]
            failed_indices  = [i for i, m in enumerate(metas) if m["failed"]]

            # Log failures (no compute)
            for i in failed_indices:
                m = metas[i]
                manifest_rows.append({
                    "video_path": m["video_path"],
                    "index": m["index"],
                    "out_path": m["out_path"],
                    "padded": m["padded"],
                    "failed": True,
                    "error": m["error"],
                    "n_frames": m["n_frames"],
                    "feature_shape": None,
                })

            if len(compute_indices) == 0:
                continue

            batch_videos = [videos[i] for i in compute_indices]  # list of [T,C,H,W] uint8
            batch_metas  = [metas[i]  for i in compute_indices]

            inputs = processor(batch_videos, return_tensors="pt")["pixel_values_videos"]  # [B,T,C,H,W] = [B,64,3,256,256]
            breakpoint()
            inputs = inputs.to(device, non_blocking=True)

            feats = model.get_vision_features(inputs)  # [B, N_tokens, embed_dim] = [B, 8192, 1024]
            breakpoint()
            if isinstance(feats, (list, tuple)):
                raise RuntimeError("Unexpected model output type; expected tensor.")

            # Save per-sample
            for i in range(feats.shape[0]):
                m = batch_metas[i]
                out_path = m["out_path"]
                ensure_parent_dir(out_path)
                features_i = feats[i].detach().cpu()
                torch.save(features_i, out_path)

                manifest_rows.append({
                    "video_path": m["video_path"],
                    "index": m["index"],
                    "out_path": out_path,
                    "padded": m["padded"],
                    "failed": False,
                    "error": "",
                    "n_frames": m["n_frames"],
                    "feature_shape": tuple(features_i.shape),
                })

    # ----------------------- Gather manifests -----------------------
    if dist.is_initialized():
        gathered: List[Optional[List[Dict]]] = [None for _ in range(world_size)]
        dist.all_gather_object(gathered, manifest_rows)
        if is_main_process():
            all_rows: List[Dict] = []
            for chunk in gathered:
                if chunk:
                    all_rows.extend(chunk)
            manifest_df = pd.DataFrame(all_rows)
            manifest_path = os.path.join(args.embeddings_root, "manifest.csv")
            ensure_parent_dir(manifest_path)
            manifest_df.to_csv(manifest_path, index=False)
            print(f"[Rank 0] Saved manifest with {len(manifest_df)} rows to {manifest_path}")
    else:
        # Single-process fallback
        manifest_df = pd.DataFrame(manifest_rows)
        manifest_path = os.path.join(args.embeddings_root, "manifest.csv")
        ensure_parent_dir(manifest_path)
        manifest_df.to_csv(manifest_path, index=False)
        print(f"Saved manifest with {len(manifest_df)} rows to {manifest_path}")

    if dist.is_initialized():
        dist.barrier()
        dist.destroy_process_group()

if __name__ == "__main__":
    # ls vjepa2/scripts
    # torchrun --nproc_per_node=2 generate_kinetics_400_vjepa2_embeddings_distributed.py
    main()
