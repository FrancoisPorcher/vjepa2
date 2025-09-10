# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import json
import os
import subprocess

import numpy as np
import torch
from torch import linalg as LA
import torch.nn.functional as F
from decord import VideoReader
from transformers import AutoModel, AutoVideoProcessor

from src.models.attentive_pooler import AttentiveClassifier


def load_pretrained_vjepa_classifier_weights(model, pretrained_weights):
    # Load weights of the VJEPA2 classifier
    # The PyTorch state_dict is already preprocessed to have the right key names
    pretrained_dict = torch.load(pretrained_weights, weights_only=True, map_location="cpu")["classifiers"][0]
    pretrained_dict = {k.replace("module.", ""): v for k, v in pretrained_dict.items()}
    msg = model.load_state_dict(pretrained_dict, strict=False)
    print("Pretrained weights found at {} and loaded with msg: {}".format(pretrained_weights, msg))


def get_video():
    vr = VideoReader("sample_video.mp4")
    # choosing some frames here, you can define more complex sampling strategy
    frame_idx = np.arange(0, 128, 2)
    video = vr.get_batch(frame_idx).asnumpy()
    return video


def forward_vjepa_video(model_hf, hf_transform):
    # Run a sample inference with VJEPA using the HuggingFace model
    with torch.inference_mode():
        # Read and pre-process the image
        video = get_video()  # T x H x W x C
        video = torch.from_numpy(video).permute(0, 3, 1, 2)  # T x C x H x W
        
        x_hf = hf_transform(video, return_tensors="pt")["pixel_values_videos"].to("cuda")
        # Extract the patch-wise features from the last layer
        out_patch_features_hf = model_hf.get_vision_features(x_hf) # shape [B, N_tokens, Embed_dim]
        

    return out_patch_features_hf


def get_vjepa_video_classification_results(classifier, out_patch_features):
    SOMETHING_SOMETHING_V2_CLASSES = json.load(open("ssv2_classes.json", "r"))

    with torch.inference_mode():
        out_classifier = classifier(out_patch_features)

    print(f"Classifier output shape: {out_classifier.shape}")

    print("Top 5 predicted class names:")
    top5_indices = out_classifier.topk(5).indices[0]
    top5_probs = F.softmax(out_classifier.topk(5).values[0]) * 100.0  # convert to percentage
    for idx, prob in zip(top5_indices, top5_probs):
        str_idx = str(idx.item())
        print(f"{SOMETHING_SOMETHING_V2_CLASSES[str_idx]} ({prob}%)")

    return


def run_sample_inference():
    # HuggingFace model repo name
    hf_model_name = (
        "facebook/vjepa2-vitl-fpc64-256"  # Replace with your favored model, e.g. facebook/vjepa2-vitg-fpc64-384
    )
    sample_video_path = "sample_video.mp4"
    # Download the video if not yet downloaded to local path
    if not os.path.exists(sample_video_path):
        video_url = "https://huggingface.co/datasets/nateraw/kinetics-mini/resolve/main/val/bowling/-WH-lxmGJVY_000005_000015.mp4"
        command = ["wget", video_url, "-O", sample_video_path]
        subprocess.run(command)
        print("Downloading video")

    # Initialize the HuggingFace model, load pretrained weights
    model_hf = AutoModel.from_pretrained(hf_model_name)
    model_hf.cuda().eval()

    # Build HuggingFace preprocessing transform
    hf_transform = AutoVideoProcessor.from_pretrained(hf_model_name)
    
    breakpoint()

    # Inference on video
    out_patch_features = forward_vjepa_video(model_hf, hf_transform)

    # Compute and display norms of the embeddings
    l1_norm = LA.vector_norm(out_patch_features, ord=1).item()
    l2_norm = LA.vector_norm(out_patch_features, ord=2).item()
    print(f"L1 norm of embeddings: {l1_norm}")
    print(f"L2 norm of embeddings: {l2_norm}")

    print(
        f"""
        Inference results on video:
        HuggingFace output shape: {out_patch_features.shape}
        """
    )

    # Initialize the classifier
    classifier_model_path = "YOUR_ATTENTIVE_PROBE_PATH"
    classifier = (
        AttentiveClassifier(
            embed_dim=model_hf.config.hidden_size, num_heads=16, depth=4, num_classes=174
        ).cuda().eval()
    )
    load_pretrained_vjepa_classifier_weights(classifier, classifier_model_path)

    # Download SSV2 classes if not already present
    ssv2_classes_path = "ssv2_classes.json"
    if not os.path.exists(ssv2_classes_path):
        command = [
            "wget",
            "https://huggingface.co/datasets/huggingface/label-files/resolve/d79675f2d50a7b1ecf98923d42c30526a51818e2/"
            "something-something-v2-id2label.json",
            "-O",
            "ssv2_classes.json",
        ]
        subprocess.run(command)
        print("Downloading SSV2 classes")

    get_vjepa_video_classification_results(classifier, out_patch_features)


if __name__ == "__main__":
    # Run with: `python -m notebooks.vjepa2_demo`
    run_sample_inference()
