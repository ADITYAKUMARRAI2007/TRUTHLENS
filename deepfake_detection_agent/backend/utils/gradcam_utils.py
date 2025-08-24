# backend/utils/gradcam_utils.py
import cv2
import numpy as np
import torch

class VitAttentionRollout:
    def __init__(self, discard_ratio=0.0):
        self.discard_ratio = discard_ratio

    def generate(self, model, inputs):
        # Forward pass with attention
        outputs = model(pixel_values=inputs, output_attentions=True)
        attentions = outputs.attentions  # tuple of (layers, batch, heads, seq_len, seq_len)

        attn = torch.stack(attentions).mean(2)  # average heads
        rollout = torch.eye(attn.size(-1))
        for i in range(attn.size(0)):
            attn_i = attn[i, 0]
            attn_i = attn_i / attn_i.sum(dim=-1, keepdim=True)
            rollout = rollout @ attn_i
        cam = rollout[0, 1:].mean(0).reshape(1, int(inputs.shape[-1]**0.5), -1)
        cam = cam.squeeze().cpu().numpy()
        cam = (cam - cam.min()) / (cam.max() - cam.min() + 1e-8)
        return cam

def overlay_heatmap(frame, cam, alpha=0.5):
    cam_resized = cv2.resize(cam, (frame.shape[1], frame.shape[0]))
    heatmap = cv2.applyColorMap(np.uint8(255 * cam_resized), cv2.COLORMAP_JET)
    overlay = cv2.addWeighted(frame, 1 - alpha, heatmap, alpha, 0)
    return overlay
