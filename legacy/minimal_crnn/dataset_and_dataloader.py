# dataset_and_dataloader.py
import os
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as T
import numpy as np

# 自定义字符集（调整为你自己的）
CHARS = "0123456789abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ"
# blank index = 0 for CTC; other chars indexed from 1..len(CHARS)
char2idx = {c: i+1 for i, c in enumerate(CHARS)}
idx2char = {i+1: c for i, c in enumerate(CHARS)}

def encode_label(s: str):
    return [char2idx[c] for c in s]

def decode_indexes(indexes):
    return ''.join(idx2char[i] for i in indexes)

class CaptchaDataset(Dataset):
    def __init__(self, images_dir, labels_csv, img_height=32, transforms=None):
        self.images_dir = images_dir
        self.samples = []
        with open(labels_csv, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                fn, label = line.split(',', 1)
                self.samples.append((fn, label))
        self.img_height = img_height
        self.transforms = transforms

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        fn, label = self.samples[idx]
        path = os.path.join(self.images_dir, fn)
        img = Image.open(path).convert('L')  # 灰度
        w, h = img.size
        new_h = self.img_height
        new_w = max(1, int(w * (new_h / h)))
        img = img.resize((new_w, new_h), Image.BILINEAR)
        if self.transforms:
            img = self.transforms(np.array(img))
        else:
            img = np.array(img).astype(np.float32) / 255.0
        # shape: H x W
        return torch.from_numpy(img).unsqueeze(0), encode_label(label), len(label)

def collate_fn(batch):
    """
    batch: list of (img_tensor: 1 x H x W, label_idx_list, label_len)
    pad images to max width in batch
    returns:
      images: B x 1 x H x W_max (float)
      targets: concatenated targets 1D (LongTensor)
      input_lengths: [seq_len_per_sample] (for CTC: T per sample)
      target_lengths: [label_len per sample]
    """
    imgs, labels, label_lens = zip(*batch)
    H = imgs[0].shape[1]
    widths = [i.shape[2] for i in imgs]
    max_w = max(widths)
    B = len(imgs)
    images_batch = torch.zeros(B, 1, H, max_w, dtype=torch.float32)
    for i, img in enumerate(imgs):
        w = img.shape[2]
        images_batch[i, :, :, :w] = img
    # Convert labels to a single vector for CTCLoss
    targets = []
    for l in labels:
        targets.extend(l)
    targets = torch.LongTensor(targets)
    # input_lengths: after CNN->FeatureMap->AdaptivePool height->sequence length = W' (time steps)
    # We'll compute seq_len as max_w // some reduction, but simpler: we will compute T from model output at runtime.
    target_lengths = torch.IntTensor(label_lens)
    return images_batch, targets, target_lengths
