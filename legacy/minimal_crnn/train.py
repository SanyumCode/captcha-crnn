# train.py
import torch
from torch import optim
from torch.nn import CTCLoss
from torch.utils.data import DataLoader
from tqdm import tqdm
from .crnn import CRNN
from .dataset_and_dataloader import CaptchaDataset, collate_fn, idx2char

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# 超参数（根据显存调节）
imgH = 32
batch_size = 64
lr = 1e-3
epochs = 30

# paths
images_dir = 'dataset/images'
labels_csv = 'dataset/labels.csv'

# dataset & dataloader
transform = None  # 可用 albumentations 做增强
train_ds = CaptchaDataset(images_dir, labels_csv, img_height=imgH, transforms=transform)
train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, collate_fn=collate_fn, num_workers=4)

# model
model = CRNN(imgH=imgH).to(device)
optimizer = optim.Adam(model.parameters(), lr=lr)
ctc_loss = CTCLoss(blank=0, reduction='mean', zero_infinity=True)

def train_epoch(model, loader, optimizer):
    model.train()
    total_loss = 0.0
    pbar = tqdm(loader)
    for imgs, targets, target_lengths in pbar:
        imgs = imgs.to(device)  # B x 1 x H x W
        targets = targets.to(device)
        target_lengths = target_lengths.to(device)
        optimizer.zero_grad()
        log_probs = model(imgs)  # T x B x C
        T, N, C = log_probs.size()
        input_lengths = torch.full(size=(N,), fill_value=T, dtype=torch.long).to(device)
        loss = ctc_loss(log_probs, targets, input_lengths, target_lengths)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
        pbar.set_description(f"loss: {loss.item():.4f}")
    return total_loss / len(loader)

# 简单验证：用 greedy decode 计算验证码整体识别率（batch 内）
def greedy_decode(log_probs):
    # log_probs: T x B x C (tensor)
    probs = log_probs.detach().cpu()
    _, max_idx = probs.max(dim=2)  # T x B
    max_idx = max_idx.transpose(0, 1).numpy()  # B x T
    results = []
    for seq in max_idx:
        # collapse repeats and remove blanks (0)
        prev = 0
        out = []
        for idx in seq:
            if idx != prev and idx != 0:
                out.append(idx)
            prev = idx
        results.append(''.join(idx2char[i] for i in out))
    return results

def validate(model, loader, num_samples=200):
    model.eval()
    total = 0
    correct = 0
    with torch.no_grad():
        for imgs, targets, target_lengths in loader:
            imgs = imgs.to(device)
            log_probs = model(imgs)  # T x B x C
            preds = greedy_decode(log_probs)
            # reconstruct labels from targets and lengths
            # targets is concatenated; we need to split by target_lengths
            labels = []
            pos = 0
            for l in target_lengths:
                l = int(l)
                labels.append(''.join(idx2char[int(x)] for x in targets[pos:pos+l].tolist()))
                pos += l
            for p, gt in zip(preds, labels):
                total += 1
                if p == gt:
                    correct += 1
            if total >= num_samples:
                break
    return correct / total if total>0 else 0.0

# 训练主循环
best_acc = 0.0
for epoch in range(1, epochs+1):
    avg_loss = train_epoch(model, train_loader, optimizer)
    # 使用训练集快速验证，建议单独准备验证集
    acc = validate(model, train_loader, num_samples=500)
    print(f"Epoch {epoch}, Loss {avg_loss:.4f}, Acc {acc:.4f}")
    # 保存 checkpoint
    if acc > best_acc:
        best_acc = acc
        torch.save({'model_state': model.state_dict(), 'optimizer': optimizer.state_dict()},
                   f'checkpoint_best.pth')
