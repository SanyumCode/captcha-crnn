# inference.py
from PIL import Image
import torch
from .crnn import CRNN
from .train import greedy_decode

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = CRNN().to(device)
ck = torch.load('checkpoint_best.pth', map_location=device)
model.load_state_dict(ck['model_state'])
model.eval()

def infer_image(img_path, imgH=32):
    img = Image.open(img_path).convert('L')
    w, h = img.size
    new_w = max(1, int(w * (imgH / h)))
    img = img.resize((new_w, imgH), Image.BILINEAR)
    import numpy as np
    arr = np.array(img).astype(np.float32) / 255.0
    tensor = torch.from_numpy(arr).unsqueeze(0).unsqueeze(0).to(device)  # 1x1xH xW
    with torch.no_grad():
        log_probs = model(tensor)  # T x 1 x C
    preds = greedy_decode(log_probs)
    return preds[0]

print(infer_image('some_test.png'))
