import os
import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# 尝试导入绘图库，如果缺失则提示
try:
    import seaborn as sns
except ImportError:
    print("[-] Missing 'seaborn'. Please run: pip install seaborn")
    import sys
    sys.exit(1)

from datasets import load_dataset
from monai.networks.nets import resnet10
from monai.data import DataLoader, Dataset
from monai.transforms import Compose, ScaleIntensityd, Resized, EnsureTyped
from sklearn.metrics import classification_report, confusion_matrix

# ==============================================================================
# 1. Dataset Adapter (与训练代码逻辑对齐)
# ==============================================================================
class ODELIAValDataset(Dataset):
    def __init__(self, hf_dataset, transform=None):
        self.hf_dataset = hf_dataset
        self.transform = transform

    def __len__(self):
        return len(self.hf_dataset)

    def __getitem__(self, index):
        item = self.hf_dataset[index]
        
        # 提取图像模态 [cite: 2, 3]
        img_pre = np.array(item['Image_Pre'], dtype=np.float32).squeeze()
        img_post1 = np.array(item['Image_Post_1'], dtype=np.float32).squeeze()
        img_t2 = np.array(item['Image_T2'], dtype=np.float32).squeeze()
        
        # 减影图逻辑与训练一致 [cite: 3]
        img_sub1 = img_post1 - img_pre
        
        # 堆叠为 (3, Z, H, W) [cite: 3]
        image_stacked = np.stack([img_pre, img_sub1, img_t2], axis=0)
        
        label = int(item['Lesion'])
        
        data_dict = {
            "image": image_stacked,
            "label": np.array(label, dtype=np.int64)
        }
        
        if self.transform:
            data_dict = self.transform(data_dict)
            
        return data_dict

# ==============================================================================
# 2. 评估核心函数
# ==============================================================================
def evaluate_model():
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    print(f"[*] Using device: {device}")

    # --- A. 加载模型权重 ---
    # 请确保路径与你 runs 文件夹下的文件名一致 [cite: 14, 15]
    weight_path = "./runs/odelia_breast_classifier/net_epoch=50.pt" 
    
    model = resnet10(spatial_dims=3, n_input_channels=3, num_classes=3).to(device)
    if os.path.exists(weight_path):
        model.load_state_dict(torch.load(weight_path, map_location=device))
        print(f"[+] Loaded weights from {weight_path}")
    else:
        print(f"[!] Error: Weight file not found at {weight_path}")
        return

    model.eval()

    # --- B. 预处理管道 (必须与训练严格对齐) ---
    val_tform = Compose([
        ScaleIntensityd(keys="image"),
        # 使用 Resized 确保维度为 (32, 256, 256) [cite: 6]
        Resized(keys="image", spatial_size=(32, 256, 256)), 
        EnsureTyped(keys=("image", "label"))
    ])

    # --- C. 加载验证集 ---
    print("[*] Loading validation split from Hugging Face...")
    hf_val_dataset = load_dataset("ODELIA-AI/ODELIA-Challenge-2025", name="unilateral", split="val")
    val_ds = ODELIAValDataset(hf_val_dataset, transform=val_tform)
    
    # 定义 val_loader [cite: 8]
    val_loader = DataLoader(val_ds, batch_size=8, shuffle=False, num_workers=4)

    # --- D. 执行预测 ---
    y_true = []
    y_pred = []
    
    print("[*] Running inference on validation set...")
    with torch.no_grad():
        for batch in val_loader:
            images = batch["image"].to(device)
            labels = batch["label"].cpu().numpy()
            
            outputs = model(images)
            preds = torch.argmax(outputs, dim=1).cpu().numpy()
            
            y_true.extend(labels)
            y_pred.extend(preds)

    # --- E. 结果分析与绘图 ---
    target_names = ['Normal', 'Benign', 'Malignant']
    
    # 1. 打印分类报告
    print("\n" + "="*60)
    print("📊 Detailed Classification Report (Validation Set)")
    print("="*60)
    print(classification_report(y_true, y_pred, target_names=target_names))
    print("="*60)

    # 2. 绘制混淆矩阵热力图
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=target_names, yticklabels=target_names)
    plt.title('Confusion Matrix - Breast Cancer Classification')
    plt.ylabel('Actual Label')
    plt.xlabel('Predicted Label')
    
    save_path = 'confusion_matrix.png'
    plt.savefig(save_path)
    print(f"\n[+] Success! Confusion matrix saved as '{save_path}'")
    print("[+] You can now use these results in your presentation.")

if __name__ == "__main__":
    evaluate_model()