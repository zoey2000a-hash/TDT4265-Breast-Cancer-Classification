import os
import torch
import numpy as np
import pandas as pd
from monai.networks.nets import resnet10
from monai.data import DataLoader, Dataset
from monai.transforms import Compose, ScaleIntensityd, Resized, EnsureTyped
from tqdm import tqdm
import nibabel as nib

# 加载器（强制读取所有 RSH 文件夹）
class ODELIATestDataset(Dataset):
    def __init__(self, data_root, transform=None):
        self.data_root = data_root
        self.transform = transform
        self.patient_list = [f for f in os.listdir(data_root) if f.startswith("Anonymized")]

    def __len__(self):
        return len(self.patient_list)

    def __getitem__(self, index):
        uid = self.patient_list[index]
        patient_dir = os.path.join(self.data_root, uid)

        img_pre  = nib.load(f"{patient_dir}/Pre.nii.gz").get_fdata().astype(np.float32).squeeze()
        img_post1 = nib.load(f"{patient_dir}/Post_1.nii.gz").get_fdata().astype(np.float32).squeeze()
        img_t2   = nib.load(f"{patient_dir}/T2.nii.gz").get_fdata().astype(np.float32).squeeze()

        img_sub1 = img_post1 - img_pre
        image = np.stack([img_pre, img_sub1, img_t2], axis=0)

        data = {"image": image, "uid": uid}
        if self.transform:
            data = self.transform(data)
        return data

def generate_submission():
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # 模型
    model = resnet10(spatial_dims=3, n_input_channels=3, num_classes=3).to(device)
    model.load_state_dict(torch.load("./runs/odelia_breast_classifier/net_epoch=50.pt", map_location=device))
    model.eval()

    # 预处理
    tform = Compose([
        ScaleIntensityd(keys="image"),
        Resized(keys="image", spatial_size=(32,256,256)),
        EnsureTyped(keys="image")
    ])

    # 直接读取你下载好的 RSH 测试集（强制路径）
    data_root = "./data/RSH/data_unilateral/"
    test_ds = ODELIATestDataset(data_root, transform=tform)
    test_loader = DataLoader(test_ds, batch_size=2, shuffle=False)

    results = []
    with torch.no_grad():
        for batch in tqdm(test_loader):
            x = batch["image"].to(device)
            uids = batch["uid"]
            logits = model(x)
            probs = torch.softmax(logits, dim=1).cpu().numpy()

            for uid, p in zip(uids, probs):
                n, b, m = p
                s = n + b + m
                results.append({
                    "ID": uid,
                    "normal": round(n/s, 4),
                    "benign": round(b/s, 4),
                    "malignant": round(m/s, 4)
                })

    # 保存 CSV
    df = pd.DataFrame(results)
    df.to_csv("predictions.csv", index=False)
    print("✅ 成功！生成 predictions.csv，可以直接提交！")

if __name__ == "__main__":
    generate_submission()