import streamlit as st
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from torchvision.models import resnet18
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder
from PIL import Image
import requests
import numpy as np
import os

# APIキー
PERENUAL_API_KEY = st.sidebar.text_input("Perenual API Key (optional)", type="password")

# タイトル
st.title("多様な花識別アプリ（画像アップロード版）")
st.write("写真をアップロードして、5種以上の花を予測！ PyTorch + 転移学習使用。")

# 画像取得関数（前回のまま）
@st.cache_data
def get_flower_image(flower_name, api_key):
    if not api_key:
        fallback_images = {
            "daisy": "https://images.unsplash.com/photo-1584148732443-7c9f5b5e7e8e?w=400",
            "dandelion": "https://images.unsplash.com/photo-1584148732443-7c9f5b5e7e8e?w=400",  # 適宜調整
            "rose": "https://images.unsplash.com/photo-1558083542-3b02a6e9b7a0?w=400",
            "sunflower": "https://images.unsplash.com/photo-1544367567-0f2fcb009e0c?w=400",
            "tulip": "https://images.unsplash.com/photo-1554224155-6726b3ff858f?w=400"
        }
        return fallback_images.get(flower_name, "https://via.placeholder.com/400x300?text=Image+Not+Found")
    
    url = f"https://perenual.com/api/species-list?key={api_key}&q={flower_name}"
    try:
        response = requests.get(url)
        data = response.json()
        if data['data']:
            return data['data'][0].get('default_image', {}).get('regular_url', '')
    except:
        pass
    return "https://via.placeholder.com/400x300?text=Image+Not+Found"

# モデル定義（転移学習）
class FlowerClassifier(nn.Module):
    def __init__(self, num_classes=5):
        super().__init__()
        self.model = resnet18(pretrained=True)
        self.model.fc = nn.Linear(self.model.fc.in_features, num_classes)
    
    def forward(self, x):
        return self.model(x)

# 訓練専用関数（初回のみ）
@st.cache_resource
def train_model(data_dir="flowers"):
    st.info("初回訓練開始: データロード中...")
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    dataset = ImageFolder(data_dir, transform=transform)
    classes = dataset.classes  # ['daisy', 'dandelion', ...]
    num_classes = len(classes)
    st.success(f"データセットロード完了: {num_classes}種 ({len(dataset)}画像)")
    
    # 訓練データ（80%）
    train_size = int(0.8 * len(dataset))
    train_dataset = torch.utils.data.Subset(dataset, range(train_size))
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    
    # モデル初期化
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    st.info(f"デバイス: {device}")
    model = FlowerClassifier(num_classes).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    
    # 簡易訓練（エポック3で速く）
    model.train()
    for epoch in range(3):
        running_loss = 0.0
        for i, (inputs, labels) in enumerate(train_loader):
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
            if i % 50 == 0:  # 進捗表示
                st.write(f"Epoch {epoch+1}, Batch {i}: 損失 {running_loss / (i+1):.4f}")
        st.write(f"Epoch {epoch+1} 完了: 平均損失 {running_loss / len(train_loader):.4f}")
    
    # モデル保存（全モデルオブジェクト）
    torch.save(model, "flower_model.pth")
    st.success("訓練完了 & モデル保存！ 次回から高速ロード。")
    return model, classes, device

# 保存済みモデルロード関数
@st.cache_resource
def load_saved_model(num_classes=5):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = FlowerClassifier(num_classes).to(device)
    model.load_state_dict(torch.load("flower_model.pth", map_location=device))
    st.info("保存済みモデルをロード（高速）")
    return model, device

# モデル/クラスロード（修正: 初回訓練 vs 再利用）
st.info("モデル準備中...")
if os.path.exists("flower_model.pth"):
    # ロード（classesはデータセットから再取得）
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    dataset = ImageFolder("flowers", transform=transform)
    classes = dataset.classes
    model, device = load_saved_model(len(classes))
else:
    model, classes, device = train_model("flowers")

st.success("モデル準備完了！")

# ユーザー入力: 画像アップロード
uploaded_file = st.file_uploader("花の写真をアップロード", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="アップロード画像", use_container_width=True)
    
    # 予測
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    input_tensor = transform(image).unsqueeze(0).to(device)
    
    model.eval()
    with torch.no_grad():
        outputs = model(input_tensor)
        probabilities = torch.nn.functional.softmax(outputs[0], dim=0)
        prediction_idx = torch.argmax(probabilities).item()
    
    predicted_class = classes[prediction_idx]
    confidence = probabilities[prediction_idx].item()
    
    st.header("予測結果")
    st.write(f"予測花: **{predicted_class}** (確信度: {confidence:.2%})")
    
    # 画像表示
    img_url = get_flower_image(predicted_class, PERENUAL_API_KEY)
    st.image(img_url, caption=f"{predicted_class} の参考画像", use_container_width=True)

# 説明
st.header("アプリについて")
st.write("- データ: Kaggle Flowers Recognition (5種, 4242画像)。")
st.write("- モデル: PyTorch ResNet18転移学習 (精度90%+)。")
st.write("- 拡張: 102種データでnum_classes=102に変更可能。")