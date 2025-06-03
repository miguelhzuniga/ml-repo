import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
from PIL import Image
import os
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, jaccard_score
from sklearn.utils import resample
import matplotlib.pyplot as plt
from tqdm import tqdm
import joblib

# Configuración
IMG_DIR = 'dataset/images'
MASK_DIR = 'dataset/annotations/trimaps'
IMG_SIZE = (96, 96)
PATCH_SIZE = 5

# Hiperparámetros del modelo y entrenamiento
HIDDEN1 = 235
HIDDEN2 = 211
HIDDEN3 = 124
LR = 0.0007533779226005483
BATCH_SIZE = 128

# Función para extraer parches
def extract_patch_features_with_coords(image_array, kernel_size=5):
    pad = kernel_size // 2
    H, W, C = image_array.shape
    padded = np.pad(image_array, ((pad, pad), (pad, pad), (0, 0)), mode='reflect')
    features = []
    for i in range(pad, H + pad):
        for j in range(pad, W + pad):
            patch = padded[i-pad:i+pad+1, j-pad:j+pad+1, :].flatten()
            x_norm = (j - pad) / (W - 1)
            y_norm = (i - pad) / (H - 1)
            features.append(np.concatenate([patch, [x_norm, y_norm]]))
    return np.array(features)

# Carga de datos
def load_dataset_with_context(max_images=None):
    X, y = [], []
    count = 0
    for fname in tqdm(os.listdir(IMG_DIR)):
        if not fname.endswith('.jpg'):
            continue
        img_path = os.path.join(IMG_DIR, fname)
        mask_path = os.path.join(MASK_DIR, fname.replace('.jpg', '.png'))
        if not os.path.exists(mask_path):
            continue
        image = Image.open(img_path).resize(IMG_SIZE).convert('RGB')
        mask = Image.open(mask_path).resize(IMG_SIZE)
        image_np = np.array(image) / 255.0
        mask_np = np.array(mask)
        mask_binary = (mask_np == 3).astype(int)
        X_patch = extract_patch_features_with_coords(image_np, kernel_size=PATCH_SIZE)
        y_patch = mask_binary.flatten()
        X.append(X_patch)
        y.append(y_patch)
        count += 1
        if max_images and count >= max_images:
            break
    return np.vstack(X), np.hstack(y)

X, y = load_dataset_with_context(max_images=None)
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Balanceo
X_combined = np.hstack((X_scaled, y.reshape(-1, 1)))
class0 = X_combined[X_combined[:, -1] == 0]
class1 = X_combined[X_combined[:, -1] == 1]
class0_down = resample(class0, replace=False, n_samples=len(class1), random_state=42)
balanced = np.vstack((class0_down, class1))
np.random.shuffle(balanced)

X_balanced = balanced[:, :-1]
y_balanced = balanced[:, -1]

X_train, X_test, y_train, y_test = train_test_split(X_balanced, y_balanced, test_size=0.2, random_state=42)

# Dataset y DataLoader
class PatchDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.float32)

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

train_loader = DataLoader(PatchDataset(X_train, y_train), batch_size=BATCH_SIZE, shuffle=True)
test_loader = DataLoader(PatchDataset(X_test, y_test), batch_size=BATCH_SIZE)

# Definición del modelo
class MLP(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, HIDDEN1),
            nn.ReLU(),
            nn.Linear(HIDDEN1, HIDDEN2),
            nn.ReLU(),
            nn.Linear(HIDDEN2, HIDDEN3),
            nn.ReLU(),
            nn.Linear(HIDDEN3, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.net(x).squeeze()

model = MLP(X_train.shape[1])
optimizer = optim.Adam(model.parameters(), lr=LR)
criterion = nn.BCELoss()

# Entrenamiento
losses = []
for epoch in range(30):
    model.train()
    running_loss = 0.0
    for xb, yb in train_loader:
        optimizer.zero_grad()
        preds = model(xb)
        loss = criterion(preds, yb)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
    avg_loss = running_loss / len(train_loader)
    losses.append(avg_loss)
    print(f"Epoch {epoch+1}, Loss: {avg_loss:.4f}")

# Guardar curva de pérdida
plt.plot(losses)
plt.title("Curva de pérdida")
plt.xlabel("Épocas")
plt.ylabel("Loss")
plt.grid(True)
plt.savefig("loss_curve_pytorch.png")
plt.show()

# Evaluación
model.eval()
y_preds = []
with torch.no_grad():
    for xb, _ in test_loader:
        preds = model(xb)
        y_preds.extend((preds > 0.5).int().cpu().numpy())
print(classification_report(y_test, y_preds))


def save_visual_example(n=10, output_dir="outputs"):
    os.makedirs(output_dir, exist_ok=True)
    img_files = [f for f in os.listdir(IMG_DIR) if f.endswith(".jpg")]
    selected_files = np.random.choice(img_files, size=n, replace=False)
    iou_scores = []

    for fname in selected_files:
        img_path = os.path.join(IMG_DIR, fname)
        mask_path = os.path.join(MASK_DIR, fname.replace(".jpg", ".png"))

        image = Image.open(img_path).resize(IMG_SIZE).convert("RGB")
        mask = Image.open(mask_path).resize(IMG_SIZE)
        image_np = np.array(image) / 255.0
        mask_np = np.array(mask)
        mask_binary = (mask_np == 3).astype(np.uint8)

        patch_features = extract_patch_features_with_coords(image_np, kernel_size=PATCH_SIZE)
        patch_scaled = scaler.transform(patch_features)
        patch_tensor = torch.tensor(patch_scaled, dtype=torch.float32)
        
        with torch.no_grad():
            prediction = model(patch_tensor)
        pred_mask = (prediction > 0.5).int().numpy().reshape(IMG_SIZE)

        # IoU
        iou = jaccard_score(mask_binary.flatten(), pred_mask.flatten(), zero_division=0)
        iou_scores.append(iou)

        # Visualización
        fig, axs = plt.subplots(1, 3, figsize=(12, 4))
        axs[0].imshow(image_np)
        axs[0].set_title("Imagen original")
        axs[0].axis("off")

        axs[1].imshow(mask_binary, cmap="gray")
        axs[1].set_title("Máscara real")
        axs[1].axis("off")

        axs[2].imshow(pred_mask, cmap="gray")
        axs[2].set_title(f"Predicción (IoU={iou:.2f})")
        axs[2].axis("off")

        plt.tight_layout()
        output_path = os.path.join(output_dir, fname.replace(".jpg", f"_iou_{iou:.2f}.png"))
        plt.savefig(output_path)
        plt.close()

    print(f"📊 IoU promedio en {n} imágenes: {np.mean(iou_scores):.4f}")


save_visual_example(n=10)
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
from PIL import Image
import os
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, jaccard_score
from sklearn.utils import resample
import matplotlib.pyplot as plt
from tqdm import tqdm
import joblib

# Configuración
IMG_DIR = 'reduced/images'
MASK_DIR = 'reduced/annotations/trimaps'
IMG_SIZE = (96, 96)
PATCH_SIZE = 5

# Hiperparámetros del modelo y entrenamiento
HIDDEN1 = 235
HIDDEN2 = 211
HIDDEN3 = 124
LR = 0.0007533779226005483
BATCH_SIZE = 128

# Función para extraer parches
def extract_patch_features_with_coords(image_array, kernel_size=5):
    pad = kernel_size // 2
    H, W, C = image_array.shape
    padded = np.pad(image_array, ((pad, pad), (pad, pad), (0, 0)), mode='reflect')
    features = []
    for i in range(pad, H + pad):
        for j in range(pad, W + pad):
            patch = padded[i-pad:i+pad+1, j-pad:j+pad+1, :].flatten()
            x_norm = (j - pad) / (W - 1)
            y_norm = (i - pad) / (H - 1)
            features.append(np.concatenate([patch, [x_norm, y_norm]]))
    return np.array(features)

# Carga de datos
def load_dataset_with_context(max_images=None):
    X, y = [], []
    count = 0
    for fname in tqdm(os.listdir(IMG_DIR)):
        if not fname.endswith('.jpg'):
            continue
        img_path = os.path.join(IMG_DIR, fname)
        mask_path = os.path.join(MASK_DIR, fname.replace('.jpg', '.png'))
        if not os.path.exists(mask_path):
            continue
        image = Image.open(img_path).resize(IMG_SIZE).convert('RGB')
        mask = Image.open(mask_path).resize(IMG_SIZE)
        image_np = np.array(image) / 255.0
        mask_np = np.array(mask)
        mask_binary = (mask_np == 3).astype(int)
        X_patch = extract_patch_features_with_coords(image_np, kernel_size=PATCH_SIZE)
        y_patch = mask_binary.flatten()
        X.append(X_patch)
        y.append(y_patch)
        count += 1
        if max_images and count >= max_images:
            break
    return np.vstack(X), np.hstack(y)

X, y = load_dataset_with_context(max_images=None)
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Balanceo
X_combined = np.hstack((X_scaled, y.reshape(-1, 1)))
class0 = X_combined[X_combined[:, -1] == 0]
class1 = X_combined[X_combined[:, -1] == 1]
class0_down = resample(class0, replace=False, n_samples=len(class1), random_state=42)
balanced = np.vstack((class0_down, class1))
np.random.shuffle(balanced)

X_balanced = balanced[:, :-1]
y_balanced = balanced[:, -1]

X_train, X_test, y_train, y_test = train_test_split(X_balanced, y_balanced, test_size=0.2, random_state=42)

# Dataset y DataLoader
class PatchDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.float32)

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

train_loader = DataLoader(PatchDataset(X_train, y_train), batch_size=BATCH_SIZE, shuffle=True)
test_loader = DataLoader(PatchDataset(X_test, y_test), batch_size=BATCH_SIZE)

# Definición del modelo
class MLP(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, HIDDEN1),
            nn.ReLU(),
            nn.Linear(HIDDEN1, HIDDEN2),
            nn.ReLU(),
            nn.Linear(HIDDEN2, HIDDEN3),
            nn.ReLU(),
            nn.Linear(HIDDEN3, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.net(x).squeeze()

model = MLP(X_train.shape[1])
optimizer = optim.Adam(model.parameters(), lr=LR)
criterion = nn.BCELoss()

# Entrenamiento
losses = []
for epoch in range(30):
    model.train()
    running_loss = 0.0
    for xb, yb in train_loader:
        optimizer.zero_grad()
        preds = model(xb)
        loss = criterion(preds, yb)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
    avg_loss = running_loss / len(train_loader)
    losses.append(avg_loss)
    print(f"Epoch {epoch+1}, Loss: {avg_loss:.4f}")

# Guardar curva de pérdida
plt.plot(losses)
plt.title("Curva de pérdida")
plt.xlabel("Épocas")
plt.ylabel("Loss")
plt.grid(True)
plt.savefig("loss_curve_pytorch.png")
plt.show()

# Evaluación
model.eval()
y_preds = []
with torch.no_grad():
    for xb, _ in test_loader:
        preds = model(xb)
        y_preds.extend((preds > 0.5).int().cpu().numpy())
print(classification_report(y_test, y_preds))


def save_visual_example(n=10, output_dir="outputs"):
    os.makedirs(output_dir, exist_ok=True)
    img_files = [f for f in os.listdir(IMG_DIR) if f.endswith(".jpg")]
    selected_files = np.random.choice(img_files, size=n, replace=False)
    iou_scores = []

    for fname in selected_files:
        img_path = os.path.join(IMG_DIR, fname)
        mask_path = os.path.join(MASK_DIR, fname.replace(".jpg", ".png"))

        image = Image.open(img_path).resize(IMG_SIZE).convert("RGB")
        mask = Image.open(mask_path).resize(IMG_SIZE)
        image_np = np.array(image) / 255.0
        mask_np = np.array(mask)
        mask_binary = (mask_np == 3).astype(np.uint8)

        patch_features = extract_patch_features_with_coords(image_np, kernel_size=PATCH_SIZE)
        patch_scaled = scaler.transform(patch_features)
        patch_tensor = torch.tensor(patch_scaled, dtype=torch.float32)
        
        with torch.no_grad():
            prediction = model(patch_tensor)
        pred_mask = (prediction > 0.5).int().numpy().reshape(IMG_SIZE)

        # IoU
        iou = jaccard_score(mask_binary.flatten(), pred_mask.flatten(), zero_division=0)
        iou_scores.append(iou)

        # Visualización
        fig, axs = plt.subplots(1, 3, figsize=(12, 4))
        axs[0].imshow(image_np)
        axs[0].set_title("Imagen original")
        axs[0].axis("off")

        axs[1].imshow(mask_binary, cmap="gray")
        axs[1].set_title("Máscara real")
        axs[1].axis("off")

        axs[2].imshow(pred_mask, cmap="gray")
        axs[2].set_title(f"Predicción (IoU={iou:.2f})")
        axs[2].axis("off")

        plt.tight_layout()
        output_path = os.path.join(output_dir, fname.replace(".jpg", f"_iou_{iou:.2f}.png"))
        plt.savefig(output_path)
        plt.close()

    print(f"📊 IoU promedio en {n} imágenes: {np.mean(iou_scores):.4f}")


save_visual_example(n=10)
