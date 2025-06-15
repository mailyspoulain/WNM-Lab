import os
import torch
import torch.nn as nn
from torchvision import datasets, models, transforms
from torch.utils.data import DataLoader
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import (
    confusion_matrix,
    classification_report,
    roc_auc_score,
    roc_curve,
    precision_score,
    recall_score
)

# === CONFIGURATION ===
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
BATCH_SIZE = 32
# Mets ici le chemin de ton dataset test
TEST_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../chest_xray/test'))
# Mets ici le chemin vers ton modèle entraîné
MODEL_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), '../models/densenet121_chest_xray_model.pth'))
CLASS_NAMES = ['NORMAL', 'PNEUMONIA']

# === TRANSFORMATIONS ===
test_transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

# === DATA ===
test_dataset = datasets.ImageFolder(TEST_DIR, transform=test_transform)
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

# === MODELE ===
model = models.densenet121(weights=None)
model.classifier = nn.Sequential(
    nn.Dropout(0.5),
    nn.Linear(1024, 512),
    nn.ReLU(),
    nn.Dropout(0.3),
    nn.Linear(512, 2)
)
model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
model = model.to(DEVICE)
model.eval()

all_labels = []
all_preds = []
all_probs = []

with torch.no_grad():
    for inputs, labels in test_loader:
        inputs = inputs.to(DEVICE)
        outputs = model(inputs)
        probs = torch.softmax(outputs, dim=1)
        preds = torch.argmax(probs, dim=1)
        all_labels.extend(labels.cpu().numpy())
        all_preds.extend(preds.cpu().numpy())
        all_probs.extend(probs[:, 1].cpu().numpy())  # proba classe "PNEUMONIA"

all_labels = np.array(all_labels)
all_preds = np.array(all_preds)
all_probs = np.array(all_probs)

# === METRIQUES ===
precision = precision_score(all_labels, all_preds)
recall = recall_score(all_labels, all_preds)
auc = roc_auc_score(all_labels, all_probs)
cm = confusion_matrix(all_labels, all_preds)
report = classification_report(all_labels, all_preds, target_names=CLASS_NAMES)

print("\n=== ÉVALUATION DU MODÈLE ===")
print(f"Précision  : {precision:.3f}")
print(f"Rappel     : {recall:.3f}")
print(f"ROC-AUC    : {auc:.3f}")
print("Matrice de confusion :\n", cm)
print("\nClassification report:\n", report)

# === COURBE ROC ===
fpr, tpr, thresholds = roc_curve(all_labels, all_probs)
plt.figure(figsize=(6,6))
plt.plot(fpr, tpr, label=f'ROC curve (AUC = {auc:.3f})')
plt.plot([0,1], [0,1], 'k--', label='Random')
plt.xlabel('Taux de faux positifs')
plt.ylabel('Taux de vrais positifs')
plt.title('Courbe ROC')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig(os.path.join(os.path.dirname(__file__), "roc_curve.png"))
plt.show()

# === MATRICE DE CONFUSION ===
plt.figure(figsize=(5,4))
plt.imshow(cm, cmap='Blues')
plt.title("Matrice de confusion")
plt.xlabel("Prédictions")
plt.ylabel("Réel")
plt.xticks([0,1], CLASS_NAMES)
plt.yticks([0,1], CLASS_NAMES)
for i in range(2):
    for j in range(2):
        plt.text(j, i, cm[i, j], ha='center', va='center', color='red', fontsize=16)
plt.tight_layout()
plt.savefig(os.path.join(os.path.dirname(__file__), "confusion_matrix.png"))
plt.show()
