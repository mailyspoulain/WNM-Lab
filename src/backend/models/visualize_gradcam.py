import os
import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import cv2

# === CONFIG ===
MODEL_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), "densenet121_chest_xray_model.pth"))
IMG_PATH = "C:/Users/willi/Documents/ECOLE/medai/WNM-Lab/src/chest_xray/test/NORMAL/IM-0099-0001.jpeg"

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
IMG_SIZE = 224

# === Prétraitement ===
transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize([0.485,0.456,0.406], [0.229,0.224,0.225])
])

def preprocess(img_path):
    img = Image.open(img_path).convert("RGB")
    return transform(img).unsqueeze(0), img

# === Charger le modèle ===
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

# === GradCAM ===
class GradCAM:
    def __init__(self, model, target_layer):
        self.model = model
        self.target_layer = target_layer
        self.gradients = None
        self.activations = None
        self.hook_handles = []
        self._register_hooks()

    def _register_hooks(self):
        def backward_hook(module, grad_input, grad_output):
            self.gradients = grad_output[0].detach()
        def forward_hook(module, input, output):
            self.activations = output.detach()
        self.hook_handles.append(self.target_layer.register_forward_hook(forward_hook))
        self.hook_handles.append(self.target_layer.register_backward_hook(backward_hook))

    def __call__(self, x, class_idx=None):
        output = self.model(x)
        if class_idx is None:
            class_idx = output.argmax().item()
        self.model.zero_grad()
        loss = output[0, class_idx]
        loss.backward()
        gradients = self.gradients[0]      # [C, H, W]
        activations = self.activations[0]  # [C, H, W]
        weights = gradients.mean(dim=(1,2))
        cam = torch.zeros(activations.shape[1:], dtype=torch.float32)
        for i, w in enumerate(weights):
            cam += w * activations[i]
        cam = torch.relu(cam)
        cam = cam - cam.min()
        cam = cam / (cam.max() + 1e-8)
        cam = cam.cpu().numpy()
        return cam

    def close(self):
        for handle in self.hook_handles:
            handle.remove()

# === Générer une heatmap GradCAM ===
def show_gradcam(img_path, model, layer_name="features.denseblock4"):
    input_tensor, pil_img = preprocess(img_path)
    input_tensor = input_tensor.to(DEVICE)

    # Prendre le dernier bloc dense de DenseNet121 (fonctionne avec torchvision >=0.12)
    target_layer = dict([*model.named_modules()])["features.denseblock4"]
    gradcam = GradCAM(model, target_layer)
    cam = gradcam(input_tensor)
    gradcam.close()

    # Redimensionner le CAM sur l’image d’origine
    cam = cv2.resize(cam, pil_img.size)
    heatmap = (cam * 255).astype(np.uint8)
    heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
    img = np.array(pil_img)
    overlay = cv2.addWeighted(img, 0.6, heatmap, 0.4, 0)

    # Affichage
    plt.figure(figsize=(10,5))
    plt.subplot(1,3,1)
    plt.title("Image d'origine")
    plt.axis('off')
    plt.imshow(pil_img)
    plt.subplot(1,3,2)
    plt.title("Heatmap Grad-CAM")
    plt.axis('off')
    plt.imshow(heatmap)
    plt.subplot(1,3,3)
    plt.title("Overlay")
    plt.axis('off')
    plt.imshow(overlay)
    plt.tight_layout()
    plt.savefig("gradcam_result.png")
    plt.show()

if __name__ == "__main__":
    show_gradcam(IMG_PATH, model)
