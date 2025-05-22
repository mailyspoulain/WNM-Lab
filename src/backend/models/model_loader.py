import torch
import torch.nn as nn
import torchvision.models as models
import numpy as np
from PIL import Image
from collections import OrderedDict
import os

# Importer le chemin de configuration
from config import MODEL_PATH

class ChestXrayDenseNet(nn.Module):
    """
    DenseNet121 adapté pour radiographies (1 canal d'entrée, sortie binaire sigmoid).
    """
    def __init__(self):
        super(ChestXrayDenseNet, self).__init__()
        self.densenet121 = models.densenet121(pretrained=False)
        # Première couche pour 1 canal (grayscale)
        self.densenet121.features.conv0 = nn.Conv2d(
            in_channels=1, out_channels=64,
            kernel_size=7, stride=2, padding=3, bias=False
        )
        # Classifieur binaire
        num_ftrs = self.densenet121.classifier.in_features
        self.densenet121.classifier = nn.Linear(num_ftrs, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.densenet121(x)
        return self.sigmoid(x)


def load_chest_xray_model():
    try:
        # Chemin vers votre modèle
        current_dir = os.path.dirname(os.path.abspath(__file__))
        model_path = os.path.join(current_dir, 'model.pth.tar')
        if not os.path.exists(model_path):
            print(f"Modèle non trouvé à {model_path}.")
            return None

        # Instancie votre DenseNet modifié
        model = ChestXrayDenseNet()

        # Charge le checkpoint
        checkpoint = torch.load(model_path, map_location='cpu')
        sd = checkpoint.get('state_dict', checkpoint)

        # Si le checkpoint a été sauvé via DataParallel, retire le préfixe "module."
        new_sd = OrderedDict()
        for k, v in sd.items():
            name = k.replace('module.', '')
            new_sd[name] = v

        # Moyenne RGB→Gris pour conv0
        if 'densenet121.features.conv0.weight' in new_sd:
            w_rgb = new_sd['densenet121.features.conv0.weight']            # shape [64,3,7,7]
            w_gray = w_rgb.mean(dim=1, keepdim=True)                       # shape [64,1,7,7]
            new_sd['densenet121.features.conv0.weight'] = w_gray

        # Charge tout le state_dict (strict=False pour ignorer les autres petites différences)
        model.load_state_dict(new_sd, strict=False)
        model.eval()
        print("Modèle DenseNet121 chargé avec succès !")
        return model

    except Exception as e:
        print(f"Erreur lors du chargement du modèle : {e}")
        return None


def predict_pneumonia(model, image_file):
    """
    Prétraite l'image et prédit la probabilité de pneumonie.
    Retourne un dict {prediction, probability, recommendation, source}.
    """
    try:
        if model is None:
            raise ValueError("Modèle non chargé ou invalide")

        img = Image.open(image_file).convert('L')
        img = img.resize((224, 224))
        arr = np.array(img, dtype=np.float32) / 255.0
        tensor = torch.tensor(arr).unsqueeze(0).unsqueeze(0)

        with torch.no_grad():
            output = model(tensor)
        prob = float(output.item())
        pred_label = "Pneumonie" if prob > 0.5 else "Normal"

        if prob > 0.7:
            rec = "Forte probabilité de pneumonie détectée. Consultation médicale fortement recommandée."
        elif prob > 0.5:
            rec = "Signes possibles de pneumonie détectés. Une consultation médicale est recommandée."
        elif prob > 0.3:
            rec = "Radiographie probablement normale, avec quelques anomalies mineures. Un suivi médical peut être envisagé."
        else:
            rec = "Radiographie normale. Aucun signe de pneumonie détecté."

        return {
            'prediction': pred_label,
            'probability': prob,
            'recommendation': rec,
            'source': 'deep_learning'
        }

    except Exception as e:
        print(f"Erreur lors de la prédiction : {e}")
        return {
            'prediction': "Erreur",
            'probability': 0.0,
            'recommendation': f"Échec de l'analyse : {e}",
            'source': 'fallback'
        }