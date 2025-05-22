import torch
import torch.nn as nn
import torchvision.models as models
import numpy as np
from PIL import Image
import os

# Définir le modèle DenseNet121 adapté aux radiographies pulmonaires (1 canal d'entrée)
class ChestXrayDenseNet(nn.Module):
    def __init__(self):
        super(ChestXrayDenseNet, self).__init__()
        self.densenet121 = models.densenet121(pretrained=False)
        # Adapter la première couche pour 1 canal (image en niveaux de gris)
        self.densenet121.features.conv0 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
        # Adapter la dernière couche pour une sortie binaire (pneumonie ou normal)
        num_ftrs = self.densenet121.classifier.in_features
        self.densenet121.classifier = nn.Linear(num_ftrs, 1)
        # Fonction d'activation sigmoïde pour produire une probabilité entre 0 et 1
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.densenet121(x)
        return self.sigmoid(x)

# Charger le modèle DenseNet121 entraîné depuis le fichier de poids
def load_chest_xray_model():
    try:
        # Construire le chemin absolu du fichier de modèle (models/model.pth.tar)
        base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        model_path = os.path.join(base_dir, 'models', 'model.pth.tar')
        if not os.path.exists(model_path):
            print(f"Modèle non trouvé à {model_path}.")
            return None

        # Initialiser le modèle DenseNet121 et charger les poids entraînés
        model = ChestXrayDenseNet()
        checkpoint = torch.load(model_path, map_location=torch.device('cpu'))
        if 'state_dict' in checkpoint:
            model.load_state_dict(checkpoint['state_dict'])
        else:
            model.load_state_dict(checkpoint)
        model.eval()  # Mettre le modèle en mode évaluation (désactive le dropout, etc.)
        print("Modèle DenseNet121 chargé avec succès.")
        return model

    except Exception as e:
        print(f"Erreur lors du chargement du modèle : {e}")
        return None

# Prédire la pneumonie à partir d'une image de radiographie
def predict_pneumonia(model, image_file):
    """
    Prétraiter l'image fournie et utiliser le modèle pour prédire la présence d'une pneumonie.
    Retourne un dictionnaire avec 'prediction', 'probability', 'recommendation' et 'source'.
    """
    try:
        # Vérifier que le modèle est disponible
        if model is None:
            raise ValueError("Modèle non chargé ou invalide")

        # Ouvrir l'image et la convertir en niveaux de gris
        img = Image.open(image_file)
        img = img.convert('L')              # niveaux de gris
        img = img.resize((224, 224))        # redimensionner à 224x224

        # Convertir l'image en tenseur PyTorch
        img_array = np.array(img, dtype=np.float32) / 255.0   # normaliser les pixels [0,1]
        img_tensor = torch.tensor(img_array).unsqueeze(0).unsqueeze(0)  # shape: (1,1,224,224)

        # Effectuer la prédiction sans calcul de gradient
        with torch.no_grad():
            output = model(img_tensor)
        pneumonia_prob = float(output.item())  # extraire la probabilité de pneumonie (entre 0 et 1)

        # Déterminer l'étiquette prédite en fonction d'un seuil de 0.5
        prediction = "Pneumonie" if pneumonia_prob > 0.5 else "Normal"

        # Formuler une recommandation basée sur la probabilité
        if pneumonia_prob > 0.7:
            recommendation = "Forte probabilité de pneumonie détectée. Consultation médicale fortement recommandée."
        elif pneumonia_prob > 0.5:
            recommendation = "Signes possibles de pneumonie détectés. Une consultation médicale est recommandée."
        elif pneumonia_prob > 0.3:
            recommendation = "Radiographie probablement normale, avec quelques anomalies mineures. Un suivi médical peut être envisagé."
        else:
            recommendation = "Radiographie normale. Aucun signe de pneumonie détecté."

        # Retourner le résultat sous forme de dictionnaire
        return {
            'prediction': prediction,
            'probability': pneumonia_prob,
            'recommendation': recommendation,
            'source': 'deep_learning'
        }

    except Exception as e:
        # En cas d'erreur, retourner un résultat par défaut indiquant l'échec
        print(f"Erreur lors de la prédiction : {e}")
        return {
            'prediction': "Erreur",
            'probability': 0.0,
            'recommendation': f"Échec de l'analyse : {e}",
            'source': 'fallback'
        }
