import torch
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as transforms
import numpy as np
from PIL import Image
from collections import OrderedDict
import os

# Configuration
MODEL_PATH = 'model.pth.tar'
CONFIDENCE_THRESHOLD = 0.65  # Seuil ajusté pour réduire les faux positifs

class ChestXrayDenseNet(nn.Module):
    """
    DenseNet121 adapté pour radiographies pulmonaires
    Compatible avec les modèles entraînés sur ChestX-ray14
    """
    def __init__(self, num_classes=14):
        super(ChestXrayDenseNet, self).__init__()
        self.densenet121 = models.densenet121(pretrained=True)
        num_ftrs = self.densenet121.classifier.in_features
        
        # Classifier pour 14 classes (ChestX-ray14) ou binaire
        if num_classes == 14:
            self.densenet121.classifier = nn.Sequential(
                nn.Linear(num_ftrs, num_classes),
                nn.Sigmoid()
            )
        else:
            # Pour classification binaire pneumonie
            self.densenet121.classifier = nn.Sequential(
                nn.Linear(num_ftrs, 256),
                nn.ReLU(),
                nn.Dropout(0.5),
                nn.Linear(256, 1),
                nn.Sigmoid()
            )

    def forward(self, x):
        return self.densenet121(x)


def get_transforms():
    """
    Transformations correctes pour les radiographies
    IMPORTANT: Utiliser la normalisation ImageNet standard
    """
    normalize = transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
    
    transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        normalize
    ])
    
    return transform


def load_chest_xray_model():
    """
    Charge le modèle avec gestion intelligente des poids
    """
    try:
        current_dir = os.path.dirname(os.path.abspath(__file__))
        model_path = os.path.join(current_dir, MODEL_PATH)
        
        if not os.path.exists(model_path):
            print(f"Modèle non trouvé à {model_path}.")
            return None

        # Charger le checkpoint
        checkpoint = torch.load(model_path, map_location='cpu')
        
        # Extraire le state_dict
        if isinstance(checkpoint, dict) and 'state_dict' in checkpoint:
            state_dict = checkpoint['state_dict']
        else:
            state_dict = checkpoint

        # Déterminer le nombre de classes de sortie
        classifier_key = None
        for k in state_dict.keys():
            if 'classifier' in k and 'weight' in k:
                classifier_key = k
                break
        
        if classifier_key:
            output_size = state_dict[classifier_key].shape[0]
            print(f"Modèle détecté avec {output_size} sorties")
        else:
            output_size = 14  # Par défaut ChestX-ray14
        
        # Créer le modèle approprié
        if output_size == 14:
            # Modèle ChestX-ray14 - on utilisera seulement la sortie pneumonie (index 6)
            model = ChestXrayDenseNet(num_classes=14)
        else:
            # Modèle binaire
            model = ChestXrayDenseNet(num_classes=1)
        
        # Nettoyer les clés (retirer 'module.' si présent)
        new_state_dict = OrderedDict()
        for k, v in state_dict.items():
            name = k.replace('module.', '')
            new_state_dict[name] = v
        
        # Charger les poids
        model.load_state_dict(new_state_dict, strict=False)
        model.eval()
        
        print("Modèle DenseNet121 chargé avec succès!")
        return model, output_size

    except Exception as e:
        print(f"Erreur lors du chargement du modèle : {e}")
        return None, 0


def apply_calibration(prob, aggressive=True):
    """
    Applique une calibration pour réduire les faux positifs
    Le modèle semble surestimer les probabilités
    """
    if aggressive:
        # Calibration agressive pour réduire fortement les faux positifs
        if prob < 0.5:
            return prob * 0.5  # Réduire de moitié les faibles probabilités
        elif prob < 0.8:
            return 0.25 + (prob - 0.5) * 0.5  # Compression au milieu
        else:
            return 0.4 + (prob - 0.8) * 0.3  # Compression forte pour les hautes probabilités
    else:
        # Calibration douce
        return 0.3 + 0.5 * prob


def predict_pneumonia(model_info, image_file, use_tta=True, calibrate=True):
    """
    Prédit la probabilité de pneumonie avec améliorations
    """
    try:
        if model_info is None:
            model, output_size = load_chest_xray_model()
            if model is None:
                raise ValueError("Impossible de charger le modèle")
        else:
            model, output_size = model_info
        
        # IMPORTANT: Convertir en RGB, pas en grayscale!
        img = Image.open(image_file).convert('RGB')
        
        # Transformation correcte
        transform = get_transforms()
        
        if use_tta:
            # Test Time Augmentation pour plus de robustesse
            tta_transforms = []
            
            # 1. Image originale
            tta_transforms.append(transform)
            
            # 2. Légères variations
            tta_transforms.append(transforms.Compose([
                transforms.Resize(256),
                transforms.RandomCrop(224),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            ]))
            
            # 3. Flip horizontal
            tta_transforms.append(transforms.Compose([
                transforms.Resize(256),
                transforms.CenterCrop(224),
                transforms.RandomHorizontalFlip(p=1.0),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            ]))
            
            predictions = []
            with torch.no_grad():
                for t in tta_transforms:
                    tensor = t(img).unsqueeze(0)
                    output = model(tensor)
                    
                    if output_size == 14:
                        # Pour ChestX-ray14, pneumonie est à l'index 6
                        prob = output[0, 6].item()
                    else:
                        prob = output.item()
                    
                    predictions.append(prob)
            
            # Moyenne des prédictions
            raw_prob = np.mean(predictions)
            confidence_std = np.std(predictions)
        else:
            # Prédiction simple
            tensor = transform(img).unsqueeze(0)
            with torch.no_grad():
                output = model(tensor)
                
                if output_size == 14:
                    raw_prob = output[0, 6].item()
                else:
                    raw_prob = output.item()
            
            confidence_std = 0
        
        # Appliquer la calibration
        if calibrate:
            prob = apply_calibration(raw_prob, aggressive=True)
        else:
            prob = raw_prob
        
        # Décision avec seuil ajusté
        pred_label = "Pneumonie" if prob > CONFIDENCE_THRESHOLD else "Normal"
        
        # Recommandations basées sur la probabilité calibrée
        if prob > 0.8:
            rec = "Forte probabilité de pneumonie détectée. Consultation médicale fortement recommandée."
        elif prob > 0.65:
            rec = "Signes possibles de pneumonie détectés. Une consultation médicale est recommandée."
        elif prob > 0.45:
            rec = "Résultat incertain. Un examen complémentaire pourrait être nécessaire."
        elif prob > 0.25:
            rec = "Radiographie probablement normale, avec quelques anomalies mineures possibles."
        else:
            rec = "Radiographie normale. Aucun signe de pneumonie détecté."
        
        # Ajouter une note sur la confiance si TTA utilisé
        if use_tta and confidence_std > 0.15:
            rec += " (Note: Variabilité dans la prédiction, résultat à interpréter avec prudence)"
        
        return {
            'prediction': pred_label,
            'probability': float(prob),
            'raw_probability': float(raw_prob),
            'confidence_std': float(confidence_std) if use_tta else None,
            'recommendation': rec,
            'source': 'deep_learning_improved',
            'threshold_used': CONFIDENCE_THRESHOLD,
            'calibration_applied': calibrate
        }

    except Exception as e:
        print(f"Erreur lors de la prédiction : {e}")
        import traceback
        traceback.print_exc()
        return {
            'prediction': "Erreur",
            'probability': 0.0,
            'recommendation': f"Échec de l'analyse : {str(e)}",
            'source': 'error'
        }


# Pour compatibilité avec l'ancienne API
def load_model():
    """Fonction de compatibilité"""
    model, _ = load_chest_xray_model()
    return model


if __name__ == "__main__":
    # Test du modèle amélioré
    print("=== TEST DU MODÈLE AMÉLIORÉ ===\n")
    
    model_info = load_chest_xray_model()
    if model_info[0]:
        # Tester sur une image
        test_images = ["radio5.jpg", "test.jpg", "normal.jpg", "pneumonia.jpg"]
        
        for img_path in test_images:
            if os.path.exists(img_path):
                print(f"\nTest sur {img_path}:")
                
                # Test avec calibration
                result = predict_pneumonia(model_info, img_path, use_tta=True, calibrate=True)
                print(f"  Avec calibration:")
                print(f"    - Prédiction: {result['prediction']}")
                print(f"    - Probabilité: {result['probability']:.4f}")
                print(f"    - Probabilité brute: {result['raw_probability']:.4f}")
                
                # Test sans calibration pour comparaison
                result_raw = predict_pneumonia(model_info, img_path, use_tta=False, calibrate=False)
                print(f"  Sans calibration:")
                print(f"    - Probabilité brute: {result_raw['probability']:.4f}")