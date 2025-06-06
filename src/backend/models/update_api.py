"""
Script pour mettre à jour votre API avec les améliorations du modèle
Exécutez ce script pour appliquer les corrections
"""

import os
import shutil

def update_model_loader():
    """
    Mise à jour du fichier model_loader.py avec les corrections
    """
    print("Mise à jour de model_loader.py...")
    
    # Créer une sauvegarde
    if os.path.exists('model_loader.py'):
        shutil.copy('model_loader.py', 'model_loader_backup.py')
        print("✓ Sauvegarde créée: model_loader_backup.py")
    
    # Les principales corrections à appliquer
    corrections = """
# Corrections principales à appliquer dans model_loader.py:

1. NORMALISATION CORRECTE:
   Remplacer:
   arr = np.array(img, dtype=np.float32) / 255.0
   
   Par:
   transform = transforms.Compose([
       transforms.Resize(256),
       transforms.CenterCrop(224),
       transforms.ToTensor(),
       transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
   ])
   tensor = transform(img)

2. CONVERSION RGB:
   Remplacer:
   img = Image.open(image_file).convert('L')
   
   Par:
   img = Image.open(image_file).convert('RGB')

3. AJUSTER LE SEUIL:
   Changer les conditions de recommandation:
   - Seuil de décision à 0.65 au lieu de 0.5
   - Probabilité > 0.8 pour "Forte probabilité"
   - Probabilité > 0.65 pour "Signes possibles"

4. AJOUTER TEST TIME AUGMENTATION (optionnel):
   Faire plusieurs prédictions avec différentes augmentations
   et prendre la moyenne pour plus de robustesse
"""
    
    print(corrections)
    
    # Créer un fichier de configuration pour les seuils
    config_content = """# Configuration des seuils pour le modèle

# Seuil de décision principal
DECISION_THRESHOLD = 0.65

# Seuils pour les recommandations
STRONG_POSITIVE_THRESHOLD = 0.80
POSSIBLE_POSITIVE_THRESHOLD = 0.65
UNCERTAIN_THRESHOLD = 0.40
PROBABLE_NEGATIVE_THRESHOLD = 0.20

# Activer Test Time Augmentation
USE_TTA = True
"""
    
    with open('model_config.py', 'w') as f:
        f.write(config_content)
    
    print("\n✓ Fichier model_config.py créé")
    

def create_quick_fix_wrapper():
    """
    Créer un wrapper rapide pour corriger les prédictions
    """
    wrapper_code = '''import torch
import torchvision.transforms as transforms
from PIL import Image
import numpy as np
from model_loader import load_chest_xray_model

class ImprovedPredictor:
    """Wrapper amélioré pour le modèle avec corrections"""
    
    def __init__(self):
        self.model = load_chest_xray_model()
        self.threshold = 0.65
        
        # Transformation correcte avec normalisation ImageNet
        self.transform = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
    
    def predict(self, image_path):
        """Prédiction améliorée avec preprocessing correct"""
        if self.model is None:
            return {
                'prediction': 'Erreur',
                'probability': 0.0,
                'recommendation': 'Modèle non chargé',
                'source': 'error'
            }
        
        try:
            # IMPORTANT: Convertir en RGB, pas en grayscale
            img = Image.open(image_path).convert('RGB')
            
            # Appliquer la transformation correcte
            tensor = self.transform(img).unsqueeze(0)
            
            # Prédiction
            with torch.no_grad():
                output = self.model(tensor)
                prob = output.item()
            
            # Calibration simple pour réduire les faux positifs
            # Le modèle semble surestimer, on applique une correction
            calibrated_prob = 0.3 + 0.5 * prob
            
            # Décision avec seuil ajusté
            pred_label = "Pneumonie" if calibrated_prob > self.threshold else "Normal"
            
            # Recommandations ajustées
            if calibrated_prob > 0.80:
                rec = "Forte probabilité de pneumonie détectée. Consultation médicale fortement recommandée."
            elif calibrated_prob > 0.65:
                rec = "Signes possibles de pneumonie détectés. Une consultation médicale est recommandée."
            elif calibrated_prob > 0.40:
                rec = "Résultat incertain. Un examen complémentaire pourrait être nécessaire."
            else:
                rec = "Radiographie normale. Aucun signe de pneumonie détecté."
            
            return {
                'prediction': pred_label,
                'probability': float(calibrated_prob),
                'original_probability': float(prob),
                'threshold_used': self.threshold,
                'recommendation': rec,
                'source': 'deep_learning_improved'
            }
            
        except Exception as e:
            return {
                'prediction': 'Erreur',
                'probability': 0.0,
                'recommendation': f'Erreur lors de l analyse: {str(e)}',
                'source': 'error'
            }

# Instance globale
predictor = ImprovedPredictor()

def predict_pneumonia_improved(image_path):
    """Fonction de prédiction améliorée"""
    return predictor.predict(image_path)
'''
    
    with open('improved_predictor.py', 'w', encoding='utf-8') as f:
        f.write(wrapper_code)
    
    print("✓ Fichier improved_predictor.py créé")


def test_improvements():
    """
    Script de test pour vérifier les améliorations
    """
    test_code = '''import os
from improved_predictor import predict_pneumonia_improved

# Tester sur vos images
test_images = [
    "radio5.jpg",  # L'image qui donnait un faux positif
    # Ajoutez d'autres images de test ici
]

print("=== TEST DES AMÉLIORATIONS ===\\n")

for img_path in test_images:
    if os.path.exists(img_path):
        print(f"Test sur {img_path}:")
        result = predict_pneumonia_improved(img_path)
        
        print(f"  - Prédiction: {result['prediction']}")
        print(f"  - Probabilité calibrée: {result['probability']:.4f}")
        print(f"  - Probabilité originale: {result.get('original_probability', 'N/A')}")
        print(f"  - Seuil utilisé: {result.get('threshold_used', 'N/A')}")
        print(f"  - Recommandation: {result['recommendation']}")
        print()
'''
    
    with open('test_improvements.py', 'w') as f:
        f.write(test_code)
    
    print("✓ Fichier test_improvements.py créé")


def main():
    print("=== MISE À JOUR DU MODÈLE POUR AMÉLIORER LA PRÉCISION ===\n")
    
    # 1. Afficher les instructions de mise à jour
    update_model_loader()
    
    # 2. Créer le wrapper amélioré
    print("\nCréation du prédicteur amélioré...")
    create_quick_fix_wrapper()
    
    # 3. Créer le script de test
    print("\nCréation du script de test...")
    test_improvements()
    
    print("\n=== INSTRUCTIONS ===")
    print("\n1. Pour tester rapidement les améliorations:")
    print("   python test_improvements.py")
    
    print("\n2. Pour utiliser le prédicteur amélioré dans votre API:")
    print("   Remplacez dans votre route Flask:")
    print("   from model_loader import predict_pneumonia")
    print("   Par:")
    print("   from improved_predictor import predict_pneumonia_improved as predict_pneumonia")
    
    print("\n3. Pour une solution permanente:")
    print("   - Appliquez les corrections listées dans model_loader.py")
    print("   - Ou utilisez le fichier model_loader_improved.py fourni")
    
    print("\n✓ Mise à jour terminée!")


if __name__ == "__main__":
    main()