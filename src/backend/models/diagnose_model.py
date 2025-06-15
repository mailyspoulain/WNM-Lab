import torch
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import seaborn as sns
import os
import sys

# Ajouter le chemin parent si nécessaire
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Essayer d'importer depuis différents endroits
try:
    from models.model_loader_improved import load_chest_xray_model, get_transforms
except ImportError:
    try:
        from model_loader_improved import load_chest_xray_model, get_transforms
    except ImportError:
        from model_loader import load_chest_xray_model, get_transforms

def diagnose_model_performance(model_info, test_image_path):
    """
    Diagnostic complet du modèle pour comprendre ses prédictions
    """
    print("=== DIAGNOSTIC DU MODÈLE ===\n")
    
    if isinstance(model_info, tuple):
        model, output_size = model_info
    else:
        model = model_info
        output_size = 1
    
    # 1. Vérifier la structure du modèle
    print("1. Structure du modèle:")
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"   - Paramètres totaux: {total_params:,}")
    print(f"   - Paramètres entraînables: {trainable_params:,}")
    print(f"   - Nombre de sorties: {output_size}")
    
    # 2. Analyser l'image de test
    print("\n2. Analyse de l'image de test:")
    img = Image.open(test_image_path).convert('RGB')
    print(f"   - Taille originale: {img.size}")
    print(f"   - Mode: {img.mode}")
    
    # Analyser la luminosité et le contraste
    img_array = np.array(img)
    brightness = np.mean(img_array)
    contrast = np.std(img_array)
    print(f"   - Luminosité moyenne: {brightness:.2f}")
    print(f"   - Contraste (std): {contrast:.2f}")
    
    # 3. Vérifier les statistiques de l'image après prétraitement
    transform = get_transforms()
    tensor = transform(img).unsqueeze(0)
    
    print("\n3. Statistiques après prétraitement:")
    print(f"   - Shape du tenseur: {tensor.shape}")
    print(f"   - Min: {tensor.min():.4f}, Max: {tensor.max():.4f}")
    mean_per_channel = tensor.mean(dim=(0,2,3)).numpy()
    std_per_channel = tensor.std(dim=(0,2,3)).numpy()
    print(f"   - Moyenne par canal (R,G,B): [{mean_per_channel[0]:.3f}, {mean_per_channel[1]:.3f}, {mean_per_channel[2]:.3f}]")
    print(f"   - Std par canal (R,G,B): [{std_per_channel[0]:.3f}, {std_per_channel[1]:.3f}, {std_per_channel[2]:.3f}]")
    
    # Vérifier si la normalisation est correcte
    expected_mean = [0.485, 0.456, 0.406]
    expected_std = [0.229, 0.224, 0.225]
    mean_diff = np.abs(mean_per_channel - expected_mean).mean()
    if mean_diff > 0.1:
        print("   ⚠️  ATTENTION: La normalisation semble incorrecte!")
    
    # 4. Analyser les activations intermédiaires
    print("\n4. Analyse des activations:")
    activations = {}
    
    def hook_fn(name):
        def hook(module, input, output):
            activations[name] = output.detach()
        return hook
    
    # Enregistrer des hooks sur certaines couches
    hooks = []
    hooks.append(model.densenet121.features.conv0.register_forward_hook(hook_fn('conv0')))
    hooks.append(model.densenet121.features.denseblock4.register_forward_hook(hook_fn('denseblock4')))
    
    # Forward pass
    with torch.no_grad():
        output = model(tensor)
    
    # Analyser les activations
    for name, act in activations.items():
        print(f"   - {name}: shape={act.shape}, mean={act.mean():.4f}, std={act.std():.4f}")
        
        # Vérifier les activations mortes
        dead_neurons = (act == 0).float().mean().item()
        if dead_neurons > 0.5:
            print(f"     ⚠️  {dead_neurons*100:.1f}% de neurones morts!")
    
    # Nettoyer les hooks
    for hook in hooks:
        hook.remove()
    
    # 5. Analyser la sortie du modèle
    print("\n5. Analyse de la sortie:")
    if output_size == 14:
        print("   Modèle multi-classes (ChestX-ray14):")
        class_names = ['Atelectasis', 'Cardiomegaly', 'Effusion', 'Infiltration', 
                      'Mass', 'Nodule', 'Pneumonia', 'Pneumothorax', 
                      'Consolidation', 'Edema', 'Emphysema', 'Fibrosis', 
                      'Pleural_Thickening', 'Hernia']
        probs = output[0].cpu().numpy()
        for i, (name, prob) in enumerate(zip(class_names, probs)):
            if i == 6:  # Pneumonie
                print(f"   - {name}: {prob:.4f} ⬅️ PNEUMONIE")
            else:
                print(f"   - {name}: {prob:.4f}")
    else:
        prob = output.item()
        print(f"   - Probabilité brute: {prob:.4f}")
    
    # 6. Test de robustesse avec différentes augmentations
    print("\n6. Test de robustesse (variations de prédiction):")
    predictions = []
    
    augmentations = [
        ("Original", lambda x: x),
        ("Luminosité +20%", lambda x: Image.eval(x, lambda p: min(255, int(p * 1.2)))),
        ("Luminosité -20%", lambda x: Image.eval(x, lambda p: int(p * 0.8))),
        ("Contraste augmenté", lambda x: Image.eval(x, lambda p: int(128 + (p - 128) * 1.5))),
        ("Zoom 95%", lambda x: x.resize((int(x.width * 0.95), int(x.height * 0.95)))),
        ("Rotation 5°", lambda x: x.rotate(5, fillcolor=(128, 128, 128)))
    ]
    
    for aug_name, aug_fn in augmentations:
        try:
            aug_img = aug_fn(img.copy())
            if aug_img.size != img.size:
                aug_img = aug_img.resize(img.size)
            aug_tensor = transform(aug_img).unsqueeze(0)
            with torch.no_grad():
                output = model(aug_tensor)
                if output_size == 14:
                    pred = output[0, 6].item()  # Pneumonie
                else:
                    pred = output.item()
            predictions.append((aug_name, pred))
            print(f"   - {aug_name}: {pred:.4f}")
        except Exception as e:
            print(f"   - {aug_name}: Erreur - {str(e)}")
    
    # Calculer la variabilité
    pred_values = [p[1] for p in predictions if isinstance(p[1], (int, float))]
    if pred_values:
        variability = np.std(pred_values)
        print(f"\n   Variabilité: std={variability:.4f}, "
              f"min={np.min(pred_values):.4f}, max={np.max(pred_values):.4f}")
    else:
        variability = 0
    
    # 7. Analyse des problèmes potentiels
    print("\n7. PROBLÈMES DÉTECTÉS:")
    problems = []
    
    base_pred = predictions[0][1] if predictions else 0
    
    if variability > 0.15:
        problems.append("⚠️  Le modèle montre une forte variabilité - manque de robustesse")
    
    if base_pred > 0.6 and base_pred < 0.8:
        problems.append("⚠️  Prédiction dans la zone d'incertitude (0.6-0.8)")
    
    if trainable_params == 0:
        problems.append("⚠️  Aucun paramètre entraînable - le modèle est complètement gelé")
    
    if brightness < 50 or brightness > 200:
        problems.append("⚠️  L'image a une luminosité inhabituelle")
    
    if contrast < 30:
        problems.append("⚠️  L'image a un faible contraste")
    
    if not problems:
        print("   ✅ Aucun problème majeur détecté")
    else:
        for problem in problems:
            print(f"   {problem}")
    
    # 8. Recommandations
    print("\n8. RECOMMANDATIONS:")
    print("   1. Utiliser la calibration pour ajuster les probabilités")
    print("   2. Appliquer Test Time Augmentation (TTA) pour plus de stabilité")
    print("   3. Vérifier que le prétraitement correspond à l'entraînement")
    print("   4. Considérer un seuil de décision plus élevé (0.65-0.7)")
    
    return {
        'base_prediction': base_pred,
        'predictions': predictions,
        'variability': variability,
        'brightness': brightness,
        'contrast': contrast,
        'problems': problems
    }


def visualize_prediction_analysis(model_info, image_path, save_path='diagnosis_plot.png'):
    """
    Créer une visualisation complète de l'analyse
    """
    img = Image.open(image_path).convert('RGB')
    transform = get_transforms()
    
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    fig.suptitle('Diagnostic du modèle de détection de pneumonie', fontsize=16)
    
    # 1. Image originale
    axes[0, 0].imshow(img)
    axes[0, 0].set_title('Image originale')
    axes[0, 0].axis('off')
    
    # 2. Image en niveaux de gris
    gray = img.convert('L')
    axes[0, 1].imshow(gray, cmap='gray')
    axes[0, 1].set_title('Niveaux de gris')
    axes[0, 1].axis('off')
    
    # 3. Histogramme des intensités
    axes[0, 2].hist(np.array(gray).flatten(), bins=50, color='gray', alpha=0.7)
    axes[0, 2].set_title('Distribution des intensités')
    axes[0, 2].set_xlabel('Valeur de pixel')
    axes[0, 2].set_ylabel('Fréquence')
    axes[0, 2].axvline(x=np.mean(gray), color='red', linestyle='--', label='Moyenne')
    axes[0, 2].legend()
    
    # 4. Test de robustesse
    results = diagnose_model_performance(model_info, image_path)
    if results['predictions']:
        aug_names = [r[0] for r in results['predictions']]
        aug_preds = [r[1] for r in results['predictions']]
        
        colors = ['green' if p < 0.5 else 'orange' if p < 0.65 else 'red' for p in aug_preds]
        bars = axes[1, 0].bar(range(len(aug_names)), aug_preds, color=colors)
        axes[1, 0].set_xticks(range(len(aug_names)))
        axes[1, 0].set_xticklabels(aug_names, rotation=45, ha='right')
        axes[1, 0].axhline(y=0.5, color='r', linestyle='--', label='Seuil 0.5', alpha=0.5)
        axes[1, 0].axhline(y=0.65, color='orange', linestyle='--', label='Seuil 0.65', alpha=0.5)
        axes[1, 0].set_ylabel('Probabilité de pneumonie')
        axes[1, 0].set_title('Robustesse aux augmentations')
        axes[1, 0].legend()
        axes[1, 0].set_ylim(0, 1)
    
    # 5. Zones de l'image
    img_array = np.array(img.convert('L'))
    h, w = img_array.shape
    
    # Diviser en quadrants et calculer la moyenne
    quadrants = np.zeros((2, 2))
    quadrants[0, 0] = np.mean(img_array[:h//2, :w//2])
    quadrants[0, 1] = np.mean(img_array[:h//2, w//2:])
    quadrants[1, 0] = np.mean(img_array[h//2:, :w//2])
    quadrants[1, 1] = np.mean(img_array[h//2:, w//2:])
    
    im = axes[1, 1].imshow(quadrants, cmap='viridis')
    axes[1, 1].set_title('Intensité moyenne par quadrant')
    axes[1, 1].set_xticks([0, 1])
    axes[1, 1].set_yticks([0, 1])
    axes[1, 1].set_xticklabels(['Gauche', 'Droite'])
    axes[1, 1].set_yticklabels(['Haut', 'Bas'])
    plt.colorbar(im, ax=axes[1, 1])
    
    # 6. Résumé
    summary_text = f"Prédiction originale: {results['base_prediction']:.4f}\n"
    summary_text += f"Variabilité: {results['variability']:.4f}\n"
    summary_text += f"Luminosité: {results['brightness']:.1f}\n"
    summary_text += f"Contraste: {results['contrast']:.1f}\n\n"
    summary_text += "Problèmes:\n"
    if results['problems']:
        for p in results['problems'][:3]:  # Max 3 problèmes
            summary_text += f"• {p[3:]}\n"  # Enlever l'emoji
    else:
        summary_text += "• Aucun problème majeur"
    
    axes[1, 2].text(0.1, 0.9, summary_text, transform=axes[1, 2].transAxes,
                    fontsize=10, verticalalignment='top', fontfamily='monospace')
    axes[1, 2].axis('off')
    axes[1, 2].set_title('Résumé du diagnostic')
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"\n✅ Graphique sauvegardé dans: {save_path}")
    plt.show()
    
    return results


if __name__ == "__main__":
    # Test du diagnostic
    print("Chargement du modèle...")
    model_info = load_chest_xray_model()
    
    if model_info and model_info[0] is not None:
        # Chercher une image de test
        test_images = ["radio5.jpg", "../radio5.jpg", "../../radio5.jpg", 
                      "test.jpg", "../test.jpg", "../../test.jpg"]
        
        image_found = None
        for test_image in test_images:
            if os.path.exists(test_image):
                image_found = test_image
                break
        
        if image_found:
            print(f"\n📊 Analyse de l'image: {image_found}")
            results = visualize_prediction_analysis(model_info, image_found)
        else:
            print("\n⚠️  Aucune image de test trouvée.")
            print("Placez une image 'radio5.jpg' dans le même dossier que ce script.")
    else:
        print("❌ Impossible de charger le modèle")