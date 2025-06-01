import torch
import torch.nn as nn
import numpy as np
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression
from sklearn.calibration import calibration_curve
import matplotlib.pyplot as plt
import pickle
import os
import sys
from PIL import Image

# Ajouter le chemin parent si nécessaire
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Importer les modules nécessaires
try:
    from models.model_loader_improved import load_chest_xray_model, get_transforms, predict_pneumonia
except ImportError:
    try:
        from model_loader_improved import load_chest_xray_model, get_transforms, predict_pneumonia
    except ImportError:
        from model_loader import load_chest_xray_model, get_transforms
        
        # Définir predict_pneumonia basique si non disponible
        def predict_pneumonia(model_info, image_path, **kwargs):
            if isinstance(model_info, tuple):
                model, _ = model_info
            else:
                model = model_info
            
            img = Image.open(image_path).convert('RGB')
            transform = get_transforms()
            tensor = transform(img).unsqueeze(0)
            
            with torch.no_grad():
                output = model(tensor)
                prob = output.item() if output.shape[-1] == 1 else output[0, 6].item()
            
            return {'probability': prob}


class ModelCalibrator:
    """
    Classe pour calibrer les probabilités du modèle de détection de pneumonie
    """
    def __init__(self, model_info, calibration_method='isotonic'):
        self.model_info = model_info
        self.calibration_method = calibration_method
        self.calibrator = None
        self.threshold = 0.5
        self.calibration_data = {
            'uncalibrated_probs': [],
            'calibrated_probs': [],
            'true_labels': []
        }
        
    def collect_calibration_data(self, image_paths_normal, image_paths_pneumonia, max_images=50):
        """
        Collecter des données pour la calibration à partir d'images
        """
        print("Collecte des données de calibration...")
        
        all_probs = []
        all_labels = []
        
        # Traiter les images normales
        for i, img_path in enumerate(image_paths_normal[:max_images]):
            if os.path.exists(img_path):
                try:
                    result = predict_pneumonia(self.model_info, img_path, use_tta=False, calibrate=False)
                    prob = result.get('raw_probability', result.get('probability', 0))
                    all_probs.append(prob)
                    all_labels.append(0)  # Normal
                    if (i + 1) % 10 == 0:
                        print(f"  Traité {i+1}/{min(len(image_paths_normal), max_images)} images normales")
                except Exception as e:
                    print(f"  Erreur avec {img_path}: {e}")
        
        # Traiter les images pneumonie
        for i, img_path in enumerate(image_paths_pneumonia[:max_images]):
            if os.path.exists(img_path):
                try:
                    result = predict_pneumonia(self.model_info, img_path, use_tta=False, calibrate=False)
                    prob = result.get('raw_probability', result.get('probability', 0))
                    all_probs.append(prob)
                    all_labels.append(1)  # Pneumonie
                    if (i + 1) % 10 == 0:
                        print(f"  Traité {i+1}/{min(len(image_paths_pneumonia), max_images)} images pneumonie")
                except Exception as e:
                    print(f"  Erreur avec {img_path}: {e}")
        
        self.calibration_data['uncalibrated_probs'] = np.array(all_probs)
        self.calibration_data['true_labels'] = np.array(all_labels)
        
        print(f"\n✅ Collecté {len(all_probs)} échantillons pour la calibration")
        print(f"   - {sum(1 for l in all_labels if l == 0)} normaux")
        print(f"   - {sum(1 for l in all_labels if l == 1)} pneumonies")
        
        return all_probs, all_labels
    
    def create_synthetic_calibration_data(self):
        """
        Créer des données de calibration synthétiques basées sur l'observation
        que le modèle surestime les probabilités
        """
        print("Création de données de calibration synthétiques...")
        
        # Simuler des probabilités typiques du modèle
        n_samples = 200
        
        # Pour les cas normaux (label=0), le modèle tend à donner des probs entre 0.3 et 0.8
        normal_probs = np.concatenate([
            np.random.beta(2, 5, n_samples//4),  # Majorité entre 0.1-0.4
            np.random.beta(5, 3, n_samples//4) * 0.7 + 0.3  # Quelques uns entre 0.3-0.8
        ])
        
        # Pour les cas pneumonie (label=1), le modèle tend à donner des probs élevées
        pneumonia_probs = np.concatenate([
            np.random.beta(5, 2, n_samples//4) * 0.3 + 0.7,  # Majorité entre 0.7-1.0
            np.random.beta(3, 3, n_samples//4) * 0.4 + 0.5   # Quelques uns entre 0.5-0.9
        ])
        
        # Combiner
        all_probs = np.concatenate([normal_probs, pneumonia_probs])
        all_labels = np.concatenate([np.zeros(len(normal_probs)), np.ones(len(pneumonia_probs))])
        
        # Mélanger
        indices = np.random.permutation(len(all_probs))
        self.calibration_data['uncalibrated_probs'] = all_probs[indices]
        self.calibration_data['true_labels'] = all_labels[indices]
        
        print(f"✅ Créé {len(all_probs)} échantillons synthétiques")
        
    def calibrate(self, probs=None, labels=None):
        """
        Calibrer le modèle
        """
        if probs is None:
            probs = self.calibration_data['uncalibrated_probs']
            labels = self.calibration_data['true_labels']
        
        if len(probs) == 0:
            print("⚠️  Pas de données de calibration, utilisation de valeurs par défaut")
            self.create_synthetic_calibration_data()
            probs = self.calibration_data['uncalibrated_probs']
            labels = self.calibration_data['true_labels']
        
        print(f"\nCalibration avec {len(probs)} échantillons...")
        
        if self.calibration_method == 'isotonic':
            self.calibrator = IsotonicRegression(out_of_bounds='clip')
            self.calibrator.fit(probs, labels)
        elif self.calibration_method == 'platt':
            self.calibrator = LogisticRegression()
            self.calibrator.fit(probs.reshape(-1, 1), labels)
        elif self.calibration_method == 'beta':
            # Calibration Beta personnalisée
            from scipy.optimize import minimize
            
            def beta_calibration(p, a, b):
                return (p ** a) / (p ** a + (1 - p) ** b)
            
            def loss(params, probs, labels):
                a, b = params
                cal_probs = beta_calibration(probs, a, b)
                # Log loss
                eps = 1e-15
                cal_probs = np.clip(cal_probs, eps, 1 - eps)
                return -np.mean(labels * np.log(cal_probs) + (1 - labels) * np.log(1 - cal_probs))
            
            result = minimize(loss, [1.0, 1.0], args=(probs, labels), bounds=[(0.1, 10), (0.1, 10)])
            self.calibrator = lambda p: beta_calibration(p, result.x[0], result.x[1])
            print(f"   Beta parameters: a={result.x[0]:.3f}, b={result.x[1]:.3f}")
        else:
            # Calibration linéaire simple
            self.calibrator = lambda p: 0.3 + 0.5 * p
        
        # Calculer les probabilités calibrées
        if self.calibration_method in ['isotonic', 'platt']:
            if self.calibration_method == 'platt':
                probs_input = probs.reshape(-1, 1)
            else:
                probs_input = probs
            calibrated_probs = self.calibrator.predict(probs_input)
        else:
            calibrated_probs = self.calibrator(probs)
        
        self.calibration_data['calibrated_probs'] = calibrated_probs
        
        # Trouver le seuil optimal
        self.threshold = self._find_optimal_threshold(calibrated_probs, labels)
        
        print(f"✅ Calibration terminée")
        print(f"   - Méthode: {self.calibration_method}")
        print(f"   - Seuil optimal: {self.threshold:.3f}")
        
    def _find_optimal_threshold(self, probs, labels, beta=1.5):
        """
        Trouver le seuil optimal en maximisant le F-beta score
        beta > 1 favorise le rappel (moins de faux négatifs)
        """
        thresholds = np.linspace(0.1, 0.9, 50)
        f_scores = []
        
        for thresh in thresholds:
            preds = (probs >= thresh).astype(int)
            tp = np.sum((preds == 1) & (labels == 1))
            fp = np.sum((preds == 1) & (labels == 0))
            fn = np.sum((preds == 0) & (labels == 1))
            
            precision = tp / (tp + fp) if (tp + fp) > 0 else 0
            recall = tp / (tp + fn) if (tp + fn) > 0 else 0
            
            if precision + recall > 0:
                f_beta = (1 + beta**2) * (precision * recall) / (beta**2 * precision + recall)
            else:
                f_beta = 0
            
            f_scores.append(f_beta)
        
        optimal_idx = np.argmax(f_scores)
        return thresholds[optimal_idx]
    
    def apply_calibration(self, prob):
        """
        Appliquer la calibration à une probabilité
        """
        if self.calibrator is None:
            # Calibration par défaut si non calibré
            return 0.3 + 0.5 * prob
        
        if self.calibration_method == 'platt':
            return self.calibrator.predict(np.array([[prob]]))[0]
        elif self.calibration_method == 'isotonic':
            return self.calibrator.predict([prob])[0]
        else:
            return self.calibrator(prob)
    
    def plot_calibration_curves(self, save_path='calibration_analysis.png'):
        """
        Visualiser les courbes de calibration
        """
        if len(self.calibration_data['uncalibrated_probs']) == 0:
            print("Pas de données pour créer les courbes")
            return
        
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        fig.suptitle('Analyse de la calibration du modèle', fontsize=16)
        
        probs_uncal = self.calibration_data['uncalibrated_probs']
        probs_cal = self.calibration_data['calibrated_probs']
        labels = self.calibration_data['true_labels']
        
        # 1. Courbe de calibration avant/après
        ax = axes[0, 0]
        fraction_pos_uncal, mean_pred_uncal = calibration_curve(labels, probs_uncal, n_bins=10)
        fraction_pos_cal, mean_pred_cal = calibration_curve(labels, probs_cal, n_bins=10)
        
        ax.plot(mean_pred_uncal, fraction_pos_uncal, 's-', label='Non calibré', color='red')
        ax.plot(mean_pred_cal, fraction_pos_cal, 's-', label='Calibré', color='green')
        ax.plot([0, 1], [0, 1], 'k--', label='Parfait')
        ax.set_xlabel('Probabilité moyenne prédite')
        ax.set_ylabel('Fraction de positifs')
        ax.set_title('Courbes de calibration')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # 2. Distribution des probabilités
        ax = axes[0, 1]
        bins = np.linspace(0, 1, 20)
        ax.hist(probs_uncal[labels == 0], bins, alpha=0.5, label='Normal (non cal)', color='blue')
        ax.hist(probs_uncal[labels == 1], bins, alpha=0.5, label='Pneumonie (non cal)', color='red')
        ax.set_xlabel('Probabilité')
        ax.set_ylabel('Fréquence')
        ax.set_title('Distribution des probabilités non calibrées')
        ax.legend()
        ax.axvline(x=0.5, color='black', linestyle='--', alpha=0.5)
        
        # 3. Fonction de calibration
        ax = axes[1, 0]
        p_test = np.linspace(0, 1, 100)
        p_cal = [self.apply_calibration(p) for p in p_test]
        ax.plot(p_test, p_cal, 'b-', linewidth=2, label='Fonction de calibration')
        ax.plot([0, 1], [0, 1], 'k--', alpha=0.5, label='Identité')
        ax.set_xlabel('Probabilité originale')
        ax.set_ylabel('Probabilité calibrée')
        ax.set_title('Fonction de transformation')
        ax.grid(True, alpha=0.3)
        ax.legend()
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        
        # 4. Métriques
        ax = axes[1, 1]
        
        # Calculer les métriques
        from sklearn.metrics import brier_score_loss, log_loss
        
        brier_uncal = brier_score_loss(labels, probs_uncal)
        brier_cal = brier_score_loss(labels, probs_cal)
        
        # ECE (Expected Calibration Error)
        def expected_calibration_error(y_true, y_prob, n_bins=10):
            bin_boundaries = np.linspace(0, 1, n_bins + 1)
            bin_lowers = bin_boundaries[:-1]
            bin_uppers = bin_boundaries[1:]
            
            ece = 0
            for bin_lower, bin_upper in zip(bin_lowers, bin_uppers):
                in_bin = (y_prob > bin_lower) & (y_prob <= bin_upper)
                prop_in_bin = in_bin.mean()
                
                if prop_in_bin > 0:
                    accuracy_in_bin = y_true[in_bin].mean()
                    avg_confidence_in_bin = y_prob[in_bin].mean()
                    ece += np.abs(avg_confidence_in_bin - accuracy_in_bin) * prop_in_bin
            
            return ece
        
        ece_uncal = expected_calibration_error(labels, probs_uncal)
        ece_cal = expected_calibration_error(labels, probs_cal)
        
        metrics_text = f"Métriques de calibration:\n\n"
        metrics_text += f"Brier Score:\n"
        metrics_text += f"  Non calibré: {brier_uncal:.4f}\n"
        metrics_text += f"  Calibré: {brier_cal:.4f}\n"
        metrics_text += f"  Amélioration: {(1 - brier_cal/brier_uncal)*100:.1f}%\n\n"
        metrics_text += f"ECE (Expected Calibration Error):\n"
        metrics_text += f"  Non calibré: {ece_uncal:.4f}\n"
        metrics_text += f"  Calibré: {ece_cal:.4f}\n\n"
        metrics_text += f"Seuil optimal: {self.threshold:.3f}"
        
        ax.text(0.1, 0.9, metrics_text, transform=ax.transAxes,
                fontsize=11, verticalalignment='top', fontfamily='monospace')
        ax.axis('off')
        ax.set_title('Métriques')
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"\n✅ Graphique sauvegardé dans: {save_path}")
        plt.show()
    
    def save_calibrator(self, path='calibrated_model.pkl'):
        """
        Sauvegarder le calibrateur
        """
        calibration_dict = {
            'calibrator': self.calibrator,
            'method': self.calibration_method,
            'threshold': self.threshold,
            'calibration_data': self.calibration_data
        }
        
        with open(path, 'wb') as f:
            pickle.dump(calibration_dict, f)
        
        print(f"✅ Calibrateur sauvegardé dans: {path}")
    
    def load_calibrator(self, path='calibrated_model.pkl'):
        """
        Charger un calibrateur sauvegardé
        """
        with open(path, 'rb') as f:
            calibration_dict = pickle.load(f)
        
        self.calibrator = calibration_dict['calibrator']
        self.calibration_method = calibration_dict['method']
        self.threshold = calibration_dict['threshold']
        self.calibration_data = calibration_dict['calibration_data']
        
        print(f"✅ Calibrateur chargé depuis: {path}")


def quick_calibrate_model():
    """
    Fonction rapide pour calibrer le modèle avec des paramètres par défaut
    """
    print("=== CALIBRATION RAPIDE DU MODÈLE ===\n")
    
    # Charger le modèle
    model_info = load_chest_xray_model()
    if not model_info or model_info[0] is None:
        print("❌ Impossible de charger le modèle")
        return None
    
    # Créer le calibrateur
    calibrator = ModelCalibrator(model_info, calibration_method='isotonic')
    
    # Chercher des images pour la calibration
    data_dirs = [
        'test_structured',
        '../test_structured',
        '../../test_structured',
        'val',
        '../val',
        '../../val'
    ]
    
    normal_images = []
    pneumonia_images = []
    
    for data_dir in data_dirs:
        if os.path.exists(data_dir):
            normal_dir = os.path.join(data_dir, 'Normal')
            pneumonia_dir = os.path.join(data_dir, 'Pneumonie')
            
            if os.path.exists(normal_dir):
                normal_images.extend([os.path.join(normal_dir, f) for f in os.listdir(normal_dir) 
                                    if f.lower().endswith(('.jpg', '.jpeg', '.png'))])
            
            if os.path.exists(pneumonia_dir):
                pneumonia_images.extend([os.path.join(pneumonia_dir, f) for f in os.listdir(pneumonia_dir) 
                                       if f.lower().endswith(('.jpg', '.jpeg', '.png'))])
    
    if normal_images and pneumonia_images:
        print(f"Trouvé {len(normal_images)} images normales et {len(pneumonia_images)} images pneumonie")
        calibrator.collect_calibration_data(normal_images, pneumonia_images, max_images=30)
    else:
        print("⚠️  Pas d'images trouvées, utilisation de données synthétiques")
        calibrator.create_synthetic_calibration_data()
    
    # Calibrer
    calibrator.calibrate()
    
    # Visualiser
    calibrator.plot_calibration_curves()
    
    # Sauvegarder
    calibrator.save_calibrator()
    
    return calibrator


def test_calibrated_prediction(calibrator, image_path):
    """
    Tester une prédiction avec le modèle calibré
    """
    print(f"\n=== TEST SUR {image_path} ===")
    
    # Prédiction non calibrée
    result_uncal = predict_pneumonia(calibrator.model_info, image_path, use_tta=True, calibrate=False)
    prob_uncal = result_uncal.get('raw_probability', result_uncal.get('probability', 0))
    
    # Appliquer la calibration
    prob_cal = calibrator.apply_calibration(prob_uncal)
    
    print(f"Probabilité originale: {prob_uncal:.4f}")
    print(f"Probabilité calibrée: {prob_cal:.4f}")
    print(f"Seuil utilisé: {calibrator.threshold:.3f}")
    print(f"Prédiction: {'Pneumonie' if prob_cal > calibrator.threshold else 'Normal'}")
    
    # Recommandation
    if prob_cal > 0.8:
        rec = "Forte probabilité de pneumonie. Consultation recommandée."
    elif prob_cal > calibrator.threshold:
        rec = "Signes possibles de pneumonie. Avis médical conseillé."
    elif prob_cal > 0.3:
        rec = "Résultat incertain. Surveillance recommandée."
    else:
        rec = "Radiographie normale."
    
    print(f"Recommandation: {rec}")
    
    return {
        'prob_original': prob_uncal,
        'prob_calibrated': prob_cal,
        'prediction': 'Pneumonie' if prob_cal > calibrator.threshold else 'Normal',
        'recommendation': rec
    }


if __name__ == "__main__":
    # Calibrer le modèle
    calibrator = quick_calibrate_model()
    
    if calibrator:
        # Tester sur des images
        test_images = ['radio5.jpg', '../radio5.jpg', '../../radio5.jpg']
        
        for img_path in test_images:
            if os.path.exists(img_path):
                test_calibrated_prediction(calibrator, img_path)
                break
        else:
            print("\n⚠️  Aucune image de test trouvée")
