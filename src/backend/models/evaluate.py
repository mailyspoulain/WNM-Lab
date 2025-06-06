import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc, confusion_matrix, classification_report
import seaborn as sns
import sys

# Ajouter le répertoire parent au path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import MODEL_PATH, IMAGE_SIZE, CLASS_NAMES
from backend.models.modelOLD import load_model


def evaluate_model(test_data_dir, output_dir=None):
    """
    Évalue les performances du modèle sur un ensemble de test
    
    Args:
        test_data_dir: Chemin vers le répertoire contenant les données de test
        output_dir: Répertoire où sauvegarder les visualisations
    """
    if output_dir is None:
        output_dir = os.path.join(os.path.dirname(MODEL_PATH), 'evaluation')
    
    os.makedirs(output_dir, exist_ok=True)
    
    # Charger le modèle
    model = load_model()
    
    # Déterminer si nous utilisons un modèle pour images RGB ou grayscale
    input_shape = model.input_shape
    if input_shape[-1] == 1:
        color_mode = 'grayscale'
        preprocess_func = lambda x: x / 255.0
    else:
        color_mode = 'rgb'
        preprocess_func = tf.keras.applications.vgg16.preprocess_input
    
    # Générateur de données pour le test
    test_datagen = ImageDataGenerator(preprocessing_function=preprocess_func)
    
    test_generator = test_datagen.flow_from_directory(
        test_data_dir,
        target_size=IMAGE_SIZE,
        batch_size=1,  # Une image à la fois pour mieux analyser
        class_mode='binary',
        color_mode=color_mode,
        shuffle=False
    )
    
    # Faire des prédictions
    predictions = model.predict(test_generator, steps=test_generator.samples)
    y_pred = (predictions > 0.5).astype(int)
    y_true = test_generator.classes
    
    # 1. Classification Report
    report = classification_report(y_true, y_pred, target_names=CLASS_NAMES)
    print("Classification Report:")
    print(report)
    
    with open(os.path.join(output_dir, 'classification_report.txt'), 'w') as f:
        f.write(report)
    
    # 2. Confusion Matrix
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=CLASS_NAMES, yticklabels=CLASS_NAMES)
    plt.xlabel('Prédictions')
    plt.ylabel('Réalité')
    plt.title('Matrice de confusion')
    plt.savefig(os.path.join(output_dir, 'confusion_matrix.png'))
    
    # 3. ROC Curve
    fpr, tpr, thresholds = roc_curve(y_true, predictions)
    roc_auc = auc(fpr, tpr)
    
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (area = {roc_auc:.2f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('Taux de faux positifs')
    plt.ylabel('Taux de vrais positifs')
    plt.title('Courbe ROC')
    plt.legend(loc="lower right")
    plt.savefig(os.path.join(output_dir, 'roc_curve.png'))
    
    # 4. Visualiser quelques exemples de prédictions
    plt.figure(figsize=(15, 10))
    for i in range(min(9, test_generator.samples)):
        plt.subplot(3, 3, i+1)
        batch = next(test_generator)
        image = batch[0][0]
        true_label = batch[1][0]
        
        # Normaliser pour affichage
        if color_mode == 'grayscale':
            image = np.squeeze(image)
        
        # Prédiction
        pred = model.predict(np.expand_dims(batch[0][0], axis=0))[0][0]
        pred_label = int(pred > 0.5)
        
        plt.imshow(image, cmap='gray' if color_mode == 'grayscale' else None)
        plt.title(f"Vrai: {CLASS_NAMES[int(true_label)]}\nPrédit: {CLASS_NAMES[pred_label]} ({pred:.2f})")
        plt.axis('off')
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'prediction_examples.png'))
    
    return {
        'accuracy': (y_pred == y_true).mean(),
        'confusion_matrix': cm,
        'roc_auc': roc_auc
    }


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Évaluer un modèle de détection de pneumonie')
    parser.add_argument('--test_dir', type=str, required=True, help='Chemin vers le répertoire de test')
    parser.add_argument('--output_dir', type=str, help='Répertoire où sauvegarder les résultats')
    
    args = parser.parse_args()
    
    results = evaluate_model(args.test_dir, args.output_dir)
    print(f"Précision globale: {results['accuracy']:.4f}")
    print(f"AUC ROC: {results['roc_auc']:.4f}")
