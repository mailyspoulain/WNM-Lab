import numpy as np
import cv2
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.models import Model
import io
from PIL import Image
import os
import sys

# Ajouter le répertoire parent au path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import IMAGE_SIZE, CLASS_NAMES
from utils.preprocessing import preprocess_image


def generate_gradcam(model, img, layer_name=None):
    """
    Génère une carte de chaleur Grad-CAM pour visualiser les zones d'intérêt du modèle
    
    Args:
        model: Modèle TensorFlow/Keras
        img: Image d'entrée (prétraitée, avec dimension de batch)
        layer_name: Nom de la couche à utiliser pour Grad-CAM (si None, utilise la dernière couche de convolution)
    
    Returns:
        np.array: Heatmap Grad-CAM superposée sur l'image originale
    """
    # Si aucun layer n'est spécifié, trouver la dernière couche de convolution
    if layer_name is None:
        for layer in reversed(model.layers):
            if isinstance(layer, tf.keras.layers.Conv2D):
                layer_name = layer.name
                break
    
    # Créer un modèle qui renvoie la sortie de la couche spécifiée et la prédiction
    grad_model = Model(
        inputs=[model.inputs],
        outputs=[model.get_layer(layer_name).output, model.output]
    )
    
    # Calculer le gradient de la sortie par rapport à la carte de caractéristiques de la couche spécifiée
    with tf.GradientTape() as tape:
        conv_outputs, predictions = grad_model(img)
        class_idx = 1  # Classe pour la pneumonie (binaire: 0=normal, 1=pneumonie)
        loss = predictions[:, 0]  # Pour un modèle avec sortie sigmoid
    
    # Gradient de la classe par rapport aux sorties de la feature map
    grads = tape.gradient(loss, conv_outputs)
    
    # Moyenne des gradients sur les axes spatiaux
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
    
    # Pondérer les canaux de la carte de caractéristiques par les gradients
    conv_outputs = conv_outputs[0]
    heatmap = tf.reduce_sum(tf.multiply(pooled_grads, conv_outputs), axis=-1)
    
    # Normaliser la heatmap
    heatmap = tf.maximum(heatmap, 0) / tf.math.reduce_max(heatmap)
    heatmap = heatmap.numpy()
    
    # Redimensionner la heatmap à la taille de l'image d'entrée
    heatmap = cv2.resize(heatmap, IMAGE_SIZE)
    
    # Convertir la heatmap en RGB
    heatmap = np.uint8(255 * heatmap)
    heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
    
    # Récupérer l'image originale et la convertir en RGB
    original_img = img[0]
    
    # Si l'image a un seul canal (grayscale), la convertir en RGB
    if original_img.shape[-1] == 1:
        original_img = np.squeeze(original_img)
        original_img = np.stack([original_img] * 3, axis=-1)
    
    # Normaliser l'image d'entrée pour l'affichage
    original_img = np.uint8(255 * original_img)
    
    # Superposer la heatmap sur l'image originale
    superimposed_img = cv2.addWeighted(original_img, 0.6, heatmap, 0.4, 0)
    
    return superimposed_img


def create_prediction_visualization(img, pred_prob, heatmap_img=None):
    """
    Crée une visualisation des prédictions avec la probabilité et la heatmap
    
    Args:
        img: Image originale (après prétraitement)
        pred_prob: Probabilité de pneumonie prédite par le modèle
        heatmap_img: Image heatmap générée par Grad-CAM (optionnel)
    
    Returns:
        bytes: Image de visualisation en format PNG (bytes)
    """
    fig, axes = plt.subplots(1, 2 if heatmap_img is not None else 1, figsize=(12, 6))
    
    # Afficher l'image originale
    if heatmap_img is not None:
        ax1 = axes[0]
    else:
        ax1 = axes
    
    # Si l'image a un seul canal (grayscale), ajuster l'affichage
    if len(img.shape) == 3 and img.shape[-1] == 1:
        img = np.squeeze(img)
        ax1.imshow(img, cmap='gray')
    else:
        ax1.imshow(img)
    
    ax1.set_title("Image originale")
    ax1.axis('off')
    
    # Ajouter des informations sur la prédiction
    pred_class = "Pneumonie" if pred_prob > 0.5 else "Normal"
    text = f"Prédiction: {pred_class}\nProbabilité de pneumonie: {pred_prob:.2f}"
    
    # Définir la couleur du texte en fonction de la prédiction
    text_color = "red" if pred_prob > 0.5 else "green"
    
    ax1.text(0.05, 0.95, text, transform=ax1.transAxes, fontsize=12,
             verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.8),
             color=text_color)
    
    # Afficher la heatmap si disponible
    if heatmap_img is not None:
        axes[1].imshow(cv2.cvtColor(heatmap_img, cv2.COLOR_BGR2RGB))
        axes[1].set_title("Zones d'attention IA (Grad-CAM)")
        axes[1].axis('off')
    
    # Probabilités pour chaque classe sous forme de barre
    class_names = CLASS_NAMES
    class_probs = [1 - pred_prob, pred_prob]  # [Normal, Pneumonie]
    
    # Ajouter un graphique de barres pour les probabilités
    plt.tight_layout()
    plt.subplots_adjust(bottom=0.3)
    
    ax_bar = fig.add_axes([0.1, 0.05, 0.8, 0.15])
    bars = ax_bar.barh([0, 1], class_probs, color=['green', 'red'])
    ax_bar.set_yticks([0, 1])
    ax_bar.set_yticklabels(class_names)
    ax_bar.set_xlim(0, 1)
    ax_bar.set_xlabel('Probabilité')
    ax_bar.set_title('Probabilités par classe')
    
    # Ajouter les valeurs sur les barres
    for bar, prob in zip(bars, class_probs):
        ax_bar.text(min(prob + 0.05, 0.95), bar.get_y() + bar.get_height()/2, 
                   f'{prob:.2f}', va='center')
    
    # Convertir la figure en bytes
    buf = io.BytesIO()
    plt.savefig(buf, format='png', dpi=150, bbox_inches='tight')
    buf.seek(0)
    plt.close(fig)
    
    return buf.getvalue()


def create_comparison_visualization(original_img, preprocessed_img, heatmap_img, pred_prob):
    """
    Crée une visualisation comparative entre l'image originale, prétraitée et la heatmap
    
    Args:
        original_img: Image originale (avant prétraitement)
        preprocessed_img: Image après prétraitement
        heatmap_img: Image heatmap générée par Grad-CAM
        pred_prob: Probabilité de pneumonie prédite par le modèle
    
    Returns:
        bytes: Image de visualisation en format PNG (bytes)
    """
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    # Image originale
    if len(original_img.shape) == 3 and original_img.shape[-1] == 1:
        axes[0].imshow(np.squeeze(original_img), cmap='gray')
    else:
        axes[0].imshow(original_img, cmap='gray')
    axes[0].set_title("Image originale")
    axes[0].axis('off')
    
    # Image prétraitée
    if len(preprocessed_img.shape) == 3 and preprocessed_img.shape[-1] == 1:
        axes[1].imshow(np.squeeze(preprocessed_img), cmap='gray')
    else:
        axes[1].imshow(preprocessed_img, cmap='gray')
    axes[1].set_title("Image prétraitée")
    axes[1].axis('off')
    
    # Heatmap
    axes[2].imshow(cv2.cvtColor(heatmap_img, cv2.COLOR_BGR2RGB))
    axes[2].set_title("Zones d'attention IA")
    axes[2].axis('off')
    
    # Ajouter des informations sur la prédiction
    pred_class = "Pneumonie" if pred_prob > 0.5 else "Normal"
    text = f"Prédiction: {pred_class}\nProbabilité: {pred_prob:.2f}"
    
    # Définir la couleur du texte en fonction de la prédiction
    text_color = "red" if pred_prob > 0.5 else "green"
    
    plt.figtext(0.5, 0.01, text, wrap=True, horizontalalignment='center', fontsize=12,
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.8), color=text_color)
    
    plt.tight_layout()
    
    # Convertir la figure en bytes
    buf = io.BytesIO()
    plt.savefig(buf, format='png', dpi=150, bbox_inches='tight')
    buf.seek(0)
    plt.close(fig)
    
    return buf.getvalue()
