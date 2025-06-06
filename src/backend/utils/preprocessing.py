import cv2
import numpy as np
import tensorflow as tf
from PIL import Image
import io
import os
import sys

# Ajouter le répertoire parent au path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import IMAGE_SIZE


def read_image_file(image_file):
    """
    Lit une image à partir d'un fichier uploadé
    
    Args:
        image_file: Fichier image (objet de type FileStorage de Flask)
    
    Returns:
        np.array: Image sous forme de tableau numpy
    """
    # Lire les données de l'image
    img_bytes = image_file.read()
    
    # Convertir en tableau numpy via PIL
    img = Image.open(io.BytesIO(img_bytes))
    
    # Convertir en niveau de gris si ce n'est pas déjà le cas
    if img.mode != 'L':
        img = img.convert('L')
    
    # Convertir en numpy array
    img_array = np.array(img)
    
    return img_array


def preprocess_image(img, model_input_shape):
    """
    Prétraite une image pour l'inférence
    
    Args:
        img: Image sous forme de tableau numpy
        model_input_shape: Forme d'entrée attendue par le modèle
    
    Returns:
        np.array: Image prétraitée prête pour l'inférence
    """
    # Redimensionner l'image
    img_resized = cv2.resize(img, IMAGE_SIZE)
    
    # Normaliser les valeurs de pixels
    img_normalized = img_resized / 255.0
    
    # Ajuster les dimensions selon ce qu'attend le modèle
    if len(model_input_shape) == 4:  # Modèle attend batch, height, width, channels
        if model_input_shape[-1] == 1:  # Grayscale
            img_normalized = np.expand_dims(img_normalized, axis=-1)
        elif model_input_shape[-1] == 3:  # RGB
            # Convertir grayscale à RGB en dupliquant les canaux
            img_normalized = np.stack([img_normalized] * 3, axis=-1)
    
    # Ajouter la dimension de batch
    img_batch = np.expand_dims(img_normalized, axis=0)
    
    return img_batch


def apply_clahe(img, clip_limit=2.0, tile_grid_size=(8, 8)):
    """
    Applique une égalisation d'histogramme adaptative avec contraste limité (CLAHE)
    pour améliorer le contraste et la visibilité des détails dans l'image
    
    Args:
        img: Image sous forme de tableau numpy
        clip_limit: Limite de contraste pour CLAHE
        tile_grid_size: Taille de la grille de tuiles
    
    Returns:
        np.array: Image traitée avec CLAHE
    """
    # Créer l'objet CLAHE
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_grid_size)
    
    # Appliquer CLAHE
    enhanced_img = clahe.apply(img.astype(np.uint8))
    
    return enhanced_img


def denoise_image(img, filter_strength=10):
    """
    Applique un débruitage à l'image pour améliorer la qualité
    
    Args:
        img: Image sous forme de tableau numpy
        filter_strength: Force du filtre de débruitage
    
    Returns:
        np.array: Image débruitée
    """
    # Débruitage non local means
    denoised = cv2.fastNlMeansDenoising(img.astype(np.uint8), None, filter_strength, 7, 21)
    
    return denoised


def preprocess_xray(img):
    """
    Prétraitement complet pour une radiographie pulmonaire
    
    Args:
        img: Image sous forme de tableau numpy
    
    Returns:
        np.array: Image prétraitée
    """
    # Appliquer une série de prétraitements pour améliorer la qualité de l'image
    # 1. Débruitage
    img_denoised = denoise_image(img)
    
    # 2. Amélioration du contraste avec CLAHE
    img_enhanced = apply_clahe(img_denoised)
    
    return img_enhanced


def augment_image_for_visualization(img):
    """
    Augmente une image pour la visualisation (non utilisé pour l'inférence)
    
    Args:
        img: Image sous forme de tableau numpy
    
    Returns:
        np.array: Image augmentée
    """
    # Créer une copie pour ne pas modifier l'original
    img_aug = img.copy()
    
    # Appliquer une rotation aléatoire
    angle = np.random.uniform(-20, 20)
    height, width = img.shape[:2]
    rotation_matrix = cv2.getRotationMatrix2D((width/2, height/2), angle, 1)
    img_aug = cv2.warpAffine(img_aug, rotation_matrix, (width, height))
    
    # Appliquer un zoom aléatoire
    zoom = np.random.uniform(0.8, 1.2)
    new_height, new_width = int(height*zoom), int(width*zoom)
    
    # Calculer les coordonnées de découpage
    top = max(0, int((new_height - height) / 2))
    left = max(0, int((new_width - width) / 2))
    
    if zoom > 1:
        img_aug = cv2.resize(img_aug, (new_width, new_height))
        img_aug = img_aug[top:top+height, left:left+width]
    else:
        # Padding si zoom < 1
        top_pad = max(0, int((height - new_height) / 2))
        left_pad = max(0, int((width - new_width) / 2))
        img_aug = cv2.resize(img_aug, (new_width, new_height))
        img_aug = cv2.copyMakeBorder(img_aug, top_pad, height-new_height-top_pad, 
                                      left_pad, width-new_width-left_pad, 
                                      cv2.BORDER_CONSTANT, value=0)
    
    return img_aug
