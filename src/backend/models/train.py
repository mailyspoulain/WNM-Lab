import os
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
import matplotlib.pyplot as plt
import numpy as np
import sys

# Ajouter le répertoire parent au path pour pouvoir importer config et model
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import MODEL_PATH, IMAGE_SIZE
from backend.models.modelOLD import create_base_model, create_transfer_learning_model


def train_model(data_dir, model_type='transfer', base_model='vgg16', epochs=20, batch_size=32):
    """
    Entraîne un modèle sur le dataset des radiographies pulmonaires
    
    Args:
        data_dir: Chemin vers le répertoire de données (avec sous-dossiers train/val)
        model_type: 'base' pour le modèle CNN de base, 'transfer' pour transfer learning
        base_model: 'vgg16' ou 'resnet50' si model_type='transfer'
        epochs: Nombre d'époques d'entraînement
        batch_size: Taille des lots (batch size)
    """
    # Vérifier que le répertoire de données existe
    if not os.path.exists(data_dir):
        raise FileNotFoundError(f"Le répertoire de données {data_dir} n'existe pas")
    
    # Vérifier que les sous-répertoires existent
    train_dir = os.path.join(data_dir, 'train')
    val_dir = os.path.join(data_dir, 'val')
    
    if not os.path.exists(train_dir) or not os.path.exists(val_dir):
        raise FileNotFoundError("Les sous-répertoires train et val doivent exister dans le répertoire de données")
    
    # Initialiser les générateurs de données avec augmentation pour l'entraînement
    if model_type == 'base':
        # Pour le modèle de base, nous utilisons des images en niveaux de gris
        train_datagen = ImageDataGenerator(
            rescale=1./255,
            rotation_range=20,
            width_shift_range=0.1,
            height_shift_range=0.1,
            shear_range=0.1,
            zoom_range=0.1,
            horizontal_flip=True,
            fill_mode='nearest'
        )
        
        val_datagen = ImageDataGenerator(rescale=1./255)
        
        color_mode = 'grayscale'
        target_size = IMAGE_SIZE
        
    else:  # Pour le transfer learning
        # Les modèles pré-entraînés attendent des images RGB normalisées
        train_datagen = ImageDataGenerator(
            preprocessing_function=tf.keras.applications.vgg16.preprocess_input,
            rotation_range=20,
            width_shift_range=0.1,
            height_shift_range=0.1,
            shear_range=0.1,
            zoom_range=0.1,
            horizontal_flip=True,
            fill_mode='nearest'
        )
        
        val_datagen = ImageDataGenerator(
            preprocessing_function=tf.keras.applications.vgg16.preprocess_input
        )
        
        color_mode = 'rgb'
        target_size = IMAGE_SIZE
    
    # Générateurs de flux de données
    train_generator = train_datagen.flow_from_directory(
        train_dir,
        target_size=target_size,
        batch_size=batch_size,
        class_mode='binary',
        color_mode=color_mode,
        shuffle=True
    )
    
    validation_generator = val_datagen.flow_from_directory(
        val_dir,
        target_size=target_size,
        batch_size=batch_size,
        class_mode='binary',
        color_mode=color_mode,
        shuffle=False
    )
    
    # Créer le modèle
    if model_type == 'base':
        model = create_base_model()
    else:  # Transfer learning
        model = create_transfer_learning_model(base_model)
    
    # Définir les callbacks
    callbacks = [
        ModelCheckpoint(MODEL_PATH, monitor='val_accuracy', save_best_only=True, verbose=1),
        EarlyStopping(monitor='val_loss', patience=5, verbose=1),
        ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=3, verbose=1)
    ]
    
    # Entraîner le modèle
    history = model.fit(
        train_generator,
        steps_per_epoch=train_generator.samples // batch_size,
        epochs=epochs,
        validation_data=validation_generator,
        validation_steps=validation_generator.samples // batch_size,
        callbacks=callbacks
    )
    
    # Sauvegarder l'historique d'entraînement
    plt.figure(figsize=(12, 4))
    
    # Courbe d'apprentissage - Précision
    plt.subplot(1, 2, 1)
    plt.plot(history.history['accuracy'])
    plt.plot(history.history['val_accuracy'])
    plt.title('Précision du modèle')
    plt.ylabel('Précision')
    plt.xlabel('Époque')
    plt.legend(['Train', 'Validation'], loc='upper left')
    
    # Courbe d'apprentissage - Perte
    plt.subplot(1, 2, 2)
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('Perte du modèle')
    plt.ylabel('Perte')
    plt.xlabel('Époque')
    plt.legend(['Train', 'Validation'], loc='upper left')
    
    plt.tight_layout()
    plt.savefig(os.path.join(os.path.dirname(MODEL_PATH), 'training_history.png'))
    
    return model, history


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Entraîner un modèle de détection de pneumonie')
    parser.add_argument('--data_dir', type=str, required=True, help='Chemin vers le répertoire de données')
    parser.add_argument('--model_type', type=str, default='transfer', choices=['base', 'transfer'], 
                        help='Type de modèle à entraîner')
    parser.add_argument('--base_model', type=str, default='vgg16', choices=['vgg16', 'resnet50'],
                        help='Modèle de base pour le transfer learning')
    parser.add_argument('--epochs', type=int, default=20, help='Nombre d\'époques d\'entraînement')
    parser.add_argument('--batch_size', type=int, default=32, help='Taille des lots')
    
    args = parser.parse_args()
    
    train_model(
        args.data_dir,
        model_type=args.model_type,
        base_model=args.base_model,
        epochs=args.epochs,
        batch_size=args.batch_size
    )
