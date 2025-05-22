from flask import Blueprint, request, jsonify, send_file
import os
import numpy as np
import io
import cv2
import base64
from PIL import Image
import sys
import time
import uuid
import json
from werkzeug.utils import secure_filename

# Ajouter le répertoire parent au path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import UPLOAD_FOLDER, RESULTS_FOLDER, ALLOWED_EXTENSIONS, IMAGE_SIZE, CLASS_NAMES
from utils.preprocessing import read_image_file, preprocess_image, preprocess_xray
from utils.visualization import generate_gradcam, create_prediction_visualization, create_comparison_visualization
from backend.models.model_loader import load_chest_xray_model, predict_pneumonia

# Créer le Blueprint
api = Blueprint('api', __name__, url_prefix='/api')

_model = load_chest_xray_model()
if _model is None:
    # loggers ou print selon ton config
    print("⚠️ Modèle DenseNet non chargé : les prédictions utiliseront le fallback.")

@api.route('/predict', methods=['POST'])
def predict():
    # 1) Vérification du fichier
    if 'file' not in request.files:
        return jsonify({"error": "Aucun fichier fourni"}), 400
    file = request.files['file']
    if file.filename == '':
        return jsonify({"error": "Aucun fichier sélectionné"}), 400

    # 2) Exécution de la prédiction
    result = predict_pneumonia(_model, file)

    # 3) Renvoi JSON
    return jsonify(result)


def allowed_file(filename):
    """Vérifie si l'extension du fichier est autorisée"""
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


@api.before_request
def load_model_if_needed():
    """Charge le modèle s'il n'est pas déjà chargé"""
    global model
    if model is None:
        model = load_model()


@api.route('/api/upload', methods=['POST'])
def upload_file():
    """
    Endpoint pour télécharger et analyser une image
    
    Returns:
        JSON avec les résultats de l'analyse
    """
    # Vérifier si le fichier est dans la requête
    if 'file' not in request.files:
        return jsonify({'error': 'Aucun fichier trouvé'}), 400
    
    file = request.files['file']
    
    # Vérifier si le fichier a un nom
    if file.filename == '':
        return jsonify({'error': 'Aucun fichier sélectionné'}), 400
    
    # Vérifier l'extension du fichier
    if not allowed_file(file.filename):
        return jsonify({'error': f'Format de fichier non pris en charge. Formats autorisés: {", ".join(ALLOWED_EXTENSIONS)}'}), 400
    
    try:
        # Générer un nom de fichier unique
        filename = secure_filename(file.filename)
        unique_filename = f"{str(uuid.uuid4())}_{filename}"
        file_path = os.path.join(UPLOAD_FOLDER, unique_filename)
        
        # Lire et prétraiter l'image
        original_img = read_image_file(file)
        
        # Sauvegarder l'image originale
        file.seek(0)  # Réinitialiser le pointeur du fichier
        file.save(file_path)
        
        # Prétraiter l'image pour le modèle de radiographie
        preprocessed_img = preprocess_xray(original_img)
        
        # Préparer l'image pour l'inférence
        img_input = preprocess_image(preprocessed_img, model.input_shape)
        
        # Prédiction
        start_time = time.time()
        prediction = model.predict(img_input)[0][0]
        inference_time = time.time() - start_time
        
        # Générer la heatmap Grad-CAM
        heatmap_img = generate_gradcam(model, img_input)
        
        # Créer la visualisation
        result_image = create_comparison_visualization(
            original_img, 
            preprocessed_img, 
            heatmap_img, 
            prediction
        )
        
        # Sauvegarder l'image de résultat
        result_filename = f"result_{unique_filename.split('.')[0]}.png"
        result_path = os.path.join(RESULTS_FOLDER, result_filename)
        
        with open(result_path, 'wb') as f:
            f.write(result_image)
        
        # Créer l'objet de résultat
        prediction_result = {
            'prediction': 'Pneumonie' if prediction > 0.5 else 'Normal',
            'probability': float(prediction),
            'probabilities': {
                'Normal': float(1 - prediction),
                'Pneumonie': float(prediction)
            },
            'inference_time': inference_time,
            'original_image': unique_filename,
            'result_image': result_filename
        }
        
        # Ajouter une recommandation en fonction de la prédiction
        if prediction > 0.5:
            if prediction > 0.85:
                recommendation = "Forte probabilité de pneumonie détectée. Consultation médicale fortement recommandée."
            else:
                recommendation = "Signes possibles de pneumonie. Une consultation médicale est recommandée."
        else:
            if prediction < 0.15:
                recommendation = "Aucun signe de pneumonie détecté. Radiographie normale."
            else:
                recommendation = "Probablement normal, mais des anomalies subtiles peuvent être présentes. Consultez un médecin en cas de symptômes persistants."
        
        prediction_result['recommendation'] = recommendation
        
        return jsonify(prediction_result)
    
    except Exception as e:
        return jsonify({'error': f'Erreur lors du traitement de l\'image: {str(e)}'}), 500


@api.route('/api/results/<filename>', methods=['GET'])
def get_result(filename):
    """
    Récupère une image de résultat
    
    Args:
        filename: Nom du fichier de résultat
    """
    file_path = os.path.join(RESULTS_FOLDER, filename)
    
    if not os.path.exists(file_path):
        return jsonify({'error': 'Fichier non trouvé'}), 404
    
    return send_file(file_path, mimetype='image/png')


@api.route('/api/images/<filename>', methods=['GET'])
def get_image(filename):
    """
    Récupère une image originale
    
    Args:
        filename: Nom du fichier d'image
    """
    file_path = os.path.join(UPLOAD_FOLDER, filename)
    
    if not os.path.exists(file_path):
        return jsonify({'error': 'Fichier non trouvé'}), 404
    
    return send_file(file_path)


@api.route('/api/health', methods=['GET'])
def health_check():
    """Endpoint pour vérifier l'état du service"""
    return jsonify({
        'status': 'ok',
        'model_loaded': model is not None
    })


@api.route('/api/model/info', methods=['GET'])
def model_info():
    """Renvoie des informations sur le modèle chargé"""
    global model
    
    if model is None:
        return jsonify({'error': 'Aucun modèle chargé'}), 500
    
    model_info = {
        'input_shape': model.input_shape[1:],
        'output_shape': model.output_shape[1:],
        'number_of_layers': len(model.layers),
        'classes': CLASS_NAMES
    }
    
    return jsonify(model_info)
