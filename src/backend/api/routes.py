from flask import Blueprint, request, jsonify, send_file
import os, uuid, time
from werkzeug.utils import secure_filename
from PIL import Image
import numpy as np

from config import UPLOAD_FOLDER, RESULTS_FOLDER, ALLOWED_EXTENSIONS, IMAGE_SIZE, CLASS_NAMES
from utils.preprocessing import read_image_file, preprocess_image, preprocess_xray
from utils.visualization import generate_gradcam, create_comparison_visualization
from backend.models.model_loader import load_chest_xray_model, predict_pneumonia

# Blueprint API avec préfixe '/api'
api = Blueprint('api', __name__, url_prefix='/api')

# Charger le modèle DenseNet121 une fois au démarrage
_model = load_chest_xray_model()
if _model is None:
    print("⚠️ Modèle DenseNet non chargé : fallback activé.")


def allowed_file(filename):
    """Vérifie si l'extension du fichier est autorisée"""
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


@api.route('/predict', methods=['POST'])
def predict():
    """
    Prédit la présence de pneumonie à partir d'une radiographie envoyée en 'file'.
    Retourne JSON { prediction, probability, recommendation, source }.
    """
    if 'file' not in request.files:
        return jsonify({"error": "Aucun fichier fourni"}), 400
    file = request.files['file']
    if file.filename == '':
        return jsonify({"error": "Aucun fichier sélectionné"}), 400

    result = predict_pneumonia(_model, file)
    return jsonify(result)


@api.route('/upload', methods=['POST'])
def upload_file():
    """
    Télécharge, analyse et génère un visuel (Grad-CAM + comparaison).
    Retourne JSON avec { original_image, result_image, prediction, probability, recommendation }.
    """
    # Vérification de la présence du fichier
    if 'file' not in request.files:
        return jsonify({'error': 'Aucun fichier trouvé'}), 400
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'Aucun fichier sélectionné'}), 400
    if not allowed_file(file.filename):
        return jsonify({'error': f'Format non pris en charge: {ALLOWED_EXTENSIONS}'}), 400

    try:
        # Enregistrement du fichier original
        filename = secure_filename(file.filename)
        unique_name = f"{uuid.uuid4()}_{filename}"
        path_in = os.path.join(UPLOAD_FOLDER, unique_name)
        file.save(path_in)

        # Prétraitement et prédiction
        original = read_image_file(path_in)
        prepped = preprocess_xray(original)
        tensor = preprocess_image(prepped, _model.input_shape)
        start = time.time()
        prob = _model.predict(tensor)[0][0]
        inf_time = time.time() - start

        # Grad-CAM et visualisation
        heatmap = generate_gradcam(_model, tensor)
        vis = create_comparison_visualization(original, prepped, heatmap, prob)

        # Enregistrement du visuel
        result_name = f"result_{unique_name.split('.')[0]}.png"
        path_out = os.path.join(RESULTS_FOLDER, result_name)
        with open(path_out, 'wb') as f:
            f.write(vis)

        # Construction du JSON de réponse
        pred_label = 'Pneumonie' if prob > 0.5 else 'Normal'
        result = {
            'original_image': unique_name,
            'result_image': result_name,
            'prediction': pred_label,
            'probability': float(prob),
            'recommendation': (
                "Forte probabilité de pneumonie détectée. Consultation médicale fortement recommandée." if prob > 0.85 else
                "Signes possibles de pneumonie. Une consultation médicale est recommandée." if prob > 0.5 else
                "Aucun signe de pneumonie détecté. Radiographie normale." if prob < 0.15 else
                "Probablement normal, mais des anomalies subtiles peuvent être présentes. Consultez un médecin en cas de symptômes persistants."
            ),
            'inference_time': inf_time
        }
        return jsonify(result)

    except Exception as e:
        return jsonify({'error': f"Erreur lors du traitement de l'image: {e}"}), 500


@api.route('/results/<filename>', methods=['GET'])
def get_result(filename):
    path = os.path.join(RESULTS_FOLDER, filename)
    if not os.path.exists(path):
        return jsonify({'error': 'Fichier non trouvé'}), 404
    return send_file(path, mimetype='image/png')


@api.route('/images/<filename>', methods=['GET'])
def get_image(filename):
    path = os.path.join(UPLOAD_FOLDER, filename)
    if not os.path.exists(path):
        return jsonify({'error': 'Fichier non trouvé'}), 404
    return send_file(path)


@api.route('/health', methods=['GET'])
def health_check():
    return jsonify({'status': 'ok', 'model_loaded': _model is not None})


@api.route('/model/info', methods=['GET'])
def model_info():
    if _model is None:
        return jsonify({'error': 'Aucun modèle chargé'}), 500
    return jsonify({
        'input_shape': _model.input_shape[1:],
        'output_shape': _model.output_shape[1:],
        'classes': CLASS_NAMES
    })