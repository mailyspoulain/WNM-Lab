from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
import sys
import os
import uuid
import logging
from datetime import datetime
from werkzeug.utils import secure_filename

# Ajouter le dossier parent au path pour importer le module modèle
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from backend.models.model_loader_improved import (
    load_chest_xray_model,
    predict_pneumonia_improved,
    ChestXrayValidator
)

# --- Initialisation de l'application Flask ---
app = Flask(__name__)
CORS(app)

# --- Configuration ---
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max
app.config['UPLOAD_FOLDER'] = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'uploads')
app.config['ALLOWED_EXTENSIONS'] = {'png', 'jpg', 'jpeg', 'gif', 'bmp'}

# Créer les dossiers nécessaires
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
os.makedirs(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'logs'), exist_ok=True)

# --- Configuration du logging ---
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'logs', 'app.log')),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# --- Charger le modèle au démarrage ---
logger.info("Chargement du modèle...")
model_info = load_chest_xray_model()
if model_info[0] is None:
    logger.error("Impossible de charger le modèle !")
    raise RuntimeError("Le modèle n'a pas pu être chargé")
logger.info("Modèle chargé avec succès !")

# --- Initialiser les statistiques ---
app_stats = {
    'total_analyses': 0,
    'today_analyses': 0,
    'average_confidence': 0.0,
    'pneumonia_rate': 0.0,
    'pneumonia_detected': 0,
    'normal_detected': 0,
    'rejected_images': 0,
    'high_urgency_cases': 0,
    'confidences': []  # Pour calculer la moyenne
}

# --- Fonctions utilitaires ---
def get_app_stats():
    """Retourne les statistiques actualisées de l'application"""
    stats = app_stats.copy()
    
    # Calculer la moyenne de confiance
    if stats['confidences']:
        stats['average_confidence'] = sum(stats['confidences']) / len(stats['confidences'])
    else:
        stats['average_confidence'] = 0.0
    
    # Calculer le taux de pneumonie
    total = stats['total_analyses']
    if total > 0:
        stats['pneumonia_rate'] = stats['pneumonia_detected'] / total
    else:
        stats['pneumonia_rate'] = 0.0
    
    # Ne pas renvoyer la liste des confidences
    stats.pop('confidences', None)
    
    return stats

def allowed_file(filename):
    """Vérifier si le fichier est autorisé"""
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in app.config['ALLOWED_EXTENSIONS']

def save_uploaded_file(file):
    """Sauvegarder le fichier uploadé avec un nom unique"""
    if file and allowed_file(file.filename):
        ext = file.filename.rsplit('.', 1)[1].lower()
        filename = f"{uuid.uuid4()}.{ext}"
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)
        return filepath
    return None

# --- Routes ---
@app.route('/')
def index():
    """Page d'accueil"""
    return render_template('index.html', stats=get_app_stats())

@app.route('/health', methods=['GET'])
def health_check():
    """Endpoint de santé"""
    return jsonify({
        'status': 'healthy',
        'model_loaded': model_info[0] is not None,
        'timestamp': datetime.now().isoformat(),
        'stats': get_app_stats()
    })

@app.route('/api/analyze', methods=['POST'])
def analyze_image():
    """Analyser une image pour détecter une pneumonie"""
    try:
        # Vérifier qu'une image est fournie
        if 'file' not in request.files:
            return jsonify({
                'status': 'error',
                'message': 'Aucune image fournie'
            }), 400
        
        file = request.files['file']
        if file.filename == '':
            return jsonify({
                'status': 'error',
                'message': 'Aucun fichier sélectionné'
            }), 400
        
        # Sauvegarder le fichier
        filepath = save_uploaded_file(file)
        if not filepath:
            return jsonify({
                'status': 'error',
                'message': 'Type de fichier non autorisé'
            }), 400
        
        logger.info(f"Analyse de l'image: {filepath}")
        
        # Analyser l'image
        result = predict_pneumonia_improved(
            model_info,
            filepath,
            use_validation=False,
            use_tta=True,
            threshold=0.91,
            min_confidence=0.2,
            temperature=1.5,    # ← jouez sur cette valeur (ex: 1.5)
            verbose=False
        )
        
        # Mettre à jour les statistiques
        app_stats['total_analyses'] += 1
        
        # Vérifier si result est None ou si ce n'est pas une radiographie
        if result is None or not result.get('is_chest_xray', True):
            app_stats['rejected_images'] += 1
            logger.warning(f"Image rejetée (non-radiographie): {filepath}")
            
            # Supprimer le fichier
            try:
                os.remove(filepath)
            except:
                pass
            
            # Gérer le cas où result est None
            if result is None:
                recommendation = 'Erreur lors de l\'analyse de l\'image'
                validation_confidence = 0
            else:
                recommendation = result.get('recommendation', 'Veuillez uploader une radiographie pulmonaire valide')
                # Gérer le cas où details pourrait ne pas exister
                details = result.get('details', {})
                validation_confidence = details.get('confidence', 0) if isinstance(details, dict) else 0
            
            return jsonify({
                'status': 'error',
                'message': "L'image ne semble pas être une radiographie pulmonaire",
                'result': {
                    'is_valid': False,
                    'recommendation': recommendation,
                    'validation_confidence': validation_confidence
                }
            }), 400
        
        # Mettre à jour les statistiques selon le résultat
        confidence = result.get('confidence', 0.5)
        app_stats['confidences'].append(confidence)
        
        if result['prediction'] == 'Pneumonie':
            app_stats['pneumonia_detected'] += 1
            if result.get('urgency') == 'haute':
                app_stats['high_urgency_cases'] += 1
        else:
            app_stats['normal_detected'] += 1
        
        logger.info(f"Résultat - Prédiction: {result['prediction']}, "
                   f"Probabilité: {result['probability']:.2%}, "
                   f"Confiance: {confidence:.2%}")
        
        # Préparer la réponse au format attendu par le frontend
        response = {
            'status': 'success',
            'result': {
                'prediction': result['prediction'],
                'probability': float(result['probability']),
                'confidence': float(confidence),
                'urgency': result.get('urgency', 'aucune'),
                'recommendation': result['recommendation'],
                'is_valid': True,
                'details': {
                    'threshold_used': result.get('threshold_used', 0.91),
                    'raw_probability': float(result.get('raw_probability', result['probability'])),
                    'image_quality_score': float(result.get('details', {}).get('image_quality_score', 1.0)),
                    'tta_consistency': float(result.get('details', {}).get('consistency_score', 1.0))
                }
            },
            'timestamp': datetime.now().isoformat(),
            'image_id': os.path.basename(filepath)
        }
        
        # Optionnel : supprimer l'image après analyse
        # os.remove(filepath)
        
        return jsonify(response)
        
    except Exception as e:
        logger.error(f"Erreur lors de l'analyse: {str(e)}", exc_info=True)
        return jsonify({
            'status': 'error',
            'message': 'Erreur lors de l\'analyse de l\'image',
            'details': str(e)
        }), 500

@app.route('/api/validate', methods=['POST'])
def validate_image():
    """Valider si une image est une radiographie"""
    try:
        if 'file' not in request.files:
            return jsonify({
                'status': 'error',
                'message': 'Aucune image fournie'
            }), 400
        
        file = request.files['file']
        filepath = save_uploaded_file(file)
        if not filepath:
            return jsonify({
                'status': 'error',
                'message': 'Type de fichier non autorisé'
            }), 400
        
        # Valider l'image
        try:
            is_xray, details = ChestXrayValidator.is_chest_xray(filepath)
        except Exception as e:
            logger.error(f"Erreur ChestXrayValidator: {str(e)}")
            is_xray = False
            details = {'confidence': 0, 'reason': 'Erreur lors de la validation'}
        
        # Supprimer le fichier
        try:
            os.remove(filepath)
        except:
            pass
        
        # S'assurer que details est un dictionnaire
        if not isinstance(details, dict):
            details = {'confidence': 0, 'reason': 'Format de réponse invalide'}
        
        return jsonify({
            'status': 'success',
            'is_chest_xray': bool(is_xray),
            'confidence': float(details.get('confidence', 0)),
            'reason': str(details.get('reason', '')),
            'features': details.get('features', {})
        })
        
    except Exception as e:
        logger.error(f"Erreur lors de la validation: {str(e)}", exc_info=True)
        return jsonify({
            'status': 'error',
            'message': 'Erreur lors de la validation',
            'details': str(e)
        }), 500

@app.route('/api/stats', methods=['GET'])
def get_stats():
    """Obtenir les statistiques de l'application"""
    return jsonify({
        'status': 'success',
        'stats': get_app_stats(),
        'timestamp': datetime.now().isoformat()
    })

@app.route('/api/batch_analyze', methods=['POST'])
def batch_analyze():
    """Analyser plusieurs images en lot"""
    try:
        files = request.files.getlist('files')
        if not files:
            return jsonify({
                'status': 'error',
                'message': 'Aucune image fournie'
            }), 400
        
        if len(files) > 10:
            return jsonify({
                'status': 'error',
                'message': 'Maximum 10 images par lot'
            }), 400
        
        results = []
        for file in files:
            if file and allowed_file(file.filename):
                filepath = save_uploaded_file(file)
                if filepath:
                    result = predict_pneumonia_improved(
                        model_info,
                        filepath,
                        use_validation=True,
                        use_tta=True,
                        threshold=0.91,
                        min_confidence=0.2,
                        temperature=1.5,    # ← même ici
                        verbose=True
                    )
                    if result and result.get('confidence', 0) < 0.3:
                        logger.warning(f"Confiance faible détectée: {result.get('confidence', 0):.2%}")
                        logger.warning(f"Probabilité de pneumonie: {result.get('probability', 0):.2%}")
                        
                        # Enrichir la recommandation
                        result['recommendation'] = (
                            f"⚠️ Analyse avec confiance faible ({result.get('confidence', 0):.1%})\n\n"
                            f"Probabilité de pneumonie: {result.get('probability', 0):.1%}\n"
                            f"Seuil utilisé: {result.get('threshold_used', 0.91)}\n\n"
                            "Recommandations:\n"
                            "1. Si symptômes cliniques présents, considérer une nouvelle radiographie\n"
                            "2. Consulter un radiologue pour confirmation\n"
                            "3. La qualité de l'image peut affecter la précision du diagnostic"
                        )
                    # Gérer le cas où result est None
                    if result is None:
                        success = False
                        error = 'Erreur lors de l\'analyse'
                    else:
                        success = result.get('is_chest_xray', True)
                        error = result.get('recommendation') if not success else None
                    
                    results.append({
                        'filename': file.filename,
                        'success': success,
                        'result': result if success and result else None,
                        'error': error
                    })
                    
                    try:
                        os.remove(filepath)
                    except:
                        pass
        
        # Résumé
        summary = {
            'total': len(results),
            'analyzed': sum(1 for r in results if r['success']),
            'rejected': sum(1 for r in results if not r['success']),
            'pneumonia_cases': sum(1 for r in results 
                                 if r['success'] and r.get('result') and 
                                 r['result'].get('prediction') == 'Pneumonie')
        }
        
        return jsonify({
            'status': 'success',
            'results': results,
            'summary': summary
        })
        
    except Exception as e:
        logger.error(f"Erreur batch: {str(e)}", exc_info=True)
        return jsonify({
            'status': 'error',
            'message': 'Erreur lors de l\'analyse en lot',
            'details': str(e)
        }), 500

# Gestionnaires d'erreurs
@app.errorhandler(413)
def too_large(e):
    return jsonify({
        'status': 'error',
        'message': 'Fichier trop volumineux. Maximum 16MB.'
    }), 413

@app.errorhandler(500)
def internal_error(e):
    logger.error(f"Erreur interne: {str(e)}", exc_info=True)
    return jsonify({
        'status': 'error',
        'message': 'Erreur interne du serveur'
    }), 500

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)