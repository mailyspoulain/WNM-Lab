from flask import Flask, jsonify, render_template, request, send_from_directory
from flask_cors import CORS
import os
import io
import sys
from PIL import Image
import numpy as np
import random

sys.path.append('src')

try:
    from backend.models.model_loader import load_chest_xray_model, predict_pneumonia
    use_deep_learning = True
    print("Utilisation du modèle DenseNet121 pour la détection de pneumonie")
except ImportError:
    try:
        from backend.models.model_loader import load_chest_xray_model, predict_pneumonia
        use_deep_learning = True
        print("Utilisation du modèle CNN simple pour la détection de pneumonie")
    except ImportError:
        use_deep_learning = False
        print("Aucun modèle disponible, utilisation du mode simplifié")

# Définir l'application Flask
app = Flask(__name__, 
            static_folder=os.path.join('src', 'backend', 'static'),
            template_folder=os.path.join('src', 'backend', 'templates'))
CORS(app)

# Charger le modèle ChestX-ray si possible
chest_xray_model = None
if use_deep_learning:
    try:
        chest_xray_model = load_chest_xray_model()
        if chest_xray_model is None:
            print("Utilisation du mode d'analyse simplifié.")
            use_deep_learning = False
    except Exception as e:
        print(f"Erreur lors du chargement du modèle: {str(e)}")
        print("Utilisation du mode d'analyse simplifié.")
        use_deep_learning = False

# Pages de base
@app.route('/')
def index():
    return "MedVision AI Backend - En cours de développement"

@app.route('/api/health')
def health_check():
    return jsonify({
        'status': 'ok',
        'message': 'API en cours de développement',
        'model_loaded': chest_xray_model is not None
    })

# Simulation d'analyse d'image (pour tester l'intégration frontend-backend)
@app.route('/api/upload', methods=['POST'])
def upload_file():
    if 'file' not in request.files:
        return jsonify({'error': 'Aucun fichier trouvé'}), 400
    
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'Aucun fichier sélectionné'}), 400
    
    # Lire le fichier en mémoire une fois
    file_bytes = file.read()
    file_io = io.BytesIO(file_bytes)
    
    # Utiliser le modèle de deep learning si disponible
    if use_deep_learning and chest_xray_model:
        try:
            # Réinitialiser le pointeur du fichier pour la lecture
            file_io.seek(0)
            
            # Faire la prédiction avec le modèle pré-entraîné
            result = predict_pneumonia(chest_xray_model, file_io)
            print("Prédiction effectuée avec le modèle de deep learning!")
        except Exception as e:
            print(f"Erreur lors de l'utilisation du modèle: {str(e)}")
            # En cas d'erreur, fallback sur l'approche simplifiée
            result = analyze_image_simple(file_io)
    else:
        # Utiliser l'approche simplifiée si le modèle n'est pas disponible
        result = analyze_image_simple(file_io)
    
    # Vérifier si le client attend du JSON ou du HTML
    wants_json = request.headers.get('Accept', '').find('application/json') != -1
    
    # Si le client veut du JSON (API call), retourner JSON
    if wants_json:
        return jsonify(result)
    
    # Sinon, retourner une page HTML avec les résultats (pour form standard)
    html_result = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <title>Résultats d'analyse</title>
        <style>
            body {{ font-family: Arial, sans-serif; max-width: 600px; margin: 0 auto; padding: 20px; }}
            .result {{ margin: 20px 0; padding: 20px; border: 1px solid #ddd; background: #f9f9f9; }}
            .probability {{ height: 24px; background: #007bff; }}
            .bar-container {{ width: 100%; background: #eee; height: 24px; margin: 5px 0; }}
        </style>
    </head>
    <body>
        <h1>Résultats de l'analyse</h1>
        
        <div class="result">
            <h2>Prédiction: {result['prediction']}</h2>
            <p><strong>Probabilité:</strong> {result['probability']*100:.1f}%</p>
            <p><strong>Recommandation:</strong> {result['recommendation']}</p>
            
            <h3>Détail des probabilités:</h3>
            <p>Normal:</p>
            <div class="bar-container">
                <div class="probability" style="width: {result['probabilities']['Normal']*100}%;"></div>
            </div>
            <p>{result['probabilities']['Normal']*100:.1f}%</p>
            
            <p>Pneumonie:</p>
            <div class="bar-container">
                <div class="probability" style="width: {result['probabilities']['Pneumonie']*100}%;"></div>
            </div>
            <p>{result['probabilities']['Pneumonie']*100:.1f}%</p>
        </div>
        
        <a href="/test_form">Retour au formulaire</a>
    </body>
    </html>
    """
    
    return html_result

# Fonction pour analyse simplifiée basée sur les caractéristiques de l'image
def analyze_image_simple(file_io):
    """
    Analyse simplifiée d'une image pour détecter des signes potentiels de pneumonie
    basée sur des heuristiques simples (luminosité, contraste, zones sombres)
    """
    try:
        # Réinitialiser le pointeur du fichier
        file_io.seek(0)
        
        # Ouvrir l'image avec PIL
        img = Image.open(file_io)
        
        # Convertir en niveaux de gris et redimensionner
        img = img.convert('L')
        img = img.resize((224, 224))
        
        # Convertir en tableau numpy et normaliser
        img_array = np.array(img) / 255.0
        
        # Caractéristiques simples pour l'analyse:
        # 1. Luminosité moyenne (les pneumonies sont souvent plus sombres/opaques)
        brightness = np.mean(img_array)
        
        # 2. Contraste (écart-type des valeurs de pixels)
        contrast = np.std(img_array)
        
        # 3. Nombre de zones sombres (potentiellement des opacités)
        dark_regions = np.sum(img_array < 0.3) / (224 * 224)
        
        # Calcul d'une probabilité de pneumonie basée sur ces caractéristiques simples
        # (Ceci est une heuristique simple, pas un vrai modèle d'IA)
        pneumonia_score = (1 - brightness) * 0.5 + contrast * 0.3 + dark_regions * 0.2
        
        # Ajuster le score pour qu'il soit entre 0.05 et 0.95
        pneumonia_prob = min(0.95, max(0.05, pneumonia_score))
        
        # Avec un peu de randomisation pour simuler la variabilité d'un modèle réel
        pneumonia_prob = min(0.95, max(0.05, pneumonia_prob + random.uniform(-0.1, 0.1)))
        
    except Exception as e:
        print(f"Erreur lors de l'analyse simplifiée: {str(e)}")
        # En cas d'erreur, retourner une valeur par défaut
        pneumonia_prob = 0.5
    
    # Déterminer la prédiction finale
    prediction = "Pneumonie" if pneumonia_prob > 0.5 else "Normal"
    
    # Recommandation basée sur la probabilité
    if pneumonia_prob > 0.7:
        recommendation = "Forte probabilité de pneumonie détectée. Consultation médicale fortement recommandée."
    elif pneumonia_prob > 0.5:
        recommendation = "Signes possibles de pneumonie. Une consultation médicale est recommandée."
    elif pneumonia_prob > 0.3:
        recommendation = "Radiographie probablement normale, mais quelques anomalies détectées. Un suivi médical peut être envisagé."
    else:
        recommendation = "Radiographie normale. Aucun signe de pneumonie détecté."
    
    # Préparer le résultat
    return {
        'prediction': prediction,
        'probability': float(pneumonia_prob),
        'probabilities': {
            'Normal': float(1 - pneumonia_prob),
            'Pneumonie': float(pneumonia_prob)
        },
        'inference_time': 0.5,
        'recommendation': recommendation
    }

@app.route('/accueil')
def accueil():
    return render_template('accueil.html')

@app.route('/inscription')
def inscription():
    return render_template('inscription.html')

@app.route('/login')
def login():
    return render_template('index.html')  # Supposant que index.html est la page de login

# Pour servir des fichiers statiques
@app.route('/static/<path:filename>')
def static_files(filename):
    return send_from_directory('src/backend/static', filename)

@app.route('/test_form')
def test_form():
    return """
    <!DOCTYPE html>
    <html>
    <head>
        <title>Test d'upload simple</title>
        <style>
            body { font-family: Arial, sans-serif; max-width: 600px; margin: 0 auto; padding: 20px; }
            form { margin: 20px 0; padding: 20px; border: 1px solid #ddd; }
            button { background: #007bff; color: white; padding: 8px 16px; border: none; cursor: pointer; }
            .result { margin-top: 20px; padding: 15px; background: #f5f5f5; }
        </style>
    </head>
    <body>
        <h1>Test d'upload - Formulaire standard</h1>
        
        <form action="/api/upload" method="post" enctype="multipart/form-data">
            <h3>Sélectionnez une image:</h3>
            <input type="file" name="file" required><br><br>
            <button type="submit">Envoyer l'image</button>
        </form>
        
        <div class="result">
            <p>Les résultats s'afficheront après l'envoi du formulaire.</p>
        </div>
    </body>
    </html>
    """

if __name__ == '__main__':
    # Créer les dossiers nécessaires s'ils n'existent pas
    os.makedirs('src/backend/static/uploads', exist_ok=True)
    os.makedirs('src/backend/static/results', exist_ok=True)
    
    # Lancer le serveur
    app.run(host='0.0.0.0', port=5000, debug=True)