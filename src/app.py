from flask import Flask, request, jsonify
# Importer le module de modèle DenseNet (renommé en model_loader.py)
from backend.models.model_loader import load_chest_xray_model, predict_pneumonia


app = Flask(__name__)

# Charger le modèle DenseNet121 une fois au démarrage de l'application
model = load_chest_xray_model()
if model is None:
    print("Attention : modèle DenseNet121 non chargé. Les prédictions utiliseront le mode fallback.")

@app.route('/predict', methods=['POST'])
def predict():
    # Vérifier qu'un fichier a bien été envoyé
    if 'file' not in request.files:
        return jsonify({"error": "Aucun fichier fourni"}), 400
    file = request.files['file']
    if file.filename == '':
        return jsonify({"error": "Aucun fichier sélectionné"}), 400

    # Utiliser le modèle IA pour prédire à partir de l'image
    result = predict_pneumonia(model, file)

    # Retourner la réponse JSON
    return jsonify(result)

# (Optionnel) Un point de terminaison simple pour vérifier que le serveur fonctionne
@app.route('/', methods=['GET'])
def index():
    return "MedVision AI backend est en cours d'exécution.", 200

if __name__ == '__main__':
    # Lancement de l'application Flask en mode développement (pour tests locaux)
    app.run(host='0.0.0.0', port=5000, debug=True)
