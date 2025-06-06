import os

# Répertoire racine du projet (contenant app.py et config.py)
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Dossier contenant les poids du modèle PyTorch
MODEL_DIR = os.path.join(BASE_DIR, 'backend', 'models')
# Chemin absolu vers le fichier de poids .tar
MODEL_PATH = os.path.join(MODEL_DIR, 'model.pth.tar')

# Dossiers d'upload et de résultats (pour ton endpoint /upload)
UPLOAD_FOLDER = os.path.join(BASE_DIR, 'static', 'uploads')
RESULTS_FOLDER = os.path.join(BASE_DIR, 'static', 'results')

# Création automatique si nécessaire
os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(RESULTS_FOLDER, exist_ok=True)

# Configuration des images
IMAGE_SIZE = (224, 224)
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}

# Noms des classes pour la prédiction
CLASS_NAMES = ['Normal', 'Pneumonie']

# Configuration Flask
DEBUG = True
SECRET_KEY = 'change-moi-en-production'

# (Éventuelle config base de données conservée si nécessaire)
DB_CONFIG = {
    'host': 'localhost',
    'user': 'root',
    'password': '',
    'database': 'medai'
}
