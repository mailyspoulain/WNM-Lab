import os

# Chemins des répertoires
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(BASE_DIR, 'static', 'models')
UPLOAD_FOLDER = os.path.join(BASE_DIR, 'static', 'uploads')
RESULTS_FOLDER = os.path.join(BASE_DIR, 'static', 'results')

# Création des répertoires s'ils n'existent pas
os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(RESULTS_FOLDER, exist_ok=True)

# Configuration des images
IMAGE_SIZE = (224, 224)  # Taille d'image pour le modèle
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}

# Paramètres du modèle
MODEL_PATH = os.path.join(MODEL_DIR, 'pneumonia_model.h5')
CLASS_NAMES = ['Normal', 'Pneumonie']

# Configuration de Flask
DEBUG = True
SECRET_KEY = 'votre_clé_secrète_ici'  # À changer en production

# Configuration de la base de données
DB_CONFIG = {
    'host': 'localhost',
    'user': 'root',
    'password': '',
    'database': 'medai'
}
