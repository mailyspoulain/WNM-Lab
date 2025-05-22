from flask import Flask, render_template, request, redirect, url_for, flash, jsonify
from flask_cors import CORS
import os
import logging
from werkzeug.middleware.proxy_fix import ProxyFix

# Import des configurations
from config import DEBUG, SECRET_KEY, UPLOAD_FOLDER, RESULTS_FOLDER

# Import des routes API
from api.routes import api

# Configuration des logs
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler()
    ]
)

logger = logging.getLogger(__name__)

def create_app():
    """Initialise l'application Flask"""
    app = Flask(__name__, static_folder='static')
    
    # Configuration de l'application
    app.config['SECRET_KEY'] = SECRET_KEY
    app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
    app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16 MB max
    app.config['DEBUG'] = DEBUG
    
    # Appliquer CORS pour permettre les requêtes cross-origin
    CORS(app)
    
    # Ajouter la prise en charge des proxys
    app.wsgi_app = ProxyFix(app.wsgi_app, x_for=1, x_proto=1, x_host=1)
    
    # Enregistrer le blueprint API
    app.register_blueprint(api)
    
    @app.route('/')
    def index():
        """Page d'accueil"""
        return render_template('index.html')
    
    @app.route('/register')
    def register():
        """Page d'inscription"""
        return render_template('register.html')
    
    @app.route('/login')
    def login():
        """Page de connexion"""
        return render_template('login.html')
    
    @app.route('/dashboard')
    def dashboard():
        """Page du tableau de bord"""
        return render_template('dashboard.html')
    
    @app.errorhandler(404)
    def page_not_found(e):
        """Gestionnaire pour les erreurs 404"""
        return render_template('404.html'), 404
    
    @app.errorhandler(500)
    def server_error(e):
        """Gestionnaire pour les erreurs 500"""
        logger.error(f"Erreur serveur: {str(e)}")
        return render_template('500.html'), 500
    
    @app.before_first_request
    def initialize():
        """Initialisation avant la première requête"""
        logger.info("Initialisation de l'application...")
        
        # Vérifier que les dossiers nécessaires existent
        os.makedirs(UPLOAD_FOLDER, exist_ok=True)
        os.makedirs(RESULTS_FOLDER, exist_ok=True)
        
        logger.info("Application initialisée avec succès")
    
    return app

if __name__ == '__main__':
    app = create_app()
    app.run(host='0.0.0.0', port=5000, debug=DEBUG)
