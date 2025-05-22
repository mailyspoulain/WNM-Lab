from flask import Blueprint, request, jsonify, session
import os
import sys
import hashlib
import time
import jwt
import mysql.connector
from functools import wraps

# Ajouter le répertoire parent au path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import DB_CONFIG, SECRET_KEY

# Créer le Blueprint
auth = Blueprint('auth', __name__)

# Clé secrète pour JWT
JWT_SECRET = SECRET_KEY

# Fonction pour établir une connexion à la base de données
def get_db_connection():
    """Établit une connexion à la base de données MySQL"""
    try:
        conn = mysql.connector.connect(
            host=DB_CONFIG['host'],
            user=DB_CONFIG['user'],
            password=DB_CONFIG['password'],
            database=DB_CONFIG['database']
        )
        return conn
    except mysql.connector.Error as err:
        print(f"Erreur de connexion à la base de données: {err}")
        return None

# Middleware pour vérifier l'authentification
def require_auth(f):
    @wraps(f)
    def decorated(*args, **kwargs):
        token = None
        
        # Vérifier si le token est dans l'en-tête Authorization
        if 'Authorization' in request.headers:
            auth_header = request.headers['Authorization']
            if auth_header.startswith('Bearer '):
                token = auth_header.split(' ')[1]
        
        # Vérifier si le token est dans les cookies
        if not token and 'auth_token' in request.cookies:
            token = request.cookies.get('auth_token')
        
        # Si aucun token n'est trouvé
        if not token:
            return jsonify({'error': 'Authentification requise'}), 401
        
        try:
            # Décodage du token JWT
            data = jwt.decode(token, JWT_SECRET, algorithms=['HS256'])
            
            # Vérifier si l'utilisateur existe toujours dans la base de données
            conn = get_db_connection()
            if not conn:
                return jsonify({'error': 'Erreur de connexion à la base de données'}), 500
            
            cursor = conn.cursor(dictionary=True)
            cursor.execute("SELECT * FROM utilisateurs WHERE id = %s", (data['user_id'],))
            current_user = cursor.fetchone()
            cursor.close()
            conn.close()
            
            if not current_user:
                return jsonify({'error': 'Utilisateur non trouvé'}), 401
            
            # Ajouter l'utilisateur à la requête
            request.current_user = current_user
            
        except jwt.ExpiredSignatureError:
            return jsonify({'error': 'Session expirée. Veuillez vous reconnecter'}), 401
        except jwt.InvalidTokenError:
            return jsonify({'error': 'Token invalide. Veuillez vous reconnecter'}), 401
        
        return f(*args, **kwargs)
    
    return decorated


@auth.route('/api/auth/register', methods=['POST'])
def register():
    """
    Endpoint pour l'inscription d'un utilisateur
    
    Attend en JSON:
    {
        "username": string,
        "password": string
    }
    """
    data = request.get_json()
    
    # Vérifier les données requises
    if not data or 'username' not in data or 'password' not in data:
        return jsonify({'error': 'Nom d\'utilisateur et mot de passe requis'}), 400
    
    username = data['username']
    password = data['password']
    
    # Vérifier la longueur du nom d'utilisateur et du mot de passe
    if len(username) < 3:
        return jsonify({'error': 'Le nom d\'utilisateur doit contenir au moins 3 caractères'}), 400
    
    if len(password) < 6:
        return jsonify({'error': 'Le mot de passe doit contenir au moins 6 caractères'}), 400
    
    # Connexion à la base de données
    conn = get_db_connection()
    if not conn:
        return jsonify({'error': 'Erreur de connexion à la base de données'}), 500
    
    cursor = conn.cursor()
    
    try:
        # Vérifier si l'utilisateur existe déjà
        cursor.execute("SELECT * FROM utilisateurs WHERE username = %s", (username,))
        if cursor.fetchone():
            return jsonify({'error': 'Ce nom d\'utilisateur est déjà pris'}), 400
        
        # Hasher le mot de passe
        hashed_password = hashlib.sha256(password.encode()).hexdigest()
        
        # Insérer le nouvel utilisateur
        cursor.execute(
            "INSERT INTO utilisateurs (username, password) VALUES (%s, %s)",
            (username, hashed_password)
        )
        conn.commit()
        
        return jsonify({'message': 'Inscription réussie'}), 201
    
    except Exception as e:
        conn.rollback()
        return jsonify({'error': f'Erreur lors de l\'inscription: {str(e)}'}), 500
    
    finally:
        cursor.close()
        conn.close()


@auth.route('/api/auth/login', methods=['POST'])
def login():
    """
    Endpoint pour la connexion d'un utilisateur
    
    Attend en JSON:
    {
        "username": string,
        "password": string
    }
    """
    data = request.get_json()
    
    # Vérifier les données requises
    if not data or 'username' not in data or 'password' not in data:
        return jsonify({'error': 'Nom d\'utilisateur et mot de passe requis'}), 400
    
    username = data['username']
    password = data['password']
    
    # Connexion à la base de données
    conn = get_db_connection()
    if not conn:
        return jsonify({'error': 'Erreur de connexion à la base de données'}), 500
    
    cursor = conn.cursor(dictionary=True)
    
    try:
        # Vérifier les informations d'identification
        cursor.execute("SELECT * FROM utilisateurs WHERE username = %s", (username,))
        user = cursor.fetchone()
        
        if not user:
            return jsonify({'error': 'Nom d\'utilisateur ou mot de passe incorrect'}), 401
        
        # Hasher le mot de passe fourni pour comparaison
        hashed_password = hashlib.sha256(password.encode()).hexdigest()
        
        if hashed_password != user['password']:
            return jsonify({'error': 'Nom d\'utilisateur ou mot de passe incorrect'}), 401
        
        # Générer un token JWT
        token_payload = {
            'user_id': user['id'],
            'username': user['username'],
            'exp': int(time.time()) + 86400  # Expiration après 24 heures
        }
        
        token = jwt.encode(token_payload, JWT_SECRET, algorithm='HS256')
        
        return jsonify({
            'message': 'Connexion réussie',
            'token': token,
            'user': {
                'id': user['id'],
                'username': user['username']
            }
        })
    
    except Exception as e:
        return jsonify({'error': f'Erreur lors de la connexion: {str(e)}'}), 500
    
    finally:
        cursor.close()
        conn.close()


@auth.route('/api/auth/user', methods=['GET'])
@require_auth
def get_user():
    """Récupère les informations de l'utilisateur connecté"""
    user = request.current_user
    
    return jsonify({
        'id': user['id'],
        'username': user['username']
    })


@auth.route('/api/auth/logout', methods=['POST'])
def logout():
    """Déconnecte l'utilisateur"""
    # Comme nous utilisons des JWT, il n'y a pas vraiment de "déconnexion" côté serveur.
    # Nous pouvons simplement renvoyer un message pour que le frontend supprime le token.
    return jsonify({'message': 'Déconnexion réussie'})
