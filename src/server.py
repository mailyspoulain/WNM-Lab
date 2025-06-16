from flask import Flask, jsonify
import subprocess
import os

app = Flask(__name__)

@app.route('/launch')
def launch_command():
    try:
        # Lancer votre commande
        subprocess.Popen(['python', 'app.py'], cwd=os.getcwd())
        return jsonify({"status": "success", "message": "Commande lancée"})
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)})

if __name__ == '__main__':
    app.run(port=5000, debug=True)