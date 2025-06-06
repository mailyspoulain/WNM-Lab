document.addEventListener('DOMContentLoaded', function() {
    // Sélectionner les éléments
    const dropArea = document.querySelector('.drop-area');
    const analyzeBtn = document.querySelector('button[type="submit"]');
    const resetBtn = document.querySelector('button[type="reset"]');
    
    // Vérifier si les éléments existent
    if (dropArea) {
        console.log("Zone de dépôt trouvée");
        
        // Ajouter les événements pour le glisser-déposer
        ['dragenter', 'dragover', 'dragleave', 'drop'].forEach(eventName => {
            dropArea.addEventListener(eventName, preventDefaults, false);
        });
        
        ['dragenter', 'dragover'].forEach(eventName => {
            dropArea.addEventListener(eventName, highlight, false);
        });
        
        ['dragleave', 'drop'].forEach(eventName => {
            dropArea.addEventListener(eventName, unhighlight, false);
        });
        
        dropArea.addEventListener('drop', handleDrop, false);
        
        // Ajouter un événement de clic pour simuler le bouton de téléchargement
        dropArea.addEventListener('click', function() {
            const fileInput = document.createElement('input');
            fileInput.type = 'file';
            fileInput.accept = 'image/*';
            fileInput.onchange = e => {
                if (e.target.files.length) {
                    handleFiles(e.target.files);
                }
            };
            fileInput.click();
        });
    }
    
    // Gérer le bouton d'analyse
    if (analyzeBtn) {
        console.log("Bouton d'analyse trouvé");
        analyzeBtn.addEventListener('click', function(e) {
            e.preventDefault();
            analyzeImage();
        });
    }
    
    // Gérer le bouton de réinitialisation
    if (resetBtn) {
        console.log("Bouton de réinitialisation trouvé");
        resetBtn.addEventListener('click', function(e) {
            e.preventDefault();
            resetInterface();
        });
    }
});

// Fonctions utilitaires
function preventDefaults(e) {
    e.preventDefault();
    e.stopPropagation();
}

function highlight() {
    this.classList.add('highlight');
}

function unhighlight() {
    this.classList.remove('highlight');
}

// Fichier actuellement sélectionné
let currentFile = null;

function handleDrop(e) {
    const dt = e.dataTransfer;
    const files = dt.files;
    
    if (files.length) {
        handleFiles(files);
    }
}

function handleFiles(files) {
    currentFile = files[0];
    previewFile(currentFile);
}

function previewFile(file) {
    // Afficher un aperçu de l'image
    const reader = new FileReader();
    reader.readAsDataURL(file);
    reader.onloadend = function() {
        const img = dropArea.querySelector('img') || document.createElement('img');
        img.src = reader.result;
        img.style.maxWidth = '100%';
        img.style.height = 'auto';
        
        if (!dropArea.querySelector('img')) {
            dropArea.innerHTML = '';
            dropArea.appendChild(img);
            
            const p = document.createElement('p');
            p.textContent = file.name;
            dropArea.appendChild(p);
        }
    }
}

function analyzeImage() {
    if (!currentFile) {
        alert('Veuillez d\'abord sélectionner une image');
        return;
    }
    
    const formData = new FormData();
    formData.append('file', currentFile);
    
    // Ajouter le type d'image sélectionné
    const selectedType = document.querySelector('input[name="image-type"]:checked');
    if (selectedType) {
        formData.append('image_type', selectedType.id);
    }
    
    // Afficher un indicateur de chargement
    document.querySelector('.result-section').innerHTML += '<div class="loading">Analyse en cours...</div>';
    
    // Envoyer la requête au backend
    fetch('/api/upload', {
        method: 'POST',
        body: formData
    })
    .then(response => response.json())
    .then(data => {
        // Supprimer l'indicateur de chargement
        const loading = document.querySelector('.loading');
        if (loading) loading.remove();
        
        // Afficher les résultats
        displayResults(data);
    })
    .catch(error => {
        console.error('Erreur:', error);
        alert('Une erreur est survenue lors de l\'analyse');
        
        // Supprimer l'indicateur de chargement
        const loading = document.querySelector('.loading');
        if (loading) loading.remove();
    });
}

function displayResults(data) {
    // Mettre à jour le résumé du diagnostic
    const diagnosisSummary = document.querySelector('.diagnosis-summary p');
    if (diagnosisSummary) {
        diagnosisSummary.textContent = `L'analyse de cette radiographie pulmonaire indique une ${data.prediction === 'Pneumonie' ? 'forte' : 'faible'} probabilité de pneumonie.`;
    }
    
    // Mettre à jour les barres de probabilité
    updateProbabilityBars(data.probabilities);
    
    console.log('Résultats affichés:', data);
}

function updateProbabilityBars(probabilities) {
    const probItems = document.querySelectorAll('.prob-item');
    
    probItems.forEach(item => {
        const label = item.querySelector('.prob-label').textContent;
        const bar = item.querySelector('.prob-bar');
        const value = item.querySelector('.prob-value');
        
        // Déterminer la probabilité correspondante
        let probability = 0;
        if (label === 'Pneumonie') {
            probability = probabilities.Pneumonie;
        } else if (label === 'Normal') {
            probability = probabilities.Normal;
        }
        
        // Mettre à jour la barre et la valeur
        if (bar) bar.style.width = `${probability * 100}%`;
        if (value) value.textContent = `${Math.round(probability * 100)}%`;
    });
}

function resetInterface() {
    // Réinitialiser le fichier
    currentFile = null;
    
    // Réinitialiser la zone de dépôt
    const dropArea = document.querySelector('.drop-area');
    if (dropArea) {
        dropArea.innerHTML = `
            <img src="/static/upload-icon.png" alt="icône upload">
            <p>Glissez-déposez votre image ici ou cliquez pour sélectionner</p>
        `;
    }
    
    // Réinitialiser les résultats
    const diagnosisSummary = document.querySelector('.diagnosis-summary p');
    if (diagnosisSummary) {
        diagnosisSummary.textContent = 'Téléchargez une image pour obtenir une analyse.';
    }
}