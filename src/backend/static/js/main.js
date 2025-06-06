// Variables globales
const API_URL = '/api';
let currentFile = null;

// Éléments DOM
const dropArea = document.querySelector('.drop-area');
const analyzeBtn = document.querySelector('.btn');
const resetBtn = document.querySelector('.btn-danger');
const loadingOverlay = document.createElement('div');
loadingOverlay.className = 'loading-overlay';
loadingOverlay.innerHTML = `
    <div class="spinner"></div>
    <p>Analyse en cours...</p>
`;

// Ajouter les écouteurs d'événements au chargement du DOM
document.addEventListener('DOMContentLoaded', function() {
    initDropArea();
    initButtons();
    initTabs();
});

// Initialiser la zone de dépôt
function initDropArea() {
    if (!dropArea) return;
    
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
    dropArea.addEventListener('click', () => {
        // Créer un input file invisible et le déclencher
        const fileInput = document.createElement('input');
        fileInput.type = 'file';
        fileInput.accept = 'image/*';
        fileInput.onchange = e => {
            if (e.target.files.length) {
                currentFile = e.target.files[0];
                previewFile(currentFile);
            }
        };
        fileInput.click();
    });
}

// Initialiser les boutons
function initButtons() {
    if (analyzeBtn) {
        analyzeBtn.addEventListener('click', analyzeImage);
    }
    
    if (resetBtn) {
        resetBtn.addEventListener('click', resetInterface);
    }
}

// Initialiser les onglets
function initTabs() {
    const tabs = document.querySelectorAll('.tab');
    if (!tabs.length) return;
    
    tabs.forEach(tab => {
        tab.addEventListener('click', () => {
            // Supprimer la classe 'active' de tous les onglets
            tabs.forEach(t => t.classList.remove('active'));
            
            // Ajouter la classe 'active' à l'onglet cliqué
            tab.classList.add('active');
            
            // Afficher la section correspondante (à implémenter si nécessaire)
            // showSection(tab.textContent.toLowerCase());
        });
    });
}

// Éviter les comportements par défaut
function preventDefaults(e) {
    e.preventDefault();
    e.stopPropagation();
}

// Mettre en évidence la zone de dépôt
function highlight() {
    dropArea.classList.add('highlight');
}

// Supprimer la mise en évidence
function unhighlight() {
    dropArea.classList.remove('highlight');
}

// Gérer le dépôt de fichier
function handleDrop(e) {
    const dt = e.dataTransfer;
    const files = dt.files;
    
    if (files.length) {
        currentFile = files[0];
        previewFile(currentFile);
    }
}

// Afficher un aperçu du fichier
function previewFile(file) {
    // Vérifier si le fichier est une image
    if (!file.type.match('image.*')) {
        alert('Veuillez sélectionner une image');
        return;
    }
    
    // Afficher l'aperçu
    const reader = new FileReader();
    reader.onload = e => {
        // Remplacer l'icône par l'aperçu de l'image
        const img = dropArea.querySelector('img');
        if (img) {
            img.src = e.target.result;
            img.style.width = '100%';
            img.style.height = 'auto';
            img.style.maxHeight = '200px';
            img.style.objectFit = 'contain';
        }
        
        // Mettre à jour le texte
        const text = dropArea.querySelector('p');
        if (text) {
            text.textContent = file.name;
        }
    };
    reader.readAsDataURL(file);
}

// Analyser l'image
function analyzeImage() {
    if (!currentFile) {
        alert('Veuillez d\'abord sélectionner une image');
        return;
    }
    
    // Afficher l'overlay de chargement
    const resultSection = document.querySelector('.result-section');
    if (resultSection) {
        resultSection.appendChild(loadingOverlay);
    }
    
    // Créer le FormData
    const formData = new FormData();
    formData.append('file', currentFile);
    
    // Ajouter le type d'image sélectionné
    const imageTypeInputs = document.querySelectorAll('input[name="image-type"]');
    let selectedType = '';
    imageTypeInputs.forEach(input => {
        if (input.checked) {
            selectedType = input.id;
        }
    });
    formData.append('image_type', selectedType);
    
    // Faire la requête API
    fetch(`${API_URL}/upload`, {
        method: 'POST',
        body: formData
    })
    .then(response => {
        if (!response.ok) {
            throw new Error(`Erreur HTTP: ${response.status}`);
        }
        return response.json();
    })
    .then(data => {
        // Supprimer l'overlay de chargement
        if (resultSection && resultSection.contains(loadingOverlay)) {
            resultSection.removeChild(loadingOverlay);
        }
        
        // Afficher les résultats
        displayResults(data);
    })
    .catch(error => {
        console.error('Erreur:', error);
        
        // Supprimer l'overlay de chargement
        if (resultSection && resultSection.contains(loadingOverlay)) {
            resultSection.removeChild(loadingOverlay);
        }
        
        alert(`Une erreur est survenue: ${error.message}`);
    });
}

// Afficher les résultats
function displayResults(data) {
    // Mettre à jour les images
    const originalImg = document.querySelector('.image-box:first-child img');
    const heatmapImg = document.querySelector('.image-box:last-child img');
    
    if (originalImg) {
        originalImg.src = `${API_URL}/images/${data.original_image}`;
    }
    
    if (heatmapImg) {
        heatmapImg.src = `${API_URL}/results/${data.result_image}`;
    }
    
    // Mettre à jour le résumé du diagnostic
    const diagnosisSummary = document.querySelector('.diagnosis-summary p');
    if (diagnosisSummary) {
        diagnosisSummary.textContent = `L'analyse de cette radiographie pulmonaire indique une ${data.prediction === 'Pneumonie' ? 'forte' : 'faible'} probabilité de pneumonie.`;
    }
    
    // Mettre à jour la recommandation
    const recommendation = document.querySelector('.diagnosis-summary p:last-of-type');
    if (recommendation) {
        recommendation.textContent = data.recommendation;
    }
    
    // Mettre à jour les probabilités
    updateProbabilityBars(data.probabilities);
}

// Mettre à jour les barres de probabilité
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
        } else if (label === 'Tuberculose') {
            // Pour l'exemple, nous n'avons pas de prédiction pour la tuberculose
            probability = 0.03;
        }
        
        // Mettre à jour la barre et la valeur
        bar.style.width = `${probability * 100}%`;
        value.textContent = `${Math.round(probability * 100)}%`;
    });
}

// Réinitialiser l'interface
function resetInterface() {
    // Réinitialiser le fichier courant
    currentFile = null;
    
    // Réinitialiser la zone de dépôt
    const img = dropArea.querySelector('img');
    if (img) {
        img.src = '/api/placeholder/100/100';
        img.style.width = '';
        img.style.height = '';
        img.style.maxHeight = '';
    }
    
    const text = dropArea.querySelector('p');
    if (text) {
        text.textContent = 'Glissez-déposez votre image ici ou cliquez pour sélectionner';
    }
    
    // Réinitialiser les résultats
    const originalImg = document.querySelector('.image-box:first-child img');
    const heatmapImg = document.querySelector('.image-box:last-child img');
    
    if (originalImg) {
        originalImg.src = '/api/placeholder/300/250';
    }
    
    if (heatmapImg) {
        heatmapImg.src = '/api/placeholder/300/250';
    }
    
    // Réinitialiser le résumé du diagnostic
    const diagnosisSummary = document.querySelector('.diagnosis-summary p');
    if (diagnosisSummary) {
        diagnosisSummary.textContent = 'Téléchargez une image pour obtenir une analyse.';
    }
    
    // Réinitialiser la recommandation
    const recommendation = document.querySelector('.diagnosis-summary p:last-of-type');
    if (recommendation) {
        recommendation.textContent = 'En attente d\'analyse...';
    }
    
    // Réinitialiser les probabilités
    const probItems = document.querySelectorAll('.prob-item');
    probItems.forEach(item => {
        const bar = item.querySelector('.prob-bar');
        const value = item.querySelector('.prob-value');
        
        bar.style.width = '0%';
        value.textContent = '0%';
    });
}
