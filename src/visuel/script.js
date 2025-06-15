console.log("JS MedAI chargé - Version Fix Ultimate");

// BLOQUER TOUT RECHARGEMENT DE PAGE
let analysisInProgress = false;
window.addEventListener('beforeunload', function(e) {
    if (analysisInProgress) {
        e.preventDefault();
        e.returnValue = 'Analyse en cours, êtes-vous sûr de vouloir quitter ?';
        return 'Analyse en cours...';
    }
});

// BLOQUER TOUTE SOUMISSION DE FORMULAIRE
document.addEventListener('submit', function(e) {
    console.warn('TENTATIVE DE SOUMISSION BLOQUÉE!', e.target);
    e.preventDefault();
    e.stopPropagation();
    return false;
}, true); // true = capture phase

// Script principal
document.addEventListener('DOMContentLoaded', () => {
    console.log('DOM chargé, initialisation...');
    
    // Vérifier si on a des résultats stockés (après un refresh)
    const storedState = sessionStorage.getItem('analysisState');
    if (storedState === 'success') {
        const storedResults = sessionStorage.getItem('analysisResults');
        if (storedResults) {
            try {
                const results = JSON.parse(storedResults);
                console.log('Résultats trouvés dans sessionStorage, affichage...');
                // Attendre un peu pour que la page soit bien chargée
                setTimeout(() => {
                    const resultContainer = document.getElementById('resultContainer');
                    if (resultContainer && window.displayResults) {
                        window.displayResults(results);
                        // Nettoyer le storage
                        sessionStorage.removeItem('analysisState');
                        sessionStorage.removeItem('analysisResults');
                    }
                }, 500);
            } catch (e) {
                console.error('Erreur lors de la récupération des résultats:', e);
            }
        }
    } else if (storedState === 'error') {
        const errorMsg = sessionStorage.getItem('analysisError');
        setTimeout(() => {
            const resultContainer = document.getElementById('resultContainer');
            if (resultContainer) {
                resultContainer.innerHTML = `
                    <div style="background: #f8d7da; color: #721c24; padding: 1rem; border-radius: 4px;">
                        <h4>Erreur</h4>
                        <p>${errorMsg || 'Une erreur est survenue'}</p>
                    </div>
                `;
            }
            sessionStorage.removeItem('analysisState');
            sessionStorage.removeItem('analysisError');
        }, 500);
    }
    
    // Chercher et désactiver tous les formulaires
    const forms = document.querySelectorAll('form');
    console.log(`Nombre de formulaires trouvés: ${forms.length}`);
    forms.forEach((form, index) => {
        console.log(`Formulaire ${index} trouvé:`, form);
        form.addEventListener('submit', (e) => {
            e.preventDefault();
            e.stopPropagation();
            console.log('Formulaire bloqué!');
            return false;
        });
    });
    
    const uploadArea      = document.getElementById('uploadArea');
    const fileInput       = document.getElementById('fileInput');
    const browseBtn       = document.getElementById('browseBtn');
    const analyzeBtn      = document.getElementById('analyzeBtn');
    const patientInput    = document.getElementById('patient-id');
    const notesInput      = document.getElementById('notes');
    const anonCheckbox    = document.getElementById('anonymize');
    const resultContainer = document.getElementById('resultContainer');

    console.log('Éléments trouvés:', {
        uploadArea: !!uploadArea,
        fileInput: !!fileInput,
        browseBtn: !!browseBtn,
        analyzeBtn: !!analyzeBtn,
        resultContainer: !!resultContainer
    });

    // Vérifier le type du bouton
    if (analyzeBtn) {
        console.log('Type du bouton analyzeBtn:', analyzeBtn.type);
        console.log('Attributs du bouton:', analyzeBtn.attributes);
        
        // FORCER le type button
        analyzeBtn.type = 'button';
        analyzeBtn.setAttribute('type', 'button');
    }

    let selectedFile = null;

    if (uploadArea && fileInput && browseBtn && analyzeBtn) {
        console.log('Configuration des événements...');

        // Upload area
        uploadArea.addEventListener('click', (e) => {
            if (e.target.id !== 'browseBtn' && e.target.tagName !== 'BUTTON') {
                fileInput.click();
            }
        });
        
        // Browse button
        browseBtn.addEventListener('click', (e) => {
            e.preventDefault();
            e.stopPropagation();
            fileInput.click();
            return false;
        });

        // Drag & Drop
        ['dragenter', 'dragover', 'dragleave', 'drop'].forEach(eventName => {
            uploadArea.addEventListener(eventName, preventDefaults, false);
        });

        function preventDefaults(e) {
            e.preventDefault();
            e.stopPropagation();
        }

        ['dragenter', 'dragover'].forEach(eventName => {
            uploadArea.addEventListener(eventName, () => uploadArea.classList.add('dragover'));
        });

        ['dragleave', 'drop'].forEach(eventName => {
            uploadArea.addEventListener(eventName, () => uploadArea.classList.remove('dragover'));
        });

        uploadArea.addEventListener('drop', (e) => {
            const files = e.dataTransfer.files;
            if (files.length > 0) {
                handleFileSelect(files[0]);
            }
        });

        // File input change
        fileInput.addEventListener('change', (e) => {
            if (e.target.files.length > 0) {
                handleFileSelect(e.target.files[0]);
            }
        });

        // Gestion de la sélection de fichier
        function handleFileSelect(file) {
            console.log('Fichier sélectionné:', file.name);
            
            const validTypes = ['image/png', 'image/jpeg', 'image/jpg', 'image/gif', 'image/bmp'];
            if (!validTypes.includes(file.type)) {
                alert('Format non supporté. Utilisez PNG, JPEG, GIF ou BMP');
                return;
            }
            
            selectedFile = file;
            
            const reader = new FileReader();
            reader.onload = function(e) {
                uploadArea.innerHTML = `
                    <div style="text-align: center;">
                        <img src="${e.target.result}" style="max-width: 200px; max-height: 200px; margin-bottom: 10px;" alt="Preview">
                        <p><strong>${file.name}</strong></p>
                        <p style="font-size: 0.9em; color: #666;">${(file.size / 1024).toFixed(2)} KB</p>
                    </div>
                `;
            };
            reader.readAsDataURL(file);
            
            analyzeBtn.disabled = false;
        }

        // ÉVÉNEMENT PRINCIPAL - ANALYSE
        analyzeBtn.addEventListener('click', async function(e) {
            console.log('Bouton analyse cliqué');
            e.preventDefault();
            e.stopPropagation();
            
            // Appeler directement la fonction d'analyse
            await performAnalysis();
            
            return false;
        });

        // Fonction d'analyse
        async function performAnalysis() {
            console.log('=== DÉBUT ANALYSE (performAnalysis) ===');
            
            if (!selectedFile) {
                alert('Veuillez sélectionner une image');
                return;
            }

            // Bloquer le refresh pendant l'analyse
            analysisInProgress = true;

            const formData = new FormData();
            formData.append('file', selectedFile);
            formData.append('patient_id', patientInput?.value || '');
            formData.append('notes', notesInput?.value || '');
            formData.append('anonymize', anonCheckbox?.checked || false);

            // UI
            analyzeBtn.disabled = true;
            const originalText = analyzeBtn.textContent;
            analyzeBtn.textContent = 'Analyse en cours...';
            
            // Stocker l'état dans sessionStorage pour persister après refresh
            sessionStorage.setItem('analysisState', 'loading');
            
            resultContainer.innerHTML = `
                <div style="text-align: center; padding: 2rem; background: #f8f9fa; border-radius: 8px;">
                    <div style="width: 50px; height: 50px; border: 5px solid #f3f3f3; border-top: 5px solid #007bff; border-radius: 50%; animation: spin 1s linear infinite; margin: 0 auto;"></div>
                    <p style="margin-top: 1rem; font-weight: bold;">Analyse en cours...</p>
                    <p style="color: #666;">L'IA examine votre radiographie</p>
                </div>
            `;

            try {
                console.log('Envoi à l\'API...');
                console.log('FormData contenu:', {
                    file: selectedFile.name,
                    patient_id: patientInput?.value || '',
                    notes: notesInput?.value || ''
                });
                
                const response = await fetch('http://127.0.0.1:5000/api/analyze', {
                    method: 'POST',
                    body: formData
                });

                console.log('Réponse reçue:', {
                    status: response.status,
                    statusText: response.statusText,
                    headers: response.headers,
                    ok: response.ok
                });

                if (!response.ok) {
                    const errorText = await response.text();
                    console.error('Erreur réponse:', errorText);
                    throw new Error(`Erreur HTTP: ${response.status} - ${errorText}`);
                }

                const responseText = await response.text();
                console.log('Réponse brute:', responseText);
                
                let data;
                try {
                    data = JSON.parse(responseText);
                    console.log('Données parsées:', data);
                } catch (parseError) {
                    console.error('Erreur parsing JSON:', parseError);
                    throw new Error('Réponse invalide du serveur');
                }

                if (data.status === 'success' && data.result) {
                    console.log('Appel displayResults avec:', data.result);
                    // Stocker les résultats dans sessionStorage
                    sessionStorage.setItem('analysisState', 'success');
                    sessionStorage.setItem('analysisResults', JSON.stringify(data.result));
                    displayResults(data.result);
                } else if (data.status === 'error') {
                    console.warn('Erreur retournée par l\'API:', data);
                    sessionStorage.setItem('analysisState', 'error');
                    sessionStorage.setItem('analysisError', data.message || 'Erreur lors de l\'analyse');
                    resultContainer.innerHTML = `
                        <div style="background: #f8d7da; color: #721c24; padding: 1rem; border-radius: 4px;">
                            <h4>Erreur d'analyse</h4>
                            <p>${data.message || 'Erreur lors de l\'analyse'}</p>
                            ${data.result ? `<p>Détails: ${data.result.recommendation || ''}</p>` : ''}
                        </div>
                    `;
                } else {
                    console.error('Format de réponse inattendu:', data);
                    resultContainer.innerHTML = `
                        <div style="background: #f8d7da; color: #721c24; padding: 1rem; border-radius: 4px;">
                            <h4>Erreur</h4>
                            <p>Format de réponse inattendu</p>
                        </div>
                    `;
                }

            } catch (error) {
                console.error('Erreur catch:', error);
                console.error('Stack trace:', error.stack);
                
                sessionStorage.setItem('analysisState', 'error');
                sessionStorage.setItem('analysisError', error.message);
                
                resultContainer.innerHTML = `
                    <div style="background: #f8d7da; color: #721c24; padding: 1rem; border-radius: 4px;">
                        <h4>Erreur</h4>
                        <p>${error.message}</p>
                        <p style="font-size: 0.9em; margin-top: 0.5rem;">Vérifiez la console pour plus de détails.</p>
                    </div>
                `;
            } finally {
                // Réactiver le bouton
                analyzeBtn.disabled = false;
                analyzeBtn.textContent = originalText;
                
                // Débloquer le refresh après un délai
                setTimeout(() => {
                    analysisInProgress = false;
                }, 2000);
            }
            
            console.log('=== FIN ANALYSE ===');
        }

        // Fonction d'affichage des résultats (globale pour être accessible après refresh)
        window.displayResults = function(res) {
            console.log('=== displayResults appelé ===');
            console.log('Données reçues:', res);
            
            const resultContainer = document.getElementById('resultContainer');
            if (!resultContainer) {
                console.error('resultContainer non trouvé!');
                return;
            }
            
            // Vérifier que res existe et contient les données nécessaires
            if (!res) {
                console.error('Résultat vide ou null');
                resultContainer.innerHTML = `
                    <div style="background: #f8d7da; color: #721c24; padding: 1rem; border-radius: 4px;">
                        <h4>Erreur</h4>
                        <p>Aucun résultat reçu du serveur</p>
                    </div>
                `;
                return;
            }
            
            // Valeurs par défaut si certaines propriétés manquent
            const prediction = res.prediction || 'Inconnu';
            const probability = res.probability || 0;
            const confidence = res.confidence || 0;
            const urgency = res.urgency || 'aucune';
            const recommendation = res.recommendation || 'Aucune recommandation disponible';
            const details = res.details || {};
            
            const isPneumonia = prediction === 'Pneumonie';
            const urgencyColors = {
                'haute': '#dc3545',
                'moyenne': '#ff8c00',
                'faible': '#ffc107',
                'aucune': '#28a745'
            };
            const urgencyColor = urgencyColors[urgency] || '#6c757d';

            console.log('Génération du HTML pour l\'affichage...');
            
            const resultatHTML = `
                <div style="margin-top: 2rem; padding: 1.5rem; border: 1px solid #ddd; border-radius: 8px; background: #f8f9fa;">
                    <h3 style="text-align: center; margin-bottom: 2rem; color: #333;">📊 Résultats de l'analyse</h3>
                    
                    <div style="background: white; padding: 1.5rem; border-radius: 8px; box-shadow: 0 2px 4px rgba(0,0,0,0.1);">
                        <div style="text-align: center; margin-bottom: 2rem;">
                            <h2 style="color: ${isPneumonia ? '#dc3545' : '#28a745'}; margin: 0;">
                                ${prediction}
                            </h2>
                            <div style="margin: 1rem auto; max-width: 400px;">
                                <div style="background: #e9ecef; border-radius: 4px; height: 30px; position: relative; overflow: hidden;">
                                    <div style="background: ${isPneumonia ? '#dc3545' : '#28a745'}; height: 100%; width: ${probability * 100}%; transition: width 0.5s;"></div>
                                    <span style="position: absolute; left: 50%; transform: translateX(-50%); line-height: 30px; font-weight: bold;">
                                        ${(probability * 100).toFixed(1)}%
                                    </span>
                                </div>
                            </div>
                        </div>
                        
                        <div style="display: grid; grid-template-columns: 1fr 1fr; gap: 2rem;">
                            <div>
                                <h4>📋 Informations</h4>
                                <p><strong>Confiance de l'analyse:</strong> ${(confidence * 100).toFixed(1)}%</p>
                                <p><strong>Niveau d'urgence:</strong> 
                                    <span style="background: ${urgencyColor}; color: white; padding: 2px 8px; border-radius: 4px;">
                                        ${urgency.toUpperCase()}
                                    </span>
                                </p>
                            </div>
                            
                            <div>
                                <h4>🔍 Détails techniques</h4>
                                <p>Seuil: ${((details.threshold_used || 0.91) * 100).toFixed(1)}%</p>
                                <p>Qualité: ${((details.image_quality_score || 1) * 100).toFixed(1)}%</p>
                                <p>Cohérence: ${((details.tta_consistency || 1) * 100).toFixed(1)}%</p>
                            </div>
                        </div>
                        
                        <div style="background: #e3f2fd; padding: 1rem; border-radius: 4px; margin-top: 1.5rem;">
                            <h4 style="margin-top: 0;">💡 Recommandation</h4>
                            <p style="margin: 0; white-space: pre-line;">${recommendation}</p>
                        </div>
                        
                        <div style="text-align: center; margin-top: 2rem;">
                            <button type="button" class="btn btn-primary" onclick="window.print()" style="margin-right: 1rem;">
                                🖨️ Imprimer
                            </button>
                            <button type="button" class="btn btn-secondary" onclick="window.location.href='nouvelle_analyse.html'">
                                🔄 Nouvelle analyse
                            </button>
                        </div>
                    </div>
                </div>
            `;
            
            console.log('Injection du HTML dans resultContainer...');
            resultContainer.innerHTML = resultatHTML;
            console.log('=== Affichage terminé ===');
            
            // Scroll vers les résultats
            resultContainer.scrollIntoView({ behavior: 'smooth', block: 'start' });
        }

    } else {
        console.error('Éléments manquants!');
    }
});

// CSS
if (!document.getElementById('medai-styles')) {
    const style = document.createElement('style');
    style.id = 'medai-styles';
    style.textContent = `
        @keyframes spin {
            0% { transform: rotate(0deg); }
            100% { transform: rotate(360deg); }
        }
        
        .dragover {
            background-color: #e3f2fd !important;
            border-color: #2196f3 !important;
        }
        
        .btn {
            padding: 0.5rem 1rem;
            border: none;
            border-radius: 4px;
            cursor: pointer;
            font-size: 1rem;
            transition: all 0.3s;
        }
        
        .btn-primary {
            background: #007bff;
            color: white;
        }
        
        .btn-primary:hover {
            background: #0056b3;
        }
        
        .btn-secondary {
            background: #6c757d;
            color: white;
        }
        
        .btn:disabled {
            opacity: 0.6;
            cursor: not-allowed;
        }
    `;
    document.head.appendChild(style);
}