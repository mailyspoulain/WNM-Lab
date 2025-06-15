<?php
session_start();
if (!isset($_SESSION['username'])) {
    header('Location: connexion.php'); // Redirige si non connecté
    exit;
}
$username = $_SESSION['username'];
?>



<html lang="fr">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <link rel="stylesheet" href="dashboard.css">
    <title>MedAI - Dashboard</title>
</head>
<body>
    <!-- En-tête de navigation -->
    <header class="header">
        <div class="logo">Med<span>AI</span></div>
        <nav class="nav">
            <a href="#" class="nav-link active">Dashboard</a>
            <a href="#" class="nav-link">Documentation</a>
            <a href="a_propos.html" class="nav-link">À propos</a>
            <a href="#" class="nav-link">
                <?php echo htmlspecialchars($username); ?>
                <img src="https://via.placeholder.com/24" alt="Avatar" class="avatar" style="margin-left: 0.5rem;" />
            </a>
        </nav>
    </header>
    
    <!-- Contenu principal -->
    <div class="main-container">
        <div class="dashboard-layout">
            <!-- Barre latérale -->
            <div class="sidebar">
                <h3>Menu principal</h3>
                <ul class="sidebar-menu">
                    <li><a href="dashboard.php" class="active">Tableau de bord</a></li>
                    <li><a href="nouvelle_analyse.html">Nouvelle analyse</a></li>
                    <li><a href="historique.html">Historique</a></li>
                    <li><a href="parametre.html">Paramètres</a></li>
                </ul>
            </div>
            
            <!-- Contenu principal -->
            <div class="content">
                <h2>Tableau de bord</h2>
                <p>Bienvenue, ...</p>
                
                <!-- Cartes de statistiques -->
                <div class="grid-3 mt-2">
                    <div class="card">
                        <div class="card-body">
                            <h4>Analyses effectuées</h4>
                            <div style="display: flex; justify-content: space-between; align-items: center;">
                                <span style="font-size: 2rem; font-weight: bold;">...%</span>
                                <span style="color: var(--success);">...</span>
                            </div>
                        </div>
                    </div>
                    
                    <div class="card">
                        <div class="card-body">
                            <h4>Précision moyenne</h4>
                            <div style="display: flex; justify-content: space-between; align-items: center;">
                                <span style="font-size: 2rem; font-weight: bold;">...%</span>
                                <span style="color: var(--success);">...</span>
                            </div>
                        </div>
                    </div>
                    
                    <div class="card">
                        <div class="card-body">
                            <h4>Temps moyen d'analyse</h4>
                            <div style="display: flex; justify-content: space-between; align-items: center;">
                                <span style="font-size: 2rem; font-weight: bold;">...s</span>
                                <span style="color: var(--success);">...</span>
                            </div>
                        </div>
                    </div>
                </div>
                
                <!-- Tableau des analyses récentes -->
                <div class="mt-4">
                    <h3>Analyses récentes</h3>
                    <table class="table">
                        <thead>
                            <tr>
                                <th>Date</th>
                                <th>Type d'image</th>
                                <th>Diagnostic</th>
                                <th>Confiance</th>
                                <th>Actions</th>
                            </tr>
                        </thead>
                        <tbody>
                            <tr>
                                <td>14/04/2025</td>
                                <td>Radiographie pulmonaire</td>
                                <td>Pneumonie</td>
                                <td>
                                    <div class="progress-bar">
                                        <div class="progress-fill" style="width: 92%;"></div>
                                    </div>
                                    <span>92%</span>
                                </td>
                                <td>
                                    <button class="btn btn-outline btn-sm">Voir</button>
                                </td>
                            </tr>
                            
                        </tbody>
                    </table>
                </div>
                
                <!-- Graphique de distribution -->
                <div class="mt-4">
                    <h3>Distribution des diagnostics</h3>
                    <div class="chart-container">
                        <!-- Ici, vous intégreriez un graphique avec une bibliothèque comme Chart.js -->
                        <p style="color: var(--info);">Graphique de distribution des diagnostics</p>
                    </div>
                </div>
                
                <!-- Boutons d'action -->
                <div class="actions">
                    <button class="btn btn-primary">Nouvelle analyse</button>
                    <button class="btn btn-outline">Voir tout l'historique</button>
                </div>
            </div>
        </div>
    </div>
</body>
</html>