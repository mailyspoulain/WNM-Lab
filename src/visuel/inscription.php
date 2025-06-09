<?php
// Connexion à la base de données MySQL
$host = 'localhost';
$dbname = 'medai_usr';
$user = 'root'; // ou ton utilisateur MySQL
$pass = ''; // ou ton mot de passe MySQL

// Variables pour les erreurs et le succès
$errorMessage = '';
$successMessage = '';

// Traitement du formulaire si soumis
if ($_SERVER['REQUEST_METHOD'] == 'POST') {
    try {
        // Création de la connexion avec PDO
        $pdo = new PDO("mysql:host=$host;dbname=$dbname;charset=utf8", $user, $pass);
        $pdo->setAttribute(PDO::ATTR_ERRMODE, PDO::ERRMODE_EXCEPTION);

        // Récupération des données du formulaire
        $username = $_POST['register_username'];
        $password = $_POST['register_password']; // Mot de passe sans hachage

        // Vérifier si l'utilisateur existe déjà
        $stmt = $pdo->prepare("SELECT COUNT(*) FROM usr WHERE username = :username");
        $stmt->execute([':username' => $username]);
        $userExists = $stmt->fetchColumn();

        if ($userExists > 0) {
            $errorMessage = "Nom d'utilisateur déjà pris. Choisissez-en un autre.";
        } else {
            // Insertion des données dans la table
            $stmt = $pdo->prepare("INSERT INTO usr (username, passwd) VALUES (:username, :password)");
            $stmt->execute([
                ':username' => $username,
                ':password' => $password
            ]);

            $successMessage = "Inscription réussie ! Bienvenue dans MedVision AI.";
        }
    } catch (PDOException $e) {
        $errorMessage = "Erreur : " . $e->getMessage();
    }
}
?>


<html lang="fr">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>MedVision AI - Analyse d'Images Médicales</title>
    <link rel="stylesheet" href="accueil.css">
</head>
<body>
    <section class="hero">
    <h1>Portail d'inscription</h1>
    </section>

    <p class="p1">"Rejoignez notre communauté d'utilisateurs qui fait confiance à notre 
        plateforme d'analyse médicale par intelligence artificielle.
        Créez votre compte sécurisé en quelques clics pour commencer à analyser vos IRM et scanners. </p>

    <div class="wrapper">
        <div class="container">
            <h2>Inscription</h2>
            <!-- Affichage du message de succès ou d'erreur -->
            <?php if ($successMessage): ?>
                <p style="color: green;"><?php echo $successMessage; ?></p>
            <?php endif; ?>
            <?php if ($errorMessage): ?>
                <p style="color: red;"><?php echo $errorMessage; ?></p>
            <?php endif; ?>
            <form action="inscription.php" method="POST">
                <input type="text" name="register_username" placeholder="Nom d'utilisateur" required>
                <input type="password" name="register_password" placeholder="Mot de passe" required>
                <button type="submit">S'inscrire</button>
                <p id="register_error" class="error"></p>
            </form>
        </div>
    </div>
</body>
</html>
