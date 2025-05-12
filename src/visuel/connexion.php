<?php
// Connexion à la base de données MySQL
$host = 'localhost';
$dbname = 'medai_usr';
$user = 'root'; // ou ton utilisateur MySQL
$pass = ''; // ou ton mot de passe MySQL

// Variables pour les erreurs et le succès
$errorMessage = '';
$successMessage = '';

// Traitement du formulaire de connexion si soumis
if ($_SERVER['REQUEST_METHOD'] == 'POST') {
    try {
        // Création de la connexion avec PDO
        $pdo = new PDO("mysql:host=$host;dbname=$dbname;charset=utf8", $user, $pass);
        $pdo->setAttribute(PDO::ATTR_ERRMODE, PDO::ERRMODE_EXCEPTION);

        // Récupération des données du formulaire
        $username = $_POST['login_username'];
        $password = $_POST['login_password']; // Mot de passe sans hachage

        // Vérification si l'utilisateur existe
        $stmt = $pdo->prepare("SELECT * FROM usr WHERE username = :username");
        $stmt->execute([':username' => $username]);
        $user = $stmt->fetch(PDO::FETCH_ASSOC);

        if ($user) {
            // Comparer le mot de passe avec celui stocké (ici sans hachage pour l'exemple)
            if ($user['passwd'] == $password) {
                $successMessage = "Connexion réussie. Bienvenue, $username!";
                // Tu pourrais rediriger l'utilisateur vers une page protégée après la connexion
                // header('Location: dashboard.php'); exit;
            } else {
                $errorMessage = "Mot de passe incorrect.";
            }
        } else {
            $errorMessage = "Nom d'utilisateur incorrect.";
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
    <h1>Portail de connexion</h1>
    </section>

    <p class="p1">Vous faites déja partie de ceux qui se soucient réelement de leur santé ? </p>
    <p>Connectez vous pour pouvoir profiter pleinement de vos droits</p>

    <div class="wrapper">
        <div class="container">
            <h2>Connexion</h2>
            <!-- Affichage du message de succès ou d'erreur -->
            <?php if ($successMessage): ?>
                <p style="color: green;"><?php echo $successMessage; ?></p>
            <?php endif; ?>
            <?php if ($errorMessage): ?>
                <p style="color: red;"><?php echo $errorMessage; ?></p>
            <?php endif; ?>
            <form action="connexion.php" method="POST">
                <input type="text" name="login_username" placeholder="Nom d'utilisateur" required>
                <input type="password" name="login_password" placeholder="Mot de passe" required>
                <button type="submit">Se connecter</button>
                <p id="login_error" class="error"></p>
            </form>
        </div>
    </div>
</body>
</html>
