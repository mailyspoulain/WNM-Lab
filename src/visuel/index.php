<?php
// Message d'alerte
$register_message = "";
if ($_SERVER["REQUEST_METHOD"] == "POST" && isset($_POST['register'])) {
    $username = $_POST['username'];
    $password = $_POST['password'];

    // Connexion à la BDD
    $conn = new mysqli("localhost", "root", "", "medai");
    if ($conn->connect_error) {
        die("Erreur de connexion : " . $conn->connect_error);
    }

    // Sécurisation
    $username = $conn->real_escape_string($username);
    $hashedPassword = password_hash($password, PASSWORD_DEFAULT);

    // Vérification d'existence
    $check = $conn->query("SELECT * FROM utilisateurs WHERE username = '$username'");
    if ($check->num_rows > 0) {
        $register_message = "⚠️ Nom d'utilisateur déjà pris.";
    } else {
        $insert = $conn->query("INSERT INTO utilisateurs (username, password) VALUES ('$username', '$hashedPassword')");
        if ($insert) {
            $register_message = "✅ Inscription réussie !";
        } else {
            $register_message = "❌ Erreur lors de l'inscription.";
        }
    }

    $conn->close();
}
?>


<!DOCTYPE html>
<html>

<head>
    <meta charset="UTF-8" />
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>MedAI</title>
    <link rel="stylesheet" href="my_css.css">
</head>

<body>
    <h1 class="p1">Connexion à MedAI</h1>

    <p>Vous avez besoin d'un diagnostic rapide et efficace ? </p>
    <p>L'IA de MedAI est faites pour vous ! </p>
    <p>Grâce à elle vous obtiendrez des informations sur tout vos Scanner/IRM </p>
    <p class="p1"> Entrer vos identifiants ou inscrivez vous gratuitement pour avoir accès à ce service </p>
    <div class="wrapper">
        <div class="container">
            <h2>Connexion</h2>
            <form id="loginForm"> 
                <input type="text" id="nom"name="nom" placeholder="Nom d'utilisateur" required>
                <input type="password" name="password" placeholder="Mot de passe" required>
                <button type="submit">Se connecter</button>
                <p id="login_error" class="error"></p>
            </form>

            <h2>Inscription</h2>
            <form id="registerForm">
                <input type="text" name="register_username" placeholder="Nom d'utilisateur" required>
                <input type="password" name="register_password" placeholder="Mot de passe" required>
                <button type="submit">S'inscrire</button>
                <p id="register_error" class="error"></p>
            </form>
        </div>
    </div>
    <p id="message"></p> <!-- Affiche le message de réponse -->
    <script src="script.js"></script>

</body>

</html>