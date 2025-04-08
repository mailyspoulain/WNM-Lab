<?php
$host = "localhost";
$dbname = "med_ai";
$user = "root"; 
$pass = "1A2B3c4d*";
$conn = "";

$conn = new mysqli($host,$user,$pass,$dbname);

if($conn){
    echo " Vous êtes bien connecté";
}
else {
    echo "Mété co'ou deyo";
}


/*
try {
    $pdo = new PDO("mysql:host=$host;dbname=$dbname;charset=utf8", $user, $pass);
    $pdo->setAttribute(PDO::ATTR_ERRMODE, PDO::ERRMODE_EXCEPTION);
} catch (PDOException $e) {
    die("Erreur de connexion : " . $e->getMessage());
}

if ($_SERVER["REQUEST_METHOD"] == "POST") {
    if (!empty($_POST["username"]) && !empty($_POST["password"])) {
        $username = htmlspecialchars($_POST["username"]);
        $password = password_hash($_POST["password"], PASSWORD_BCRYPT); 

        try {
            $stmt = $pdo->prepare("INSERT INTO users (username, password) VALUES (?, ?)");
            $stmt->execute([$username, $password]);
            echo "Inscription réussie !";
        } catch (PDOException $e) {
            echo "Erreur : Nom d'utilisateur déjà pris.";
        }
    } else {
        echo "Veuillez remplir tous les champs.";
    }
}
?>
*/