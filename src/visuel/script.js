document.getElementById("registerForm").addEventListener("submit", function(event) {
    event.preventDefault();

    let formData = new FormData();
    formData.append("username", document.getElementById("username").value);
    formData.append("password", document.getElementById("password").value);

    fetch("inscription.php", {
        method: "POST",
        body: formData
    })
    .then(response => response.text())
    .then(data => {
        document.getElementById("message").innerText = data;
    })
    .catch(error => console.error("Erreur :", error));
});
