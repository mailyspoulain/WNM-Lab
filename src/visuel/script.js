document.getElementById("registerForm").addEventListener("submit", function(event) {
    event.preventDefault();

    let formData = new FormData();
    formData.append("username", document.getElementById("username").value);
    formData.append("password", document.getElementById("password").value);

    fetch("register.php", {
        method: "POST",
        body: formData
    })
    .then(response => response.text())
    .then(data => {
        console.log("Réponse du serveur :", data); // 🔴 DEBUG ICI
        document.getElementById("message").innerText = data;
    })
    .catch(error => console.error("Erreur :", error));
});

var sidenav = document.getElementById("mySidenav");
var openBtn = document.getElementById("openBtn");
var closeBtn = document.getElementById("closeBtn");

openBtn.onclick = openNav;
closeBtn.onclick = closeNav;

/* Set the width of the side navigation to 250px */
function openNav(e) {
  e.preventDefault();
  sidenav.classList.add("active");
}

/* Set the width of the side navigation to 0 */
function closeNav(e) {
  e.preventDefault();
  sidenav.classList.remove("active");
}