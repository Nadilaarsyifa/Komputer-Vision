// ==========================================================
// KONFIGURASI
// ==========================================================
const API_URL = "http://localhost:8000/predict"; // Ganti jika backend di server lain
const CAPTURE_INTERVAL_MS = 300;               // Kirim frame setiap 300ms (~3fps ke server)

// ==========================================================
// STATE
// ==========================================================
let currentMode = "huruf";
let isRunning   = false;
let intervalId  = null;
let stream      = null;

// ==========================================================
// DOM ELEMENTS
// ==========================================================
const video       = document.getElementById("video");
const canvas      = document.getElementById("canvas");
const ctx         = canvas.getContext("2d");
const btnStart    = document.getElementById("btn-start");
const btnStop     = document.getElementById("btn-stop");
const displayMode = document.getElementById("display-mode");
const displayRes  = document.getElementById("display-result");
const displayStat = document.getElementById("display-status");
const noCam       = document.getElementById("no-cam");
const kataInfo    = document.getElementById("kata-info");
const modeButtons = document.querySelectorAll(".mode-btn");

// ==========================================================
// MODE SELECTOR
// ==========================================================
modeButtons.forEach(btn => {
  btn.addEventListener("click", () => {
    modeButtons.forEach(b => b.classList.remove("active"));
    btn.classList.add("active");
    currentMode = btn.dataset.mode;
    displayMode.textContent = currentMode.toUpperCase();
    displayRes.textContent = "-";

    // Tampilkan info kata jika mode kata
    kataInfo.style.display = currentMode === "kata" ? "block" : "none";
  });
});

// ==========================================================
// START KAMERA
// ==========================================================
btnStart.addEventListener("click", async () => {
  try {
    setStatus("Meminta akses kamera...", "");

    // Minta izin kamera
    stream = await navigator.mediaDevices.getUserMedia({
      video: { width: 640, height: 480, facingMode: "user" },
      audio: false
    });

    video.srcObject = stream;
    await video.play();

    // Sembunyikan overlay "kamera tidak aktif"
    noCam.classList.add("hidden");

    isRunning = true;
    btnStart.disabled = true;
    btnStop.disabled  = false;

    setStatus("Kamera aktif — mendeteksi...", "active");

    // Mulai kirim frame ke backend secara berkala
    intervalId = setInterval(sendFrame, CAPTURE_INTERVAL_MS);

  } catch (err) {
    setStatus("Gagal akses kamera: " + err.message, "error");
  }
});

// ==========================================================
// STOP KAMERA
// ==========================================================
btnStop.addEventListener("click", () => {
  stopCamera();
});

function stopCamera() {
  isRunning = false;
  clearInterval(intervalId);

  if (stream) {
    stream.getTracks().forEach(t => t.stop());
    stream = null;
  }

  video.srcObject = null;
  noCam.classList.remove("hidden");

  btnStart.disabled = false;
  btnStop.disabled  = true;

  displayRes.textContent = "-";
  setStatus("Kamera dihentikan", "");
}

// ==========================================================
// AMBIL FRAME & KIRIM KE BACKEND
// ==========================================================
async function sendFrame() {
  if (!isRunning) return;

  // Set ukuran canvas sesuai video
  canvas.width  = video.videoWidth;
  canvas.height = video.videoHeight;

  // Gambar frame video ke canvas
  // Note: video sudah di-mirror via CSS; gambar di canvas tidak di-mirror
  // agar mediapipe di backend menerima gambar normal (tidak mirror)
  ctx.drawImage(video, 0, 0);

  // Konversi canvas ke base64 JPEG (kualitas 0.7 untuk hemat bandwidth)
  const base64 = canvas.toDataURL("image/jpeg", 0.7).split(",")[1];

  try {
    const response = await fetch(API_URL, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        image: base64,
        mode: currentMode
      })
    });

    if (!response.ok) {
      const err = await response.json();
      setStatus("Error: " + (err.detail || "Unknown"), "error");
      return;
    }

    const data = await response.json();

    // Tampilkan hasil
    displayRes.textContent = data.result || "-";
    setStatus("Mendeteksi... (" + currentMode + ")", "active");

  } catch (err) {
    // Biasanya network error / backend tidak jalan
    setStatus("Tidak bisa koneksi ke backend", "error");
  }
}

// ==========================================================
// HELPER: SET STATUS TEXT
// ==========================================================
function setStatus(text, type) {
  displayStat.textContent = text;
  displayStat.className   = "result-value status-display";
  if (type) displayStat.classList.add(type);
}