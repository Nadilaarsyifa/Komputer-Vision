// ==========================================================
// KONFIGURASI
// ==========================================================
const API_URL = "http://localhost:8000/predict";
const CAPTURE_INTERVAL_MS = 300;

// ==========================================================
// STATE
// ==========================================================
let currentMode     = "huruf";
let isRunning       = false;
let intervalId      = null;
let stream          = null;
let activeKataGuide = null;

// ==========================================================
// DOM ELEMENTS
// ==========================================================
const video          = document.getElementById("video");
const canvas         = document.getElementById("canvas");
const ctx            = canvas.getContext("2d");
const btnStart       = document.getElementById("btn-start");
const btnStop        = document.getElementById("btn-stop");
const displayMode    = document.getElementById("display-mode");
const displayRes     = document.getElementById("display-result");
const displayStat    = document.getElementById("display-status");
const statusDot      = document.getElementById("status-dot");
const camModeBadge   = document.getElementById("cam-mode-badge");
const noCam          = document.getElementById("no-cam");
const kataInfo       = document.getElementById("kata-info");
const modeButtons    = document.querySelectorAll(".mode-btn");

// Panduan blocks
const guideHuruf     = document.getElementById("guide-huruf");
const guideAngka     = document.getElementById("guide-angka");
const guideKata      = document.getElementById("guide-kata");

// ==========================================================
// MODE SELECTOR
// ==========================================================
modeButtons.forEach(btn => {
  btn.addEventListener("click", () => {
    modeButtons.forEach(b => b.classList.remove("active"));
    btn.classList.add("active");
    currentMode = btn.dataset.mode;

    // Update tampilan mode
    const modeLabel = currentMode.toUpperCase();
    displayMode.textContent   = modeLabel;
    camModeBadge.textContent  = modeLabel;
    displayRes.textContent    = "–";

    // Tampilkan daftar kata hanya di mode kata
    kataInfo.style.display = currentMode === "kata" ? "block" : "none";

    // Ganti blok panduan
    updateGuideBlock(currentMode);
  });
});

// ==========================================================
// PANDUAN: tampilkan blok sesuai mode
// ==========================================================
function updateGuideBlock(mode) {
  guideHuruf.classList.toggle("hidden", mode !== "huruf");
  guideAngka.classList.toggle("hidden", mode !== "angka");
  guideKata.classList.toggle("hidden",  mode !== "kata");
  closeAllGuideImages();
}

function closeAllGuideImages() {
  document.getElementById("img-wrap-huruf").classList.add("hidden");
  document.getElementById("img-wrap-angka").classList.add("hidden");
  document.getElementById("img-wrap-kata").classList.add("hidden");

  const btnH = document.getElementById("btn-guide-huruf");
  const btnA = document.getElementById("btn-guide-angka");
  if (btnH) { btnH.classList.remove("active"); btnH.textContent = "Lihat Panduan"; }
  if (btnA) { btnA.classList.remove("active"); btnA.textContent = "Lihat Panduan"; }

  document.querySelectorAll(".btn-kata-guide").forEach(b => b.classList.remove("active"));
  activeKataGuide = null;
}

// ==========================================================
// TOGGLE PANDUAN HURUF / ANGKA
// ==========================================================
function toggleGuide(type) {
  const wrap    = document.getElementById("img-wrap-" + type);
  const btn     = document.getElementById("btn-guide-" + type);
  const isHidden = wrap.classList.contains("hidden");

  wrap.classList.toggle("hidden", !isHidden);
  btn.classList.toggle("active", isHidden);
  btn.textContent = isHidden
    ? "Tutup Panduan"
    : "Lihat Panduan";
}

// ==========================================================
// PANDUAN KATA — per kata
// ==========================================================
function showKataGuide(kata) {
  const wrap    = document.getElementById("img-wrap-kata");
  const img     = document.getElementById("kata-guide-img");
  const caption = document.getElementById("kata-guide-caption");

  // Klik tombol yang sama = tutup
  if (activeKataGuide === kata) {
    wrap.classList.add("hidden");
    document.querySelectorAll(".btn-kata-guide").forEach(b => b.classList.remove("active"));
    activeKataGuide = null;
    return;
  }

  img.src        = "images/panduan_" + kata + ".jpg";
  img.alt        = "Panduan gerakan: " + kata;
  caption.textContent = "Gerakan untuk kata: " + kata.charAt(0).toUpperCase() + kata.slice(1);

  wrap.classList.remove("hidden");
  activeKataGuide = kata;

  // Tandai tombol aktif
  document.querySelectorAll(".btn-kata-guide").forEach(b => {
    const label = b.textContent.replace(/^\S+\s/, "").toLowerCase(); // hapus emoji
    b.classList.toggle("active", label === kata);
  });
}

// ==========================================================
// START KAMERA
// ==========================================================
btnStart.addEventListener("click", async () => {
  try {
    setStatus("Meminta akses kamera...", "");

    stream = await navigator.mediaDevices.getUserMedia({
      video: { width: 640, height: 480, facingMode: "user" },
      audio: false
    });

    video.srcObject = stream;
    await video.play();

    noCam.classList.add("hidden");
    isRunning         = true;
    btnStart.disabled = true;
    btnStop.disabled  = false;

    setStatus("... (" + currentMode + ")", "active");
    intervalId = setInterval(sendFrame, CAPTURE_INTERVAL_MS);

  } catch (err) {
    setStatus("Gagal akses kamera: " + err.message, "error");
  }
});

// ==========================================================
// STOP KAMERA
// ==========================================================
btnStop.addEventListener("click", stopCamera);

function stopCamera() {
  isRunning = false;
  clearInterval(intervalId);

  if (stream) {
    stream.getTracks().forEach(t => t.stop());
    stream = null;
  }

  video.srcObject   = null;
  noCam.classList.remove("hidden");
  btnStart.disabled = false;
  btnStop.disabled  = true;

  displayRes.textContent = "–";
  setStatus("Kamera dihentikan", "");
}

// ==========================================================
// KIRIM FRAME KE BACKEND
// ==========================================================
async function sendFrame() {
  if (!isRunning) return;

  canvas.width  = video.videoWidth;
  canvas.height = video.videoHeight;
  ctx.drawImage(video, 0, 0);

  const base64 = canvas.toDataURL("image/jpeg", 0.7).split(",")[1];

  try {
    const response = await fetch(API_URL, {
      method:  "POST",
      headers: { "Content-Type": "application/json" },
      body:    JSON.stringify({ image: base64, mode: currentMode })
    });

    if (!response.ok) {
      const err = await response.json();
      setStatus("Error: " + (err.detail || "Unknown"), "error");
      return;
    }

    const data = await response.json();
    displayRes.textContent = data.result || "–";
    setStatus("Mendeteksi...", "active");

  } catch (err) {
    setStatus("Tidak bisa koneksi ke backend", "error");
  }
}

// ==========================================================
// HELPER: SET STATUS
// ==========================================================
function setStatus(text, type) {
  displayStat.textContent = text;
  displayStat.className   = "result-bar-status";
  statusDot.className     = "status-dot";

  if (type) {
    displayStat.classList.add(type);
    statusDot.classList.add(type);
  }
}