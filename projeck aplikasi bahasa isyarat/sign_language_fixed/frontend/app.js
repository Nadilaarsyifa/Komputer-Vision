// ==========================================================
// KONFIGURASI
// ==========================================================
const API_URL = "http://127.0.0.1:8000/predict";
const CAPTURE_INTERVAL_MS = 500; // lebih ringan daripada 300ms

// ==========================================================
// STATE
// ==========================================================
let currentMode = "huruf";
let isRunning = false;
let intervalId = null;
let stream = null;
let activeKataGuide = null;
let isPredicting = false;

// ==========================================================
// DOM ELEMENTS
// ==========================================================
const video = document.getElementById("video");
const canvas = document.getElementById("canvas");
const ctx = canvas.getContext("2d");
const btnStart = document.getElementById("btn-start");
const btnStop = document.getElementById("btn-stop");
const displayMode = document.getElementById("display-mode");
const displayRes = document.getElementById("display-result");
const displayConf = document.getElementById("display-confidence");
const displayStat = document.getElementById("display-status");
const statusDot = document.getElementById("status-dot");
const camModeBadge = document.getElementById("cam-mode-badge");
const noCam = document.getElementById("no-cam");
const kataInfo = document.getElementById("kata-info");
const modeButtons = document.querySelectorAll(".mode-btn");

const guideHuruf = document.getElementById("guide-huruf");
const guideAngka = document.getElementById("guide-angka");
const guideKata = document.getElementById("guide-kata");

// ==========================================================
// MODE SELECTOR
// ==========================================================
modeButtons.forEach((btn) => {
  btn.addEventListener("click", () => {
    modeButtons.forEach((b) => b.classList.remove("active"));
    btn.classList.add("active");
    currentMode = btn.dataset.mode;

    const modeLabel = currentMode.toUpperCase();
    displayMode.textContent = modeLabel;
    camModeBadge.textContent = modeLabel;
    displayRes.textContent = "–";
    if (displayConf) displayConf.textContent = "–";

    if (kataInfo) {
      kataInfo.style.display = currentMode === "kata" ? "block" : "none";
    }

    updateGuideBlock(currentMode);
  });
});

// ==========================================================
// PANDUAN
// ==========================================================
function updateGuideBlock(mode) {
  if (guideHuruf) guideHuruf.classList.toggle("hidden", mode !== "huruf");
  if (guideAngka) guideAngka.classList.toggle("hidden", mode !== "angka");
  if (guideKata) guideKata.classList.toggle("hidden", mode !== "kata");
  closeAllGuideImages();
}

function closeAllGuideImages() {
  ["huruf", "angka", "kata"].forEach((type) => {
    const wrap = document.getElementById("img-wrap-" + type);
    if (wrap) wrap.classList.add("hidden");
  });

  const btnH = document.getElementById("btn-guide-huruf");
  const btnA = document.getElementById("btn-guide-angka");
  if (btnH) { btnH.classList.remove("active"); btnH.textContent = "Lihat Panduan"; }
  if (btnA) { btnA.classList.remove("active"); btnA.textContent = "Lihat Panduan"; }

  document.querySelectorAll(".btn-kata-guide").forEach((b) => b.classList.remove("active"));
  activeKataGuide = null;
}

function toggleGuide(type) {
  const wrap = document.getElementById("img-wrap-" + type);
  const btn = document.getElementById("btn-guide-" + type);
  if (!wrap || !btn) return;

  const isHidden = wrap.classList.contains("hidden");
  wrap.classList.toggle("hidden", !isHidden);
  btn.classList.toggle("active", isHidden);
  btn.textContent = isHidden ? "Tutup Panduan" : "Lihat Panduan";
}

function showKataGuide(kata) {
  const wrap = document.getElementById("img-wrap-kata");
  const img = document.getElementById("kata-guide-img");
  const caption = document.getElementById("kata-guide-caption");
  if (!wrap || !img || !caption) return;

  if (activeKataGuide === kata) {
    wrap.classList.add("hidden");
    document.querySelectorAll(".btn-kata-guide").forEach((b) => b.classList.remove("active"));
    activeKataGuide = null;
    return;
  }

  img.src = "images/panduan_" + kata + ".jpg";
  img.alt = "Panduan gerakan: " + kata;
  caption.textContent = "Gerakan untuk kata: " + kata.charAt(0).toUpperCase() + kata.slice(1);

  wrap.classList.remove("hidden");
  activeKataGuide = kata;

  document.querySelectorAll(".btn-kata-guide").forEach((b) => {
    const label = b.textContent.trim().toLowerCase();
    b.classList.toggle("active", label === kata);
  });
}

window.toggleGuide = toggleGuide;
window.showKataGuide = showKataGuide;

// ==========================================================
// START KAMERA
// ==========================================================
btnStart.addEventListener("click", async () => {
  try {
    setStatus("Meminta akses kamera...", "");

    stream = await navigator.mediaDevices.getUserMedia({
      video: { width: 640, height: 480, facingMode: "user" },
      audio: false,
    });

    video.srcObject = stream;
    await video.play();

    noCam.classList.add("hidden");
    isRunning = true;
    btnStart.disabled = true;
    btnStop.disabled = false;

    setStatus("Mendeteksi...", "active");
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
    stream.getTracks().forEach((t) => t.stop());
    stream = null;
  }

  video.srcObject = null;
  noCam.classList.remove("hidden");
  btnStart.disabled = false;
  btnStop.disabled = true;

  displayRes.textContent = "–";
  if (displayConf) displayConf.textContent = "–";
  setStatus("Kamera dihentikan", "");
}

// ==========================================================
// KIRIM FRAME KE BACKEND
// ==========================================================
async function sendFrame() {
  if (!isRunning || isPredicting || video.videoWidth === 0) return;

  isPredicting = true;
  canvas.width = video.videoWidth;
  canvas.height = video.videoHeight;
  ctx.drawImage(video, 0, 0);

  const base64 = canvas.toDataURL("image/jpeg", 0.7).split(",")[1];

  try {
    const response = await fetch(API_URL, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ image: base64, mode: currentMode }),
    });

    if (!response.ok) {
      const err = await response.json();
      setStatus("Error: " + (err.detail || "Unknown"), "error");
      return;
    }

    const data = await response.json();
    displayRes.textContent = data.result || "–";
    if (displayConf) displayConf.textContent = data.confidence !== undefined ? data.confidence + "%" : "–";
    setStatus("Mendeteksi...", "active");
  } catch (err) {
    setStatus("Tidak bisa koneksi ke backend", "error");
  } finally {
    isPredicting = false;
  }
}

// ==========================================================
// HELPER
// ==========================================================
function setStatus(text, type) {
  displayStat.textContent = text;
  displayStat.className = "result-bar-status";
  statusDot.className = "status-dot";

  if (type) {
    displayStat.classList.add(type);
    statusDot.classList.add(type);
  }
}

updateGuideBlock(currentMode);
