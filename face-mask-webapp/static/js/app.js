const video = document.getElementById("video");
const overlay = document.getElementById("overlay");
const startBtn = document.getElementById("startBtn");
const stopBtn = document.getElementById("stopBtn");
const statustext = document.getElementById("statustext");
const capturedImage = document.getElementById("capturedImage");
const captureContainer = document.getElementById("captureContainer");
const downloadLink = document.getElementById("downloadLink");

let stream = null;
let sending = false;
let sendInterval = null;

async function startCamera() {
  try {
    stream = await navigator.mediaDevices.getUserMedia({
      video: { width: 640, height: 480 },
      audio: false
    });
    video.srcObject = stream;
    await video.play();

    // Sync overlay size with video
    await new Promise(r => setTimeout(r, 100));
    overlay.width = video.videoWidth;
    overlay.height = video.videoHeight;

    statustext.textContent = "camera started";

    sending = true;
    sendInterval = setInterval(sendFrameForPrediction, 250); // 4 FPS
    startBtn.disabled = true;
    stopBtn.disabled = false;
    captureContainer.style.display = "none"; // hide old capture
  } catch (err) {
    console.error(err);
    statustext.textContent = "camera error: " + err.message;
  }
}

function stopCamera() {
  // Capture before stopping video stream
  captureCurrentFrame();

  if (stream) {
    stream.getTracks().forEach(t => t.stop());
    stream = null;
  }
  if (sendInterval) clearInterval(sendInterval);
  sending = false;
  startBtn.disabled = false;
  stopBtn.disabled = true;
  statustext.textContent = "stopped";
}

function captureCurrentFrame() {
  if (!video || video.readyState < 2) {
    statustext.textContent = "capture failed (video not ready)";
    return;
  }

  // Create merged canvas
  const mergedCanvas = document.createElement("canvas");
  mergedCanvas.width = video.videoWidth;
  mergedCanvas.height = video.videoHeight;
  const ctx = mergedCanvas.getContext("2d");

  // Draw video frame
  ctx.drawImage(video, 0, 0, mergedCanvas.width, mergedCanvas.height);

  // Draw detection overlay
  ctx.drawImage(overlay, 0, 0, mergedCanvas.width, mergedCanvas.height);

  // Get final image
  const dataUrl = mergedCanvas.toDataURL("image/png");
  capturedImage.src = dataUrl;
  downloadLink.href = dataUrl;
  captureContainer.style.display = "block"; // show preview
}

function clearCanvas() {
  const ctx = overlay.getContext("2d");
  ctx.clearRect(0, 0, overlay.width, overlay.height);
}

async function sendFrameForPrediction() {
  if (!video || video.readyState < 2) return;

  const tmp = document.createElement("canvas");
  tmp.width = video.videoWidth;
  tmp.height = video.videoHeight;
  const tctx = tmp.getContext("2d");
  tctx.drawImage(video, 0, 0, tmp.width, tmp.height);

  const dataUrl = tmp.toDataURL("image/jpeg", 0.8);

  try {
    statustext.textContent = "sending frame...";
    const res = await fetch("/predict_frame", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ image: dataUrl })
    });
    if (!res.ok) {
      const txt = await res.text();
      console.error("Server error:", txt);
      statustext.textContent = "server error";
      return;
    }
    const json = await res.json();
    drawDetections(json.detections || []);
    statustext.textContent = `detections: ${(json.detections || []).length}`;
  } catch (err) {
    console.error(err);
    statustext.textContent = "network error";
  }
}

// Helper to draw rounded rectangle
function roundRect(ctx, x, y, w, h, r) {
  ctx.beginPath();
  ctx.moveTo(x + r, y);
  ctx.lineTo(x + w - r, y);
  ctx.quadraticCurveTo(x + w, y, x + w, y + r);
  ctx.lineTo(x + w, y + h - r);
  ctx.quadraticCurveTo(x + w, y + h, x + w - r, y + h);
  ctx.lineTo(x + r, y + h);
  ctx.quadraticCurveTo(x, y + h, x, y + h - r);
  ctx.lineTo(x, y + r);
  ctx.quadraticCurveTo(x, y, x + r, y);
  ctx.closePath();
}

function drawDetections(dets) {
  const ctx = overlay.getContext("2d");
  ctx.clearRect(0, 0, overlay.width, overlay.height);
  ctx.lineWidth = 4;
  ctx.font = "bold 18px Arial";
  ctx.shadowColor = "rgba(0,0,0,0.6)";
  ctx.shadowBlur = 6;

  for (const d of dets) {
    const x = d.x, y = d.y, w = d.w, h = d.h;
    const label = d.label || "??";
    const conf = d.conf || 0;

    let gradient, tagColor;
    if (String(label).toLowerCase().includes("with")) {
      gradient = ctx.createLinearGradient(x, y, x + w, y + h);
      gradient.addColorStop(0, "#185817ff"); // neon green
      gradient.addColorStop(1, "#145512ff");
      tagColor = "#1e6c1eff";
    } else {
      gradient = ctx.createLinearGradient(x, y, x + w, y + h);
      gradient.addColorStop(0, "#ff4d4d"); // neon red
      gradient.addColorStop(1, "#cc0000");
      tagColor = "#cc0000";
    }

    ctx.strokeStyle = gradient;
    ctx.lineWidth = 4;
    roundRect(ctx, x, y, w, h, 12);
    ctx.stroke();

    const text = `${label} ${(conf * 100).toFixed(1)}%`;
    const textPadding = 6;
    const textWidth = ctx.measureText(text).width + textPadding * 2;
    const textHeight = 22;

    ctx.fillStyle = tagColor;
    roundRect(ctx, x, Math.max(0, y - textHeight - 4), textWidth, textHeight, 8);
    ctx.fill();

    ctx.fillStyle = "white";
    ctx.fillText(text, x + textPadding, Math.max(16, y - 8));
  }

  ctx.shadowBlur = 0;
}

startBtn.addEventListener("click", startCamera);
stopBtn.addEventListener("click", stopCamera);
