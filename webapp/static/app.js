// Action Detection frontend: streams webcam or video-file frames to the backend,
// which runs MediaPipe Holistic + the LSTM, then draws landmarks and predictions.
(() => {
  const $ = (id) => document.getElementById(id);

  const display = $("display");
  const ctx = display.getContext("2d");
  const video = $("video");
  const capture = document.createElement("canvas");
  const cctx = capture.getContext("2d");

  const els = {
    tabs: document.querySelectorAll(".tab"),
    cameraPlaceholder: $("cameraPlaceholder"),
    dropzone: $("dropzone"),
    fileInput: $("fileInput"),
    statusPill: $("statusPill"),
    bufferPill: $("bufferPill"),
    caption: $("caption"),
    cameraToolbar: $("cameraToolbar"),
    videoToolbar: $("videoToolbar"),
    startBtn: $("startBtn"),
    startBtnInline: $("startBtnInline"),
    cameraSelect: $("cameraSelect"),
    snapshotBtn: $("snapshotBtn"),
    snapshotBtn2: $("snapshotBtn2"),
    chooseBtn: $("chooseBtn"),
    playBtn: $("playBtn"),
    videoName: $("videoName"),
    stateBadge: $("stateBadge"),
    topName: $("topName"),
    topConf: $("topConf"),
    bars: $("bars"),
    bufferText: $("bufferText"),
    bufferFill: $("bufferFill"),
    chips: $("chips"),
    clearBtn: $("clearBtn"),
    statFps: $("statFps"),
    statRtt: $("statRtt"),
    statHolistic: $("statHolistic"),
    statModel: $("statModel"),
    threshold: $("threshold"),
    thresholdOut: $("thresholdOut"),
    stable: $("stable"),
    showLandmarks: $("showLandmarks"),
    showFace: $("showFace"),
    mirror: $("mirror"),
    mirrorSwitch: $("mirrorSwitch"),
    modelChip: $("modelChip"),
  };

  const SEND_WIDTH = 640; // the model was trained on 640x480 webcam frames
  const STABLE_FRAMES = 10; // from the refined notebook's predictions[-10:] check
  const MAX_SENTENCE = 8;

  const state = {
    mode: "camera",
    info: null,
    stream: null,
    videoUrl: null,
    ws: null,
    running: false,
    inflight: false,
    retryTimer: null,
    sentAt: 0,
    responseTimes: [],
    latest: null, // last server response
    predHistory: [],
    sentence: [],
  };

  const threshold = () => parseFloat(els.threshold.value);
  const labelColor = (i) => `var(--c${i % 4})`;
  const isMirrored = () => state.mode === "camera" && els.mirror.checked;

  function cssVar(name) {
    return getComputedStyle(document.documentElement).getPropertyValue(name).trim();
  }

  function setStatus(text, isError = false) {
    els.statusPill.hidden = !text;
    els.statusPill.textContent = text || "";
    els.statusPill.classList.toggle("error", isError);
  }

  // ---------- drawing ----------

  function drawConnections(pts, connections, W, H, color, width, minVis = 0) {
    if (!pts) return;
    ctx.strokeStyle = color;
    ctx.lineWidth = width;
    ctx.beginPath();
    for (const [a, b] of connections) {
      const p = pts[a], q = pts[b];
      if (!p || !q) continue;
      if (minVis && ((p[2] ?? 1) < minVis || (q[2] ?? 1) < minVis)) continue;
      ctx.moveTo(p[0] * W, p[1] * H);
      ctx.lineTo(q[0] * W, q[1] * H);
    }
    ctx.stroke();
  }

  function drawPoints(pts, W, H, color, r, minVis = 0) {
    if (!pts) return;
    ctx.fillStyle = color;
    ctx.beginPath();
    for (const p of pts) {
      if (minVis && (p[2] ?? 1) < minVis) continue;
      ctx.moveTo(p[0] * W + r, p[1] * H);
      ctx.arc(p[0] * W, p[1] * H, r, 0, Math.PI * 2);
    }
    ctx.fill();
  }

  function drawLandmarks(lm, W, H) {
    const conn = state.info?.connections;
    if (!lm || !conn) return;
    const s = W / 640;
    ctx.save();
    ctx.lineCap = "round";
    if (els.showFace.checked) {
      drawConnections(lm.face, conn.face, W, H, "rgba(94, 234, 212, 0.75)", 1.2 * s);
    }
    if (els.showLandmarks.checked) {
      drawConnections(lm.pose, conn.pose, W, H, "rgba(255, 255, 255, 0.8)", 2.5 * s, 0.5);
      drawPoints(lm.pose, W, H, "rgba(255, 255, 255, 0.95)", 3 * s, 0.5);
      drawConnections(lm.left_hand, conn.hand, W, H, "#a78bfa", 2.5 * s);
      drawPoints(lm.left_hand, W, H, "#ede9fe", 2.6 * s);
      drawConnections(lm.right_hand, conn.hand, W, H, "#f472b6", 2.5 * s);
      drawPoints(lm.right_hand, W, H, "#fce7f3", 2.6 * s);
    }
    ctx.restore();
  }

  function render() {
    if (!state.running) return;
    const W = video.videoWidth, H = video.videoHeight;
    if (W && H) {
      if (display.width !== W || display.height !== H) { display.width = W; display.height = H; }
      ctx.save();
      if (isMirrored()) { ctx.translate(W, 0); ctx.scale(-1, 1); }
      ctx.drawImage(video, 0, 0, W, H);
      drawLandmarks(state.latest?.landmarks, W, H);
      ctx.restore();
    }
    requestAnimationFrame(render);
  }

  function snapshot() {
    const out = document.createElement("canvas");
    out.width = display.width; out.height = display.height;
    const o = out.getContext("2d");
    o.drawImage(display, 0, 0);
    if (!els.caption.hidden) {
      const fs = Math.round(out.width / 22);
      o.font = `700 ${fs}px system-ui, sans-serif`;
      const text = els.caption.textContent;
      const tw = o.measureText(text).width + fs * 1.4;
      const x = (out.width - tw) / 2, y = out.height - fs * 2.6;
      o.fillStyle = "rgba(0,0,0,0.62)";
      o.beginPath(); o.roundRect(x, y, tw, fs * 1.9, fs * 0.4); o.fill();
      o.fillStyle = "#fff"; o.textBaseline = "middle";
      o.fillText(text, x + fs * 0.7, y + fs * 0.95);
    }
    out.toBlob((blob) => {
      const a = document.createElement("a");
      a.href = URL.createObjectURL(blob);
      a.download = `action-snapshot-${Date.now()}.png`;
      a.click();
      setTimeout(() => URL.revokeObjectURL(a.href), 1000);
    }, "image/png");
  }

  // ---------- prediction UI ----------

  function buildBars() {
    els.bars.innerHTML = "";
    state.info.labels.forEach((label, i) => {
      const row = document.createElement("div");
      row.className = "bar-row";
      row.style.setProperty("--bar", labelColor(i));
      row.innerHTML = `<span class="name"></span><span class="value">0.00</span>
        <div class="bar-track"><div class="bar-fill"></div><div class="bar-mark"></div></div>`;
      row.querySelector(".name").textContent = label;
      els.bars.append(row);
    });
    updateThresholdMarks();
  }

  function updateThresholdMarks() {
    const t = threshold();
    els.thresholdOut.textContent = t.toFixed(2);
    els.bars.querySelectorAll(".bar-mark").forEach((m) => (m.style.left = `${t * 100}%`));
  }

  function renderChips() {
    els.chips.innerHTML = "";
    if (!state.sentence.length) {
      els.chips.innerHTML = '<span class="empty">Actions you sign will show up here.</span>';
    }
    for (const idx of state.sentence) {
      const chip = document.createElement("span");
      chip.className = "chip";
      chip.style.setProperty("--chip", labelColor(idx));
      chip.textContent = state.info.labels[idx];
      els.chips.append(chip);
    }
    els.clearBtn.disabled = !state.sentence.length;
  }

  function updatePrediction(r) {
    const labels = state.info?.labels;
    if (!labels) return;
    const seqLen = state.info.sequence_length;
    const buffered = r ? r.buffered : 0;

    els.bufferFill.style.width = `${(buffered / seqLen) * 100}%`;
    els.bufferText.textContent = buffered >= seqLen ? `Using the last ${seqLen} frames`
      : buffered ? `Collecting frames ${buffered}/${seqLen}` : "Frame buffer empty";
    els.bufferPill.hidden = !r || buffered >= seqLen;
    els.bufferPill.textContent = `Buffering ${buffered}/${seqLen}`;

    const lm = r?.landmarks;
    const person = lm && (lm.pose || lm.left_hand || lm.right_hand);
    const rows = els.bars.children;

    if (!r || !r.probs) {
      for (const row of rows) {
        row.classList.remove("top");
        row.querySelector(".bar-fill").style.width = "0";
        row.querySelector(".value").textContent = "–";
      }
      els.topName.textContent = "–";
      els.topConf.textContent = "";
      els.caption.hidden = true;
      setBadge(r ? (person ? "Buffering" : "No person") : "Idle", false);
      return;
    }

    const probs = r.probs;
    const top = probs.indexOf(Math.max(...probs));
    probs.forEach((p, i) => {
      rows[i].classList.toggle("top", i === top);
      rows[i].querySelector(".bar-fill").style.width = `${(p * 100).toFixed(1)}%`;
      rows[i].querySelector(".value").textContent = p.toFixed(2);
    });

    state.predHistory.push(top);
    if (state.predHistory.length > STABLE_FRAMES) state.predHistory.shift();
    const stable = !els.stable.checked ||
      (state.predHistory.length === STABLE_FRAMES && state.predHistory.every((p) => p === top));
    const confident = probs[top] >= threshold() && stable;

    els.topName.textContent = labels[top];
    els.topName.style.color = confident ? labelColor(top) : "";
    els.topConf.textContent = `${(probs[top] * 100).toFixed(0)}%`;

    if (confident) {
      if (state.sentence[state.sentence.length - 1] !== top) {
        state.sentence.push(top);
        if (state.sentence.length > MAX_SENTENCE) state.sentence.shift();
        renderChips();
      }
      els.caption.textContent = labels[top];
      els.caption.style.setProperty("--caption", labelColor(top));
      els.caption.hidden = false;
    } else {
      els.caption.hidden = true;
    }
    setBadge(confident ? "Detected" : person ? "Watching" : "No person", confident);
  }

  function setBadge(text, on) {
    els.stateBadge.textContent = text;
    els.stateBadge.classList.toggle("on", on);
  }

  function resetStats() {
    state.latest = null;
    state.predHistory = [];
    state.responseTimes = [];
    els.statFps.textContent = els.statRtt.textContent = els.statHolistic.textContent = els.statModel.textContent = "–";
    updatePrediction(null);
  }

  function handleResult(r, rtt) {
    if (r.error) { setStatus(r.error, true); return; }
    state.latest = r;
    const now = performance.now();
    state.responseTimes.push(now);
    while (state.responseTimes[0] < now - 1000) state.responseTimes.shift();
    els.statFps.textContent = state.responseTimes.length;
    els.statRtt.textContent = Math.round(rtt);
    els.statHolistic.textContent = r.holistic_ms.toFixed(0);
    els.statModel.textContent = r.model_ms != null ? r.model_ms.toFixed(0) : "–";
    updatePrediction(r);
  }

  // ---------- streaming ----------

  function connect() {
    const ws = new WebSocket(`${location.protocol === "https:" ? "wss" : "ws"}://${location.host}/ws`);
    state.ws = ws;
    state.inflight = false;
    setStatus("Connecting…");
    ws.onopen = () => { setStatus(""); pump(); };
    ws.onmessage = (ev) => {
      state.inflight = false;
      handleResult(JSON.parse(ev.data), performance.now() - state.sentAt);
      pump();
    };
    ws.onclose = () => {
      if (state.ws !== ws || !state.running) return;
      setStatus("Connection lost — retrying…", true);
      setTimeout(() => state.running && connect(), 1000);
    };
  }

  function sourceReady() {
    if (!video.videoWidth) return false;
    return state.mode === "camera" || (!video.paused && !video.ended);
  }

  async function pump() {
    const ws = state.ws;
    if (!state.running || state.inflight || !ws || ws.readyState !== WebSocket.OPEN) return;
    if (!sourceReady()) {
      // Paused video or camera still warming up: check again shortly.
      clearTimeout(state.retryTimer);
      state.retryTimer = setTimeout(pump, 100);
      return;
    }
    state.inflight = true;
    const scale = Math.min(1, SEND_WIDTH / video.videoWidth);
    capture.width = Math.round(video.videoWidth * scale);
    capture.height = Math.round(video.videoHeight * scale);
    cctx.drawImage(video, 0, 0, capture.width, capture.height);
    const blob = await new Promise((res) => capture.toBlob(res, "image/jpeg", 0.85));
    if (!state.running || ws.readyState !== WebSocket.OPEN) { state.inflight = false; return; }
    state.sentAt = performance.now();
    ws.send(await blob.arrayBuffer());
  }

  function startPipeline() {
    state.running = true;
    resetStats();
    connect();
    requestAnimationFrame(render);
  }

  function stopSource() {
    state.running = false;
    clearTimeout(state.retryTimer);
    if (state.ws) { const ws = state.ws; state.ws = null; ws.close(); }
    if (state.stream) state.stream.getTracks().forEach((t) => t.stop());
    state.stream = null;
    video.pause();
    video.srcObject = null;
    video.removeAttribute("src");
    if (state.videoUrl) { URL.revokeObjectURL(state.videoUrl); state.videoUrl = null; }
    els.startBtn.textContent = "Start camera";
    els.snapshotBtn.disabled = els.snapshotBtn2.disabled = els.playBtn.disabled = true;
    setStatus("");
    els.caption.hidden = els.bufferPill.hidden = true;
  }

  function clearStage() {
    resetStats();
    ctx.clearRect(0, 0, display.width, display.height);
  }

  // ---------- camera ----------

  async function listCameras(activeId) {
    const devices = (await navigator.mediaDevices.enumerateDevices()).filter((d) => d.kind === "videoinput");
    els.cameraSelect.innerHTML = "";
    devices.forEach((d, i) => {
      const opt = new Option(d.label || `Camera ${i + 1}`, d.deviceId);
      opt.selected = d.deviceId === activeId;
      els.cameraSelect.add(opt);
    });
    els.cameraSelect.disabled = devices.length < 2;
  }

  async function startCamera(deviceId) {
    if (!navigator.mediaDevices?.getUserMedia) {
      setStatus("Camera access needs a secure context (use http://localhost).", true);
      return;
    }
    stopSource();
    setStatus("Requesting camera…");
    try {
      const size = { width: { ideal: 640 }, height: { ideal: 480 } };
      state.stream = await navigator.mediaDevices.getUserMedia({
        video: deviceId ? { deviceId: { exact: deviceId }, ...size } : size,
        audio: false,
      });
    } catch (err) {
      setStatus(`Camera unavailable: ${err.message || err.name}`, true);
      return;
    }
    video.srcObject = state.stream;
    await video.play();
    els.cameraPlaceholder.hidden = true;
    els.startBtn.textContent = "Stop camera";
    els.snapshotBtn.disabled = false;
    listCameras(state.stream.getVideoTracks()[0]?.getSettings().deviceId).catch(() => {});
    startPipeline();
  }

  // ---------- video file ----------

  async function loadVideo(file) {
    if (!file || !file.type.startsWith("video/")) {
      setStatus("That file isn't a video.", true);
      return;
    }
    stopSource();
    state.videoUrl = URL.createObjectURL(file);
    video.src = state.videoUrl;
    video.loop = true;
    try {
      await video.play();
    } catch (err) {
      setStatus(`Could not play that video: ${err.message || err.name}`, true);
      return;
    }
    els.dropzone.hidden = true;
    els.videoName.textContent = file.name;
    els.playBtn.disabled = els.snapshotBtn2.disabled = false;
    els.playBtn.textContent = "Pause";
    startPipeline();
  }

  // Seeking or looping breaks the frame sequence, so start the buffer over.
  video.addEventListener("seeked", () => {
    if (state.mode !== "video" || state.ws?.readyState !== WebSocket.OPEN) return;
    state.ws.send("reset");
    state.predHistory = [];
  });

  // ---------- mode switching ----------

  function setMode(mode) {
    if (state.mode === mode) return;
    stopSource();
    state.mode = mode;
    els.tabs.forEach((t) => {
      const active = t.dataset.mode === mode;
      t.classList.toggle("active", active);
      t.setAttribute("aria-selected", active);
    });
    const camera = mode === "camera";
    els.cameraToolbar.hidden = !camera;
    els.videoToolbar.hidden = camera;
    els.mirrorSwitch.hidden = !camera;
    els.cameraPlaceholder.hidden = !camera;
    els.dropzone.hidden = camera;
    els.videoName.textContent = "";
    clearStage();
  }

  // ---------- events ----------

  els.tabs.forEach((t) => t.addEventListener("click", () => setMode(t.dataset.mode)));

  els.startBtn.addEventListener("click", () => {
    if (!state.running) return startCamera();
    stopSource();
    clearStage();
    els.cameraPlaceholder.hidden = false;
  });
  els.startBtnInline.addEventListener("click", () => startCamera());
  els.cameraSelect.addEventListener("change", () => startCamera(els.cameraSelect.value));
  els.snapshotBtn.addEventListener("click", snapshot);
  els.snapshotBtn2.addEventListener("click", snapshot);

  els.chooseBtn.addEventListener("click", () => els.fileInput.click());
  els.fileInput.addEventListener("change", () => { loadVideo(els.fileInput.files[0]); els.fileInput.value = ""; });
  els.playBtn.addEventListener("click", () => {
    if (video.paused) video.play(); else video.pause();
  });
  video.addEventListener("play", () => (els.playBtn.textContent = "Pause"));
  video.addEventListener("pause", () => (els.playBtn.textContent = "Play"));

  const stage = $("stage");
  stage.addEventListener("dragover", (e) => {
    if (state.mode !== "video") return;
    e.preventDefault();
    els.dropzone.classList.add("dragging");
  });
  stage.addEventListener("dragleave", () => els.dropzone.classList.remove("dragging"));
  stage.addEventListener("drop", (e) => {
    if (state.mode !== "video") return;
    e.preventDefault();
    els.dropzone.classList.remove("dragging");
    loadVideo(e.dataTransfer.files[0]);
  });

  els.threshold.addEventListener("input", updateThresholdMarks);
  els.clearBtn.addEventListener("click", () => { state.sentence = []; renderChips(); });

  fetch("/api/info")
    .then((r) => r.json())
    .then((info) => {
      state.info = info;
      els.modelChip.textContent = `${info.model} · ${info.sequence_length}×${info.keypoints} · ${info.labels.length} actions · ${(info.parameters / 1e3).toFixed(0)}K params`;
      buildBars();
      updatePrediction(null);
    })
    .catch(() => setStatus("Could not load model info from the server.", true));
})();
