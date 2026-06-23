import { FaceLandmarker, FilesetResolver, DrawingUtils } from "https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.22-rc.20250304";

const EMOTIONS = [
  { key: "anxiety", zh: "焦虑", en: "Anxiety" },
  { key: "confusion", zh: "困惑", en: "Confusion" },
  { key: "fatigue", zh: "疲劳", en: "Fatigue" },
  { key: "concentration", zh: "专注", en: "Concentration" },
  { key: "excitement", zh: "兴奋", en: "Excitement" },
  { key: "boredom", zh: "厌倦", en: "Boredom" },
  { key: "stress", zh: "压力", en: "Stress" },
];

const video = document.querySelector("#webcam");
const canvas = document.querySelector("#overlay");
const ctx = canvas.getContext("2d");
const button = document.querySelector("#cameraButton");
const statusEl = document.querySelector("#runtimeStatus");
const listEl = document.querySelector("#emotionList");
const dominantEmotion = document.querySelector("#dominantEmotion");
const dominantScore = document.querySelector("#dominantScore");
const fpsEl = document.querySelector("#fps");
const qualityEl = document.querySelector("#quality");
const stableFramesEl = document.querySelector("#stableFrames");

const rows = new Map();
let landmarker;
let running = false;
let lastVideoTime = -1;
let lastTick = performance.now();
let stableFrames = 0;
let smoothed = Object.fromEntries(EMOTIONS.map(({ key }) => [key, 0]));
let previousTop = "";

for (const emotion of EMOTIONS) {
  const row = document.createElement("div");
  row.className = "emotion-row";
  row.innerHTML = `<div class="emotion-top"><span>${emotion.zh} <small>${emotion.en}</small></span><b>0%</b></div><div class="bar"><i></i></div>`;
  listEl.append(row);
  rows.set(emotion.key, row);
}

const clamp01 = (value) => Math.min(1, Math.max(0, value));
const score = (value, low, high) => clamp01((value - low) / (high - low));
const byName = (cats, name) => cats.find((cat) => cat.categoryName === name)?.score ?? 0;
const dist = (a, b) => Math.hypot(a.x - b.x, a.y - b.y);

async function loadModel() {
  statusEl.textContent = "正在加载模型...";
  const fileset = await FilesetResolver.forVisionTasks("https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.22-rc.20250304/wasm");
  landmarker = await FaceLandmarker.createFromOptions(fileset, {
    baseOptions: {
      modelAssetPath: "https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/latest/face_landmarker.task",
      delegate: "GPU",
    },
    runningMode: "VIDEO",
    numFaces: 2,
    outputFaceBlendshapes: true,
    outputFacialTransformationMatrixes: true,
    minFaceDetectionConfidence: 0.62,
    minFacePresenceConfidence: 0.62,
    minTrackingConfidence: 0.55,
  });
  statusEl.textContent = "模型已就绪";
  statusEl.className = "status-pill ready";
}

function estimateAffect(landmarks, blendshapes, matrix) {
  const cats = blendshapes?.categories ?? [];
  const mouthSmile = (byName(cats, "mouthSmileLeft") + byName(cats, "mouthSmileRight")) / 2;
  const browDown = (byName(cats, "browDownLeft") + byName(cats, "browDownRight")) / 2;
  const browInnerUp = byName(cats, "browInnerUp");
  const eyeSquint = (byName(cats, "eyeSquintLeft") + byName(cats, "eyeSquintRight")) / 2;
  const blink = (byName(cats, "eyeBlinkLeft") + byName(cats, "eyeBlinkRight")) / 2;
  const jawOpen = byName(cats, "jawOpen");
  const mouthFrown = (byName(cats, "mouthFrownLeft") + byName(cats, "mouthFrownRight")) / 2;
  const mouthPress = (byName(cats, "mouthPressLeft") + byName(cats, "mouthPressRight")) / 2;
  const eyeWide = (byName(cats, "eyeWideLeft") + byName(cats, "eyeWideRight")) / 2;

  const faceWidth = dist(landmarks[234], landmarks[454]);
  const faceHeight = dist(landmarks[10], landmarks[152]);
  const aspect = faceHeight / Math.max(faceWidth, 0.001);
  const yawProxy = matrix?.data ? Math.abs(matrix.data[8]) : Math.abs(landmarks[1].x - 0.5) * 2;
  const pitchProxy = score(aspect, 1.35, 1.75);
  const attention = clamp01(1 - yawProxy * 1.9 - blink * 0.8);

  return {
    anxiety: clamp01(0.34 * browInnerUp + 0.26 * eyeWide + 0.22 * mouthPress + 0.18 * score(jawOpen, 0.08, 0.35)),
    confusion: clamp01(0.36 * browDown + 0.24 * browInnerUp + 0.24 * yawProxy + 0.16 * score(eyeSquint, 0.08, 0.35)),
    fatigue: clamp01(0.42 * blink + 0.22 * pitchProxy + 0.18 * mouthFrown + 0.18 * (1 - attention)),
    concentration: clamp01(0.48 * attention + 0.22 * eyeSquint + 0.18 * browDown + 0.12 * (1 - jawOpen)),
    excitement: clamp01(0.46 * mouthSmile + 0.28 * eyeWide + 0.18 * score(jawOpen, 0.1, 0.45) + 0.08 * browInnerUp),
    boredom: clamp01(0.34 * (1 - attention) + 0.26 * mouthFrown + 0.22 * blink + 0.18 * (1 - mouthSmile)),
    stress: clamp01(0.34 * browDown + 0.25 * mouthPress + 0.21 * eyeWide + 0.2 * browInnerUp),
  };
}

function updateUi(scores, quality) {
  for (const [key, raw] of Object.entries(scores)) {
    smoothed[key] = smoothed[key] * 0.74 + raw * 0.26;
    const percent = Math.round(smoothed[key] * 100);
    const row = rows.get(key);
    row.querySelector("i").style.width = `${percent}%`;
    row.querySelector("b").textContent = `${percent}%`;
  }
  const sorted = EMOTIONS.map((e) => ({ ...e, value: smoothed[e.key] })).sort((a, b) => b.value - a.value);
  const top = sorted[0];
  if (!previousTop || top.value - (smoothed[previousTop] ?? 0) > 0.08) previousTop = top.key;
  const stableTop = EMOTIONS.find((e) => e.key === previousTop) ?? top;
  dominantEmotion.textContent = `${stableTop.zh} / ${stableTop.en}`;
  dominantScore.textContent = `${Math.round(smoothed[stableTop.key] * 100)}%`;
  qualityEl.textContent = `${Math.round(quality * 100)}%`;
  stableFramesEl.textContent = stableFrames;
}

function draw(landmarks) {
  ctx.clearRect(0, 0, canvas.width, canvas.height);
  if (!landmarks) return;
  const drawingUtils = new DrawingUtils(ctx);
  drawingUtils.drawConnectors(landmarks, FaceLandmarker.FACE_LANDMARKS_TESSELATION, { color: "rgba(113,231,255,.18)", lineWidth: 1 });
  drawingUtils.drawConnectors(landmarks, FaceLandmarker.FACE_LANDMARKS_FACE_OVAL, { color: "#8cffb5", lineWidth: 2 });
  drawingUtils.drawConnectors(landmarks, FaceLandmarker.FACE_LANDMARKS_LEFT_EYE, { color: "#ffd166", lineWidth: 2 });
  drawingUtils.drawConnectors(landmarks, FaceLandmarker.FACE_LANDMARKS_RIGHT_EYE, { color: "#ffd166", lineWidth: 2 });
}

function predict() {
  if (!running) return;
  if (video.videoWidth && (canvas.width !== video.videoWidth || canvas.height !== video.videoHeight)) {
    canvas.width = video.videoWidth;
    canvas.height = video.videoHeight;
  }
  if (video.currentTime !== lastVideoTime) {
    lastVideoTime = video.currentTime;
    const now = performance.now();
    const result = landmarker.detectForVideo(video, now);
    fpsEl.textContent = Math.round(1000 / Math.max(1, now - lastTick));
    lastTick = now;
    if (result.faceLandmarks.length) {
      stableFrames += 1;
      const quality = clamp01(dist(result.faceLandmarks[0][234], result.faceLandmarks[0][454]) * 2.8);
      const scores = estimateAffect(result.faceLandmarks[0], result.faceBlendshapes[0], result.facialTransformationMatrixes[0]);
      updateUi(scores, quality);
      draw(result.faceLandmarks[0]);
      statusEl.textContent = "实时检测中";
      statusEl.className = "status-pill ready";
    } else {
      stableFrames = Math.max(0, stableFrames - 1);
      draw(null);
      statusEl.textContent = "未检测到稳定人脸";
      statusEl.className = "status-pill warn";
    }
  }
  requestAnimationFrame(predict);
}

button.addEventListener("click", async () => {
  if (!landmarker) await loadModel();
  const stream = await navigator.mediaDevices.getUserMedia({ video: { width: 1280, height: 720, frameRate: { ideal: 30 } }, audio: false });
  video.srcObject = stream;
  await video.play();
  running = true;
  button.textContent = "摄像头运行中";
  button.disabled = true;
  predict();
});

loadModel().catch((error) => {
  console.error(error);
  statusEl.textContent = "模型加载失败，请检查网络";
  statusEl.className = "status-pill warn";
});
