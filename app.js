const uploadForm = document.getElementById("uploadForm");
const videoInput = document.getElementById("videoInput");
const selectedFileName = document.getElementById("selectedFileName");
const statusBox = document.getElementById("statusBox");
const statusText = document.getElementById("statusText");
const processBtn = document.getElementById("processBtn");
const resultVideo = document.getElementById("resultVideo");
const videoPlaceholder = document.getElementById("videoPlaceholder");
const downloadBtn = document.getElementById("downloadBtn");
const statsGrid = document.getElementById("statsGrid");
const dropZone = document.getElementById("dropZone");

videoInput.addEventListener("change", () => {
  if (videoInput.files.length > 0) {
    selectedFileName.textContent = videoInput.files[0].name;
  } else {
    selectedFileName.textContent = "No file selected";
  }
});

["dragenter", "dragover"].forEach((eventName) => {
  dropZone.addEventListener(eventName, (e) => {
    e.preventDefault();
    e.stopPropagation();
    dropZone.classList.add("dragover");
  });
});

["dragleave", "drop"].forEach((eventName) => {
  dropZone.addEventListener(eventName, (e) => {
    e.preventDefault();
    e.stopPropagation();
    dropZone.classList.remove("dragover");
  });
});

dropZone.addEventListener("drop", (e) => {
  const files = e.dataTransfer.files;
  if (files.length > 0) {
    videoInput.files = files;
    selectedFileName.textContent = files[0].name;
  }
});

function setStatus(type, text) {
  statusBox.classList.remove("processing", "success", "error");
  if (type) {
    statusBox.classList.add(type);
  }
  statusText.textContent = text;
}

function renderStats(stats) {
  statsGrid.innerHTML = "";

  const statEntries = [
    ["Input Video", stats.input_video],
    ["Output Video", stats.output_video],
    ["Duration (s)", stats.duration_seconds],
    ["FPS", stats.fps],
    ["Approx Frames", stats.approx_frames],
    ["Processing Time (s)", stats.processing_seconds],
  ];

  statEntries.forEach(([label, value]) => {
    const item = document.createElement("div");
    item.className = "stat-item";
    item.innerHTML = `
      <div class="stat-label">${label}</div>
      <div class="stat-value">${value}</div>
    `;
    statsGrid.appendChild(item);
  });
}

uploadForm.addEventListener("submit", async (e) => {
  e.preventDefault();

  if (!videoInput.files.length) {
    setStatus("error", "Please select a video file first.");
    return;
  }

  const file = videoInput.files[0];
  const formData = new FormData();
  formData.append("video", file);

  setStatus("processing", "Processing video... please wait.");
  processBtn.disabled = true;
  processBtn.textContent = "Processing...";
  downloadBtn.classList.add("hidden");
  resultVideo.classList.add("hidden");
  videoPlaceholder.classList.remove("hidden");
  statsGrid.innerHTML = "";

  try {
    const response = await fetch("/api/process", {
      method: "POST",
      body: formData,
    });

    const data = await response.json();

    if (!response.ok || !data.ok) {
      throw new Error(data.message || "Processing failed.");
    }

    setStatus("success", data.message);

    resultVideo.src = data.preview_url;
    resultVideo.load();
    resultVideo.classList.remove("hidden");
    videoPlaceholder.classList.add("hidden");

    downloadBtn.href = data.download_url;
    downloadBtn.classList.remove("hidden");

    renderStats(data.stats);
  } catch (error) {
    setStatus("error", error.message);
  } finally {
    processBtn.disabled = false;
    processBtn.textContent = "Process Video";
  }
}); 