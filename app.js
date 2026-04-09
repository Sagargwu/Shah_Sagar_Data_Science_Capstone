const uploadForm = document.getElementById("uploadForm");
const videoInput = document.getElementById("videoInput");
const selectedFileName = document.getElementById("selectedFileName");
const processBtn = document.getElementById("processBtn");
const statusBox = document.getElementById("statusBox");
const statusText = document.getElementById("statusText");
const dropZone = document.getElementById("dropZone");

const originalVideo = document.getElementById("originalVideo");
const processedVideo = document.getElementById("processedVideo");
const originalPlaceholder = document.getElementById("originalPlaceholder");
const processedPlaceholder = document.getElementById("processedPlaceholder");

const statsGrid = document.getElementById("statsGrid");
const downloadBtn = document.getElementById("downloadBtn");
const syncPlayBtn = document.getElementById("syncPlayBtn");

const heatmapImage = document.getElementById("heatmapImage");
const heatmapPlaceholder = document.getElementById("heatmapPlaceholder");

const dashboardMetricsGrid = document.getElementById("dashboardMetricsGrid");

const chartHeatmap = document.getElementById("chartHeatmap");
const chartConfidence = document.getElementById("chartConfidence");
const chartLatency = document.getElementById("chartLatency");
const chartCoverage = document.getElementById("chartCoverage");
const chartDayNight = document.getElementById("chartDayNight");
const chartBestFrame = document.getElementById("chartBestFrame");
const chartWorstFrame = document.getElementById("chartWorstFrame");

const chartHeatmapPlaceholder = document.getElementById("chartHeatmapPlaceholder");
const chartConfidencePlaceholder = document.getElementById("chartConfidencePlaceholder");
const chartLatencyPlaceholder = document.getElementById("chartLatencyPlaceholder");
const chartCoveragePlaceholder = document.getElementById("chartCoveragePlaceholder");
const chartDayNightPlaceholder = document.getElementById("chartDayNightPlaceholder");
const chartBestFramePlaceholder = document.getElementById("chartBestFramePlaceholder");
const chartWorstFramePlaceholder = document.getElementById("chartWorstFramePlaceholder");

let selectedFile = null;
let originalObjectUrl = null;

function setStatus(type, message) {
  statusBox.classList.remove("processing", "success", "error");
  if (type) statusBox.classList.add(type);
  statusText.textContent = message;
}

function formatNumber(value, digits = 2) {
  if (value === null || value === undefined || Number.isNaN(Number(value))) {
    return "N/A";
  }
  return Number(value).toFixed(digits);
}

function renderStats(items) {
  statsGrid.innerHTML = "";
  items.forEach(item => {
    const div = document.createElement("div");
    div.className = "stat-item";
    div.innerHTML = `
      <div class="stat-label">${item.label}</div>
      <div class="stat-value">${item.value}</div>
    `;
    statsGrid.appendChild(div);
  });
}

function renderDashboardMetrics(stats = {}) {
  if (!dashboardMetricsGrid) return;

  dashboardMetricsGrid.innerHTML = `
    <div class="stat-item">
      <div class="stat-label">Processed Frames</div>
      <div class="stat-value">${stats.processed_frames ?? "N/A"}</div>
    </div>
    <div class="stat-item">
      <div class="stat-label">Average Latency</div>
      <div class="stat-value">${formatNumber(stats.avg_inference_latency_ms)} ms</div>
    </div>
    <div class="stat-item">
      <div class="stat-label">Processing FPS</div>
      <div class="stat-value">${formatNumber(stats.processing_fps)}</div>
    </div>
    <div class="stat-item">
      <div class="stat-label">Mean Confidence</div>
      <div class="stat-value">${formatNumber((stats.mean_lane_confidence || 0) * 100)}%</div>
    </div>
    <div class="stat-item">
      <div class="stat-label">Mean Lane Coverage</div>
      <div class="stat-value">${formatNumber((stats.mean_lane_coverage || 0) * 100)}%</div>
    </div>
    <div class="stat-item">
      <div class="stat-label">Mean Brightness</div>
      <div class="stat-value">${formatNumber(stats.mean_brightness)}</div>
    </div>
    <div class="stat-item">
      <div class="stat-label">Day Frames</div>
      <div class="stat-value">${stats.day_frames_detected ?? "N/A"}</div>
    </div>
    <div class="stat-item">
      <div class="stat-label">Night Frames</div>
      <div class="stat-value">${stats.night_frames_detected ?? "N/A"}</div>
    </div>
  `;
}

function resetImage(imgEl, placeholderEl) {
  if (!imgEl || !placeholderEl) return;
  imgEl.removeAttribute("src");
  imgEl.classList.add("hidden");
  placeholderEl.classList.remove("hidden");
}

function showDashboardImage(imgEl, placeholderEl, src) {
  if (!imgEl || !placeholderEl || !src) return;
  imgEl.src = `${src}?t=${Date.now()}`;
  imgEl.classList.remove("hidden");
  placeholderEl.classList.add("hidden");
}

function resetDashboardCharts() {
  resetImage(chartHeatmap, chartHeatmapPlaceholder);
  resetImage(chartConfidence, chartConfidencePlaceholder);
  resetImage(chartLatency, chartLatencyPlaceholder);
  resetImage(chartCoverage, chartCoveragePlaceholder);
  resetImage(chartDayNight, chartDayNightPlaceholder);
  resetImage(chartBestFrame, chartBestFramePlaceholder);
  resetImage(chartWorstFrame, chartWorstFramePlaceholder);

  if (dashboardMetricsGrid) {
    dashboardMetricsGrid.innerHTML = `
      <div class="stat-item">
        <div class="stat-label">Dashboard Status</div>
        <div class="stat-value">Waiting for analysis</div>
      </div>
    `;
  }
}

function setSelectedFile(file) {
  selectedFile = file;
  selectedFileName.textContent = file ? file.name : "No file selected";

  if (originalObjectUrl) {
    URL.revokeObjectURL(originalObjectUrl);
    originalObjectUrl = null;
  }

  if (file) {
    originalObjectUrl = URL.createObjectURL(file);
    originalVideo.src = originalObjectUrl;
    originalVideo.classList.remove("hidden");
    originalPlaceholder.classList.add("hidden");
    setStatus("", "Video selected. Ready to process.");
  } else {
    originalVideo.removeAttribute("src");
    originalVideo.load();
    originalVideo.classList.add("hidden");
    originalPlaceholder.classList.remove("hidden");
    setStatus("", "Waiting for video upload...");
  }

  processedVideo.pause();
  processedVideo.removeAttribute("src");
  processedVideo.load();
  processedVideo.classList.add("hidden");
  processedPlaceholder.classList.remove("hidden");

  heatmapImage.removeAttribute("src");
  heatmapImage.classList.add("hidden");
  heatmapPlaceholder.classList.remove("hidden");

  downloadBtn.classList.add("hidden");
  downloadBtn.removeAttribute("href");

  renderStats([
    { label: "Status", value: file ? "Ready to process" : "Waiting" }
  ]);

  resetDashboardCharts();
}

videoInput.addEventListener("change", (e) => {
  const file = e.target.files[0];
  setSelectedFile(file || null);
});

dropZone.addEventListener("dragover", (e) => {
  e.preventDefault();
  dropZone.classList.add("dragover");
});

dropZone.addEventListener("dragleave", () => {
  dropZone.classList.remove("dragover");
});

dropZone.addEventListener("drop", (e) => {
  e.preventDefault();
  dropZone.classList.remove("dragover");

  const file = e.dataTransfer.files[0];
  if (!file) return;

  videoInput.files = e.dataTransfer.files;
  setSelectedFile(file);
});

uploadForm.addEventListener("submit", async (e) => {
  e.preventDefault();

  if (!selectedFile) {
    setStatus("error", "Please select a video file first.");
    return;
  }

  const formData = new FormData();
  formData.append("video", selectedFile);

  processBtn.disabled = true;
  processBtn.textContent = "Processing...";
  setStatus("processing", "Processing video... please wait.");

  renderStats([
    { label: "Status", value: "Processing started" },
    { label: "File", value: selectedFile.name },
    { label: "Mode", value: "Frame-by-frame lane inference" },
    { label: "Visualization", value: "Preparing analytics..." }
  ]);

  renderDashboardMetrics({
    processed_frames: "Running...",
    avg_inference_latency_ms: null,
    processing_fps: null,
    mean_lane_confidence: null,
    mean_lane_coverage: null,
    mean_brightness: null,
    day_frames_detected: null,
    night_frames_detected: null
  });

  try {
    const response = await fetch("/api/process", {
      method: "POST",
      body: formData
    });

    const data = await response.json();

    if (!response.ok || !data.ok) {
      throw new Error(data.message || "Processing failed.");
    }

    processedVideo.src = data.preview_url;
    processedVideo.classList.remove("hidden");
    processedPlaceholder.classList.add("hidden");
    processedVideo.load();

    if (data.assets && data.assets.heatmap) {
      heatmapImage.src = `${data.assets.heatmap}?t=${Date.now()}`;
      heatmapImage.classList.remove("hidden");
      heatmapPlaceholder.classList.add("hidden");
    }

    if (data.download_url) {
      downloadBtn.href = data.download_url;
      downloadBtn.classList.remove("hidden");
    }

    const stats = data.stats || {};
    renderStats([
      { label: "Input Video", value: stats.input_video || "N/A" },
      { label: "Output Video", value: stats.output_video || "N/A" },
      { label: "Duration", value: `${formatNumber(stats.duration_seconds)} s` },
      { label: "Input FPS", value: formatNumber(stats.input_fps) },
      { label: "Frames Processed", value: stats.processed_frames ?? "N/A" },
      { label: "Avg Inference Latency", value: `${formatNumber(stats.avg_inference_latency_ms)} ms/frame` },
      { label: "End-to-End Time", value: `${formatNumber(stats.processing_seconds)} s` },
      { label: "Processing Throughput", value: `${formatNumber(stats.processing_fps)} FPS` },
      { label: "Mean Lane Confidence", value: `${formatNumber((stats.mean_lane_confidence || 0) * 100)}%` },
      { label: "Night Frames", value: stats.night_frames_detected ?? "N/A" }
    ]);

    renderDashboardMetrics(stats);

    const assets = data.assets || {};
    showDashboardImage(chartHeatmap, chartHeatmapPlaceholder, assets.heatmap);
    showDashboardImage(chartConfidence, chartConfidencePlaceholder, assets.confidence_plot);
    showDashboardImage(chartLatency, chartLatencyPlaceholder, assets.latency_plot);
    showDashboardImage(chartCoverage, chartCoveragePlaceholder, assets.coverage_histogram);
    showDashboardImage(chartDayNight, chartDayNightPlaceholder, assets.day_night_chart);
    showDashboardImage(chartBestFrame, chartBestFramePlaceholder, assets.best_frame);
    showDashboardImage(chartWorstFrame, chartWorstFramePlaceholder, assets.worst_frame);

    setStatus("success", data.message || "Lane detection completed successfully.");
  } catch (error) {
    setStatus("error", error.message || "Something went wrong.");

    renderStats([
      { label: "Status", value: "Processing failed" },
      { label: "Reason", value: error.message || "Unknown error" }
    ]);

    if (dashboardMetricsGrid) {
      dashboardMetricsGrid.innerHTML = `
        <div class="stat-item">
          <div class="stat-label">Dashboard Status</div>
          <div class="stat-value">Processing failed</div>
        </div>
        <div class="stat-item">
          <div class="stat-label">Reason</div>
          <div class="stat-value">${error.message || "Unknown error"}</div>
        </div>
      `;
    }
  } finally {
    processBtn.disabled = false;
    processBtn.textContent = "Process Video";
  }
});

syncPlayBtn.addEventListener("click", () => {
  if (!originalVideo.src || !processedVideo.src) {
    setStatus("error", "Load and process a video first before sync playback.");
    return;
  }

  const currentTime = Math.min(originalVideo.currentTime || 0, processedVideo.currentTime || 0);
  originalVideo.currentTime = currentTime;
  processedVideo.currentTime = currentTime;

  const playOriginal = originalVideo.play();
  const playProcessed = processedVideo.play();

  Promise.allSettled([playOriginal, playProcessed]).then(() => {
    setStatus("success", "Original and processed videos are now playing together.");
  });
});

function syncFromSource(source, target) {
  if (!source || !target) return;
  if (Math.abs((source.currentTime || 0) - (target.currentTime || 0)) > 0.35) {
    target.currentTime = source.currentTime;
  }
}

originalVideo.addEventListener("play", () => {
  if (processedVideo.src && processedVideo.paused) {
    processedVideo.currentTime = originalVideo.currentTime;
    processedVideo.play().catch(() => {});
  }
});

processedVideo.addEventListener("play", () => {
  if (originalVideo.src && originalVideo.paused) {
    originalVideo.currentTime = processedVideo.currentTime;
    originalVideo.play().catch(() => {});
  }
});

originalVideo.addEventListener("pause", () => {
  if (!processedVideo.paused) processedVideo.pause();
});

processedVideo.addEventListener("pause", () => {
  if (!originalVideo.paused) originalVideo.pause();
});

originalVideo.addEventListener("seeked", () => {
  syncFromSource(originalVideo, processedVideo);
});

processedVideo.addEventListener("seeked", () => {
  syncFromSource(processedVideo, originalVideo);
});

renderStats([
  { label: "Status", value: "Waiting" }
]);

resetDashboardCharts();
