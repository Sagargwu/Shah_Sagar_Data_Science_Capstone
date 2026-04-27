const uploadForm = document.getElementById("uploadForm");
const videoInput = document.getElementById("videoInput");
const dropZone = document.getElementById("dropZone");
const selectedFileName = document.getElementById("selectedFileName");
const processBtn = document.getElementById("processBtn");

const statusBox = document.getElementById("statusBox");
const statusText = document.getElementById("statusText");

const originalVideo = document.getElementById("originalVideo");
const processedVideo = document.getElementById("processedVideo");
const originalPlaceholder = document.getElementById("originalPlaceholder");
const processedPlaceholder = document.getElementById("processedPlaceholder");
const downloadBtn = document.getElementById("downloadBtn");
const syncPlayBtn = document.getElementById("syncPlayBtn");

const statsGrid = document.getElementById("statsGrid");
const dashboardMetricsGrid = document.getElementById("dashboardMetricsGrid");

const heatmapImage = document.getElementById("heatmapImage");
const heatmapPlaceholder = document.getElementById("heatmapPlaceholder");

const chartRiskTrend = document.getElementById("chartRiskTrend");
const chartOffsetTrend = document.getElementById("chartOffsetTrend");
const chartZoneBar = document.getElementById("chartZoneBar");
const chartDayNight = document.getElementById("chartDayNight");
const chartRiskPie = document.getElementById("chartRiskPie");
const chartBestFrame = document.getElementById("chartBestFrame");

const chartRiskTrendPlaceholder = document.getElementById("chartRiskTrendPlaceholder");
const chartOffsetTrendPlaceholder = document.getElementById("chartOffsetTrendPlaceholder");
const chartZoneBarPlaceholder = document.getElementById("chartZoneBarPlaceholder");
const chartDayNightPlaceholder = document.getElementById("chartDayNightPlaceholder");
const chartRiskPiePlaceholder = document.getElementById("chartRiskPiePlaceholder");
const chartBestFramePlaceholder = document.getElementById("chartBestFramePlaceholder");

const filterRiskBadge = document.getElementById("filterRiskBadge");
const filterSummaryList = document.getElementById("filterSummaryList");
const filterButtons = document.querySelectorAll(".filter-btn");
const filterItems = document.querySelectorAll(".dashboard-filter-item");

function setStatus(type, message) {
  statusBox.classList.remove("processing", "success", "error");
  if (type) {
    statusBox.classList.add(type);
  }
  statusText.textContent = message;
}

function formatValue(value, suffix = "") {
  if (value === undefined || value === null || Number.isNaN(value)) {
    return "N/A";
  }
  return `${value}${suffix}`;
}

function getRiskClass(risk) {
  if (risk === "LOW") return "risk-low";
  if (risk === "MEDIUM") return "risk-medium";
  return "risk-high";
}

function getRiskBadgeClass(risk) {
  if (risk === "LOW") return "low";
  if (risk === "MEDIUM") return "medium";
  if (risk === "HIGH") return "high";
  return "neutral";
}

function createStatCard(label, value, extraClass = "") {
  return `
    <div class="stat-item">
      <div class="stat-label">${label}</div>
      <div class="stat-value ${extraClass}">${value}</div>
    </div>
  `;
}

function resetImage(imgEl, placeholderEl) {
  imgEl.src = "";
  imgEl.classList.add("hidden");
  placeholderEl.classList.remove("hidden");
}

function setImage(imgEl, placeholderEl, src) {
  if (!src) {
    resetImage(imgEl, placeholderEl);
    return;
  }
  imgEl.src = `${src}?t=${Date.now()}`;
  imgEl.classList.remove("hidden");
  placeholderEl.classList.add("hidden");
}

function updateFilterBadge(risk) {
  filterRiskBadge.className = `risk-badge ${getRiskBadgeClass(risk)}`;
  filterRiskBadge.textContent = risk || "Waiting";
}

function updateFilterSummary(stats) {
  filterSummaryList.innerHTML = `
    <li><strong>Average Risk:</strong> ${formatValue(stats.avg_risk_score, "%")}</li>
    <li><strong>Lane Departures:</strong> ${formatValue(stats.lane_departure_events)}</li>
    <li><strong>Safe Frames:</strong> ${formatValue(stats.safe_frames)}</li>
    <li><strong>Caution Frames:</strong> ${formatValue(stats.caution_frames)}</li>
    <li><strong>Danger Frames:</strong> ${formatValue(stats.danger_frames)}</li>
    <li><strong>Night Frames:</strong> ${formatValue(stats.night_frames_detected)}</li>
    <li><strong>Snow Frames:</strong> ${formatValue(stats.snow_frames_detected)}</li>
  `;
}

function resetDashboard() {
  statsGrid.innerHTML = createStatCard("Status", "Waiting");
  dashboardMetricsGrid.innerHTML = createStatCard("Dashboard Status", "Waiting for analysis");

  resetImage(heatmapImage, heatmapPlaceholder);
  resetImage(chartRiskTrend, chartRiskTrendPlaceholder);
  resetImage(chartOffsetTrend, chartOffsetTrendPlaceholder);
  resetImage(chartZoneBar, chartZoneBarPlaceholder);
  resetImage(chartDayNight, chartDayNightPlaceholder);
  resetImage(chartRiskPie, chartRiskPiePlaceholder);
  resetImage(chartBestFrame, chartBestFramePlaceholder);

  updateFilterBadge("Waiting");
  filterSummaryList.innerHTML = "<li>Waiting for processing...</li>";
}

function updateStats(stats) {
  const riskClass = getRiskClass(stats.overall_risk);

  statsGrid.innerHTML = `
    ${createStatCard("Input Video", stats.input_video || "N/A")}
    ${createStatCard("Output Video", stats.output_video || "N/A")}
    ${createStatCard("Duration", formatValue(stats.duration_seconds, " s"))}
    ${createStatCard("Input FPS", formatValue(stats.input_fps))}
    ${createStatCard("Frames Processed", formatValue(stats.processed_frames))}
    ${createStatCard("Avg Inference Latency", formatValue(stats.avg_inference_latency_ms, " ms/frame"))}
    ${createStatCard("Processing Throughput", formatValue(stats.processing_fps, " FPS"))}
    ${createStatCard("Overall Risk", stats.overall_risk || "N/A", riskClass)}
    ${createStatCard("Avg Risk Score", formatValue(stats.avg_risk_score, "%"))}
    ${createStatCard("Lane Departures", formatValue(stats.lane_departure_events))}
    ${createStatCard("Night Frames", formatValue(stats.night_frames_detected))}
    ${createStatCard("Snow Frames", formatValue(stats.snow_frames_detected))}
  `;

  dashboardMetricsGrid.innerHTML = `
    ${createStatCard("Processed Frames", formatValue(stats.processed_frames))}
    ${createStatCard("Average Latency", formatValue(stats.avg_inference_latency_ms, " ms"))}
    ${createStatCard("Processing FPS", formatValue(stats.processing_fps))}
    ${createStatCard("Overall Risk", stats.overall_risk || "N/A", riskClass)}
    ${createStatCard("Average Risk Score", formatValue(stats.avg_risk_score, "%"))}
    ${createStatCard("Maximum Risk Score", formatValue(stats.max_risk_score, "%"))}
    ${createStatCard("Lane Departure Events", formatValue(stats.lane_departure_events))}
    ${createStatCard("Max Departure", formatValue(stats.max_departure_percent, "%"))}
    ${createStatCard("Safe Frames", formatValue(stats.safe_frames))}
    ${createStatCard("Caution Frames", formatValue(stats.caution_frames))}
    ${createStatCard("Danger Frames", formatValue(stats.danger_frames))}
    ${createStatCard("Road Position Bias", formatValue(stats.road_position_bias_pixels, " px"))}
  `;

  updateFilterBadge(stats.overall_risk);
  updateFilterSummary(stats);
}

function updateAssets(assets) {
  setImage(heatmapImage, heatmapPlaceholder, assets.heatmap);
  setImage(chartRiskTrend, chartRiskTrendPlaceholder, assets.risk_plot);
  setImage(chartOffsetTrend, chartOffsetTrendPlaceholder, assets.offset_plot);
  setImage(chartZoneBar, chartZoneBarPlaceholder, assets.zone_chart);
  setImage(chartDayNight, chartDayNightPlaceholder, assets.day_night_chart);
  setImage(chartRiskPie, chartRiskPiePlaceholder, assets.risk_pie_chart);
  setImage(chartBestFrame, chartBestFramePlaceholder, assets.best_frame);
}

function applyFilter(category) {
  filterButtons.forEach((btn) => {
    btn.classList.toggle("active", btn.dataset.filter === category);
  });

  filterItems.forEach((item) => {
    if (category === "all" || item.dataset.category === category) {
      item.classList.remove("hidden");
    } else {
      item.classList.add("hidden");
    }
  });
}

filterButtons.forEach((btn) => {
  btn.addEventListener("click", () => {
    applyFilter(btn.dataset.filter);
  });
});

videoInput.addEventListener("change", () => {
  const file = videoInput.files[0];
  selectedFileName.textContent = file ? file.name : "No file selected";

  if (file) {
    const fileURL = URL.createObjectURL(file);
    originalVideo.src = fileURL;
    originalVideo.classList.remove("hidden");
    originalPlaceholder.classList.add("hidden");
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
  if (files && files.length > 0) {
    videoInput.files = files;
    const file = files[0];
    selectedFileName.textContent = file.name;

    const fileURL = URL.createObjectURL(file);
    originalVideo.src = fileURL;
    originalVideo.classList.remove("hidden");
    originalPlaceholder.classList.add("hidden");
  }
});

syncPlayBtn.addEventListener("click", () => {
  if (!originalVideo.src || !processedVideo.src) return;

  const targetTime = Math.min(originalVideo.currentTime || 0, processedVideo.currentTime || 0);
  originalVideo.currentTime = targetTime;
  processedVideo.currentTime = targetTime;

  originalVideo.play();
  processedVideo.play();
});

uploadForm.addEventListener("submit", async (e) => {
  e.preventDefault();

  const file = videoInput.files[0];
  if (!file) {
    setStatus("error", "Please select a video before processing.");
    return;
  }

  resetDashboard();
  processedVideo.classList.add("hidden");
  processedPlaceholder.classList.remove("hidden");
  downloadBtn.classList.add("hidden");

  const formData = new FormData();
  formData.append("video", file);

  processBtn.disabled = true;
  processBtn.textContent = "Processing...";
  setStatus("processing", "Processing video and generating risk assessment dashboard...");

  try {
    const response = await fetch("/api/process", {
      method: "POST",
      body: formData
    });

    const data = await response.json();

    if (!response.ok || !data.ok) {
      throw new Error(data.message || "Processing failed.");
    }

    setStatus("success", data.message || "Processing complete.");

    if (data.preview_url) {
      processedVideo.src = `${data.preview_url}?t=${Date.now()}`;
      processedVideo.classList.remove("hidden");
      processedPlaceholder.classList.add("hidden");
    }

    if (data.download_url) {
      downloadBtn.href = data.download_url;
      downloadBtn.classList.remove("hidden");
    }

    if (data.stats) {
      updateStats(data.stats);
    }

    if (data.assets) {
      updateAssets(data.assets);
    }

    applyFilter("all");
    document.getElementById("risk-dashboard").scrollIntoView({ behavior: "smooth" });
  } catch (error) {
    console.error(error);
    setStatus("error", error.message || "Something went wrong while processing the video.");
  } finally {
    processBtn.disabled = false;
    processBtn.textContent = "Process Video";
  }
});

resetDashboard();
applyFilter("all");
