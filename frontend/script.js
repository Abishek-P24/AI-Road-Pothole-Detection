/**
 * AI Road Pothole Detection - Frontend Application Controller
 * Handles backend REST API communication, file uploads, sample presets,
 * detection rendering, table population, and pipeline stage visualization.
 */

// Application State
const state = {
  currentFile: null,
  fileType: null, // 'image' or 'video'
  detectionResult: null,
  activeView: 'annotated', // 'annotated', 'original', 'comparison'
  activePipelineStage: 'original',
  pipelineImages: {}
};

// DOM Elements
const DOM = {
  backendStatusBadge: document.getElementById('backendStatusBadge'),
  backendStatusText: document.getElementById('backendStatusText'),
  dropZone: document.getElementById('dropZone'),
  fileInput: document.getElementById('fileInput'),
  browseBtn: document.getElementById('browseBtn'),
  fileMetaCard: document.getElementById('fileMetaCard'),
  fileName: document.getElementById('fileName'),
  fileSize: document.getElementById('fileSize'),
  fileTypeIcon: document.getElementById('fileTypeIcon'),
  clearFileBtn: document.getElementById('clearFileBtn'),
  detectBtn: document.getElementById('detectBtn'),
  resetBtn: document.getElementById('resetBtn'),
  downloadBtn: document.getElementById('downloadBtn'),
  processingCard: document.getElementById('processingCard'),
  processStatusTitle: document.getElementById('processStatusTitle'),
  processStatusDesc: document.getElementById('processStatusDesc'),
  
  // Tuning Sliders
  minAreaRange: document.getElementById('minAreaRange'),
  minAreaVal: document.getElementById('minAreaVal'),
  confThreshRange: document.getElementById('confThreshRange'),
  confThreshVal: document.getElementById('confThreshVal'),

  // Metrics
  metricCount: document.getElementById('metricCount'),
  metricCondition: document.getElementById('metricCondition'),
  metricConditionDesc: document.getElementById('metricConditionDesc'),
  metricSeverityHint: document.getElementById('metricSeverityHint'),
  metricConfidence: document.getElementById('metricConfidence'),
  metricTime: document.getElementById('metricTime'),

  // Media Viewer
  viewerPlaceholder: document.getElementById('viewerPlaceholder'),
  resultImage: document.getElementById('resultImage'),
  resultVideo: document.getElementById('resultVideo'),
  comparisonView: document.getElementById('comparisonView'),
  compOriginalImg: document.getElementById('compOriginalImg'),
  compAnnotatedImg: document.getElementById('compAnnotatedImg'),
  viewAnnotatedBtn: document.getElementById('viewAnnotatedBtn'),
  viewOriginalBtn: document.getElementById('viewOriginalBtn'),
  viewComparisonBtn: document.getElementById('viewComparisonBtn'),

  // Table
  detectionsTbody: document.getElementById('detectionsTbody'),
  tableCountBadge: document.getElementById('tableCountBadge'),

  // Pipeline Stepper
  stageTitle: document.getElementById('stageTitle'),
  stageDesc: document.getElementById('stageDesc'),
  stageImg: document.getElementById('stageImg'),
  stagePlaceholder: document.getElementById('stagePlaceholder'),
  stagePills: document.querySelectorAll('.stage-pill'),
  sampleBtns: document.querySelectorAll('.sample-btn')
};

// Educational descriptions for each computer vision pipeline stage
const STAGE_DESCRIPTIONS = {
  original: {
    title: 'Stage 1: Original Standardized Input',
    desc: 'Image standardized to 800px width while preserving native aspect ratio for uniform feature scale.'
  },
  grayscale: {
    title: 'Stage 2: Grayscale Conversion',
    desc: 'Converts 3-channel BGR color image to 1-channel luminance intensity map using cv2.cvtColor.'
  },
  blur: {
    title: 'Stage 3: Gaussian Blur & Noise Reduction',
    desc: 'Applies Gaussian low-pass filter (7x7 kernel) to suppress high-frequency asphalt aggregate noise.'
  },
  threshold: {
    title: 'Stage 4: Morphological Black-Hat & Statistical Thresholding',
    desc: 'Extracts localized sunken depressions darker than surrounding road surface; cleans speckles with opening/closing.'
  },
  edges: {
    title: 'Stage 5: Canny Edge Detection',
    desc: 'Calculates high-gradient rim transitions around fractured asphalt boundaries using multi-stage hysteresis.'
  },
  contours: {
    title: 'Stage 6: Contour Candidate Extraction',
    desc: 'Extracts boundary contours using cv2.findContours and evaluates geometry, solidity, and darkness contrast.'
  },
  annotated: {
    title: 'Stage 7: Validated Potholes & HUD Annotation',
    desc: 'Draws color-coded bounding boxes, confidence tags, severity rating, and road condition HUD banner.'
  }
};

// Initialize Application
document.addEventListener('DOMContentLoaded', () => {
  checkBackendHealth();
  setupEventListeners();
  setupTuningSliders();
});

// Check API Health
async function checkBackendHealth() {
  try {
    const res = await fetch('/health');
    if (res.ok) {
      const data = await res.json();
      DOM.backendStatusBadge.className = 'badge-status-pill badge-online';
      DOM.backendStatusText.textContent = `Online • ${data.backend}`;
    } else {
      throw new Error();
    }
  } catch (err) {
    DOM.backendStatusBadge.className = 'badge-status-pill badge-offline';
    DOM.backendStatusText.textContent = 'Backend Offline';
  }
}

// Setup Event Listeners
function setupEventListeners() {
  // File Browse
  DOM.browseBtn.addEventListener('click', () => DOM.fileInput.click());
  DOM.dropZone.addEventListener('click', (e) => {
    if (e.target !== DOM.browseBtn) DOM.fileInput.click();
  });

  DOM.fileInput.addEventListener('change', (e) => {
    if (e.target.files && e.target.files[0]) {
      handleFileSelection(e.target.files[0]);
    }
  });

  // Drag and Drop
  ['dragenter', 'dragover'].forEach(eventName => {
    DOM.dropZone.addEventListener(eventName, (e) => {
      e.preventDefault();
      DOM.dropZone.classList.add('dragover');
    });
  });

  ['dragleave', 'drop'].forEach(eventName => {
    DOM.dropZone.addEventListener(eventName, (e) => {
      e.preventDefault();
      DOM.dropZone.classList.remove('dragover');
    });
  });

  DOM.dropZone.addEventListener('drop', (e) => {
    if (e.dataTransfer.files && e.dataTransfer.files[0]) {
      handleFileSelection(e.dataTransfer.files[0]);
    }
  });

  // Clear File
  DOM.clearFileBtn.addEventListener('click', resetAll);
  DOM.resetBtn.addEventListener('click', resetAll);

  // Detect Button
  DOM.detectBtn.addEventListener('click', runDetection);

  // View Mode Toggles
  DOM.viewAnnotatedBtn.addEventListener('click', () => switchMediaView('annotated'));
  DOM.viewOriginalBtn.addEventListener('click', () => switchMediaView('original'));
  DOM.viewComparisonBtn.addEventListener('click', () => switchMediaView('comparison'));

  // Quick Test Sample Presets
  DOM.sampleBtns.forEach(btn => {
    btn.addEventListener('click', async () => {
      const sampleUrl = btn.getAttribute('data-sample');
      const sampleType = btn.getAttribute('data-type');
      const sampleName = btn.getAttribute('data-name');
      await loadSamplePreset(sampleUrl, sampleType, sampleName);
    });
  });

  // Pipeline Stepper Tabs
  DOM.stagePills.forEach(pill => {
    pill.addEventListener('click', () => {
      const stage = pill.getAttribute('data-stage');
      switchPipelineStage(stage);
    });
  });

  // Download Button
  DOM.downloadBtn.addEventListener('click', downloadCurrentResult);
}

// Setup Parameter Tuning Sliders
function setupTuningSliders() {
  DOM.minAreaRange.addEventListener('input', (e) => {
    DOM.minAreaVal.textContent = e.target.value;
  });

  DOM.confThreshRange.addEventListener('input', (e) => {
    DOM.confThreshVal.textContent = `${e.target.value}%`;
  });
}

// Handle File Selection
function handleFileSelection(file) {
  state.currentFile = file;
  const isVideo = file.type.startsWith('video/') || /\.(mp4|avi|mov|mkv)$/i.test(file.name);
  state.fileType = isVideo ? 'video' : 'image';

  // Update Metadata Card
  DOM.fileName.textContent = file.name;
  DOM.fileSize.textContent = formatBytes(file.size);
  DOM.fileTypeIcon.className = isVideo ? 'fa-solid fa-file-video text-info' : 'fa-solid fa-file-image text-info';
  DOM.fileMetaCard.classList.remove('d-none');
  DOM.detectBtn.disabled = false;

  // Show preview
  const objectUrl = URL.createObjectURL(file);
  showInitialPreview(objectUrl, isVideo);
}

// Show Initial Media Preview
function showInitialPreview(url, isVideo) {
  DOM.viewerPlaceholder.classList.add('d-none');
  DOM.comparisonView.classList.add('d-none');

  if (isVideo) {
    DOM.resultImage.classList.add('d-none');
    DOM.resultVideo.src = url;
    DOM.resultVideo.classList.remove('d-none');
    DOM.resultVideo.load();
  } else {
    DOM.resultVideo.classList.add('d-none');
    DOM.resultImage.src = url;
    DOM.resultImage.classList.remove('d-none');
  }
}

// Load Preset Sample File
async function loadSamplePreset(url, type, name) {
  try {
    const res = await fetch(url);
    if (!res.ok) throw new Error('Sample not found');
    const blob = await res.blob();
    const filename = url.split('/').pop();
    const file = new File([blob], filename, { type: type === 'video' ? 'video/mp4' : 'image/jpeg' });
    handleFileSelection(file);
  } catch (err) {
    alert(`Could not load preset sample: ${err.message}`);
  }
}

// Run Detection via REST API
async function runDetection() {
  if (!state.currentFile) return;

  const isVideo = state.fileType === 'video';
  const endpoint = isVideo ? '/detect/video' : '/detect/image';

  // Prepare FormData
  const formData = new FormData();
  formData.append('file', state.currentFile);
  formData.append('min_area', DOM.minAreaRange.value);
  formData.append('confidence_threshold', DOM.confThreshRange.value);

  // UI Processing State
  DOM.detectBtn.disabled = true;
  DOM.processingCard.classList.remove('d-none');
  DOM.processStatusTitle.textContent = isVideo ? 'Processing Road Video Frames...' : 'Executing Computer Vision Pipeline...';
  DOM.processStatusDesc.textContent = isVideo
    ? 'Extracting frames → Preprocessing → Contour analysis → Video writer'
    : 'Grayscale → Blur → Morphological Black-Hat → Canny → Contour Filtering';

  try {
    const response = await fetch(endpoint, {
      method: 'POST',
      body: formData
    });

    if (!response.ok) {
      const errData = await response.json();
      throw new Error(errData.detail || 'Detection failed');
    }

    const data = await response.json();
    state.detectionResult = data;

    if (isVideo) {
      displayVideoResults(data);
    } else {
      displayImageResults(data);
    }

    DOM.downloadBtn.disabled = false;
  } catch (error) {
    alert(`Detection Error: ${error.message}`);
  } finally {
    DOM.detectBtn.disabled = false;
    DOM.processingCard.classList.add('d-none');
  }
}

// Display Image Detection Results
function displayImageResults(data) {
  // Update Metrics
  DOM.metricCount.textContent = data.pothole_count;
  DOM.metricSeverityHint.textContent = data.pothole_count === 0 ? 'Clean Roadway' : `${data.pothole_count} candidate(s)`;

  DOM.metricCondition.textContent = data.road_condition;
  DOM.metricCondition.className = `metric-value ${data.road_condition_info.badge_class}`;
  DOM.metricConditionDesc.textContent = data.road_condition_info.description;

  DOM.metricConfidence.textContent = data.pothole_count > 0 ? `${data.average_confidence}%` : '--%';
  DOM.metricTime.textContent = `${data.processing_time_ms} ms`;

  // Update Media Viewer
  DOM.resultImage.src = data.annotated_image_url;
  DOM.compOriginalImg.src = data.original_image_url;
  DOM.compAnnotatedImg.src = data.annotated_image_url;
  switchMediaView('annotated');

  // Populate Detections Breakdown Table
  populateDetectionsTable(data.detections);

  // Store pipeline image stages
  state.pipelineImages = {
    original: data.original_image_url,
    grayscale: data.pipeline_stages.grayscale,
    blur: data.pipeline_stages.blur,
    threshold: data.pipeline_stages.threshold,
    edges: data.pipeline_stages.edges,
    contours: data.pipeline_stages.contours,
    annotated: data.annotated_image_url
  };

  switchPipelineStage('annotated');
}

// Display Video Detection Results
function displayVideoResults(data) {
  DOM.metricCount.textContent = data.max_potholes_detected;
  DOM.metricSeverityHint.textContent = `Max in single frame (Avg: ${data.average_potholes_per_frame})`;

  DOM.metricCondition.textContent = data.road_condition;
  DOM.metricCondition.className = `metric-value ${data.road_condition_info.badge_class}`;
  DOM.metricConditionDesc.textContent = `${data.processed_frames} frames analyzed @ ${data.fps} FPS`;

  DOM.metricConfidence.textContent = `${data.average_confidence}%`;
  DOM.metricTime.textContent = `${data.processing_time_sec} s`;

  DOM.resultImage.classList.add('d-none');
  DOM.comparisonView.classList.add('d-none');
  DOM.resultVideo.src = data.video_result_url;
  DOM.resultVideo.classList.remove('d-none');
  DOM.resultVideo.load();
  DOM.resultVideo.play().catch(() => {});

  // Table summary for video
  DOM.detectionsTbody.innerHTML = `
    <tr>
      <td class="ps-3 font-mono">1</td>
      <td class="font-mono">Full Video Stream (${data.total_frames} frames)</td>
      <td>Multi-frame</td>
      <td>16:9 / 4:3</td>
      <td><span class="badge ${data.max_potholes_detected >= 3 ? 'bg-danger' : (data.max_potholes_detected > 0 ? 'bg-warning text-dark' : 'bg-success')}">${data.road_condition}</span></td>
      <td class="pe-3"><span class="font-mono fw-bold">${data.average_confidence}%</span></td>
    </tr>
  `;
  DOM.tableCountBadge.textContent = `${data.max_potholes_detected} peak potholes`;
}

// Populate Table with Detections
function populateDetectionsTable(detections) {
  DOM.tableCountBadge.textContent = `${detections.length} items`;

  if (!detections || detections.length === 0) {
    DOM.detectionsTbody.innerHTML = `
      <tr>
        <td colspan="6" class="text-center py-4 text-secondary small">
          <i class="fa-solid fa-circle-check text-success me-2"></i>No potholes detected. Road surface verified clean.
        </td>
      </tr>
    `;
    return;
  }

  let html = '';
  detections.forEach(d => {
    const sevBadge = d.severity === 'HIGH' ? 'bg-danger' : (d.severity === 'MEDIUM' ? 'bg-warning text-dark' : 'bg-info text-dark');
    const bboxStr = `X: ${d.bbox.x}, Y: ${d.bbox.y}, W: ${d.bbox.width}, H: ${d.bbox.height}`;

    html += `
      <tr>
        <td class="ps-3 font-mono fw-bold text-info">#${d.id}</td>
        <td class="font-mono small text-secondary">${bboxStr}</td>
        <td class="font-mono small">${d.area.toLocaleString()} px²</td>
        <td class="font-mono small">${d.aspect_ratio}</td>
        <td><span class="badge ${sevBadge} font-mono">${d.severity}</span></td>
        <td class="pe-3">
          <div class="d-flex align-items-center gap-2">
            <span class="font-mono small fw-bold text-white">${d.confidence}%</span>
            <div class="progress flex-grow-1" style="height: 6px; background-color: rgba(255,255,255,0.1);">
              <div class="progress-bar bg-info" style="width: ${d.confidence}%"></div>
            </div>
          </div>
        </td>
      </tr>
    `;
  });

  DOM.detectionsTbody.innerHTML = html;
}

// Switch Media View (Annotated / Original / Side-by-Side)
function switchMediaView(viewMode) {
  state.activeView = viewMode;
  [DOM.viewAnnotatedBtn, DOM.viewOriginalBtn, DOM.viewComparisonBtn].forEach(btn => btn.classList.remove('active'));

  if (state.fileType === 'video') return;

  if (viewMode === 'annotated') {
    DOM.viewAnnotatedBtn.classList.add('active');
    DOM.resultImage.src = state.detectionResult ? state.detectionResult.annotated_image_url : DOM.resultImage.src;
    DOM.resultImage.classList.remove('d-none');
    DOM.comparisonView.classList.add('d-none');
  } else if (viewMode === 'original') {
    DOM.viewOriginalBtn.classList.add('active');
    DOM.resultImage.src = state.detectionResult ? state.detectionResult.original_image_url : DOM.resultImage.src;
    DOM.resultImage.classList.remove('d-none');
    DOM.comparisonView.classList.add('d-none');
  } else if (viewMode === 'comparison') {
    DOM.viewComparisonBtn.classList.add('active');
    DOM.resultImage.classList.add('d-none');
    DOM.comparisonView.classList.remove('d-none');
  }
}

// Switch Pipeline Stage Viewer
function switchPipelineStage(stage) {
  state.activePipelineStage = stage;
  DOM.stagePills.forEach(pill => {
    pill.classList.toggle('active', pill.getAttribute('data-stage') === stage);
  });

  const info = STAGE_DESCRIPTIONS[stage] || STAGE_DESCRIPTIONS.original;
  DOM.stageTitle.textContent = info.title;
  DOM.stageDesc.textContent = info.desc;

  if (state.pipelineImages && state.pipelineImages[stage]) {
    DOM.stageImg.src = state.pipelineImages[stage];
    DOM.stageImg.style.display = 'inline-block';
    DOM.stagePlaceholder.style.display = 'none';
  } else {
    DOM.stageImg.style.display = 'none';
    DOM.stagePlaceholder.style.display = 'block';
  }
}

// Reset Entire State
function resetAll() {
  state.currentFile = null;
  state.fileType = null;
  state.detectionResult = null;
  state.pipelineImages = {};

  DOM.fileInput.value = '';
  DOM.fileMetaCard.classList.add('d-none');
  DOM.detectBtn.disabled = true;
  DOM.downloadBtn.disabled = true;

  DOM.minAreaRange.value = 800;
  DOM.minAreaVal.textContent = '800';
  DOM.confThreshRange.value = 50;
  DOM.confThreshVal.textContent = '50%';

  DOM.metricCount.textContent = '--';
  DOM.metricSeverityHint.textContent = 'Ready for analysis';
  DOM.metricCondition.textContent = '--';
  DOM.metricCondition.className = 'metric-value';
  DOM.metricConditionDesc.textContent = 'Awaiting input';
  DOM.metricConfidence.textContent = '--%';
  DOM.metricTime.textContent = '-- ms';

  DOM.viewerPlaceholder.classList.remove('d-none');
  DOM.resultImage.classList.add('d-none');
  DOM.resultVideo.classList.add('d-none');
  DOM.comparisonView.classList.add('d-none');

  DOM.detectionsTbody.innerHTML = `
    <tr>
      <td colspan="6" class="text-center py-4 text-secondary small">
        No potholes detected yet. Run detection to see candidate breakdown.
      </td>
    </tr>
  `;
  DOM.tableCountBadge.textContent = '0 items';

  DOM.stageImg.style.display = 'none';
  DOM.stagePlaceholder.style.display = 'block';
  switchPipelineStage('original');
}

// Download Processed Result
function downloadCurrentResult() {
  if (!state.detectionResult) return;

  const url = state.fileType === 'video'
    ? state.detectionResult.video_result_url
    : state.detectionResult.annotated_image_url;

  const a = document.createElement('a');
  a.href = url;
  a.download = `pothole_detection_${Date.now()}.${state.fileType === 'video' ? 'mp4' : 'jpg'}`;
  document.body.appendChild(a);
  a.click();
  document.body.removeChild(a);
}

// Utility: Format File Size
function formatBytes(bytes) {
  if (bytes === 0) return '0 Bytes';
  const k = 1024;
  const sizes = ['Bytes', 'KB', 'MB', 'GB'];
  const i = Math.floor(Math.log(bytes) / Math.log(k));
  return parseFloat((bytes / Math.pow(k, i)).toFixed(2)) + ' ' + sizes[i];
}
