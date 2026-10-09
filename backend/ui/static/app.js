// Universal ML Engine — Frontend Controller
let currentUploadId = null;
let currentRunId = null;
let currentResults = null;
let pollingInterval = null;
let timerInterval = null;
let runStartTime = null;

document.addEventListener("DOMContentLoaded", () => {
    initDropZone();
    initBenchmarkSelector();
    initTargetSelectListener();
    fetchRunsList();
});

// -----------------------------------------------------------------------------
// Drop Zone & File Upload
// -----------------------------------------------------------------------------
function initDropZone() {
    const dropZone = document.getElementById("dropZone");
    const fileInput = document.getElementById("fileInput");

    dropZone.addEventListener("click", () => fileInput.click());

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
        if (e.dataTransfer.files.length > 0) {
            handleFileUpload(e.dataTransfer.files[0]);
        }
    });

    fileInput.addEventListener("change", (e) => {
        if (e.target.files.length > 0) {
            handleFileUpload(e.target.files[0]);
        }
    });
}

async function handleFileUpload(file) {
    closeErrorBanner();
    const formData = new FormData();
    formData.append("file", file);

    const dropText = document.querySelector(".drop-text");
    const originalText = dropText.innerHTML;
    dropText.innerHTML = "Uploading and profiling dataset...";

    try {
        const resp = await fetch("/api/upload", {
            method: "POST",
            body: formData,
        });

        const data = await resp.json();
        if (!resp.ok) {
            throw new Error(data.detail || "Failed to upload dataset.");
        }

        renderInspection(data);
    } catch (err) {
        showError("Dataset Upload Error", err.message);
    } finally {
        dropText.innerHTML = originalText;
    }
}

async function loadSampleDataset(filename) {
    closeErrorBanner();
    try {
        const resp = await fetch("/api/load-sample", {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({ filename }),
        });

        const data = await resp.json();
        if (!resp.ok) {
            throw new Error(data.detail || "Failed to load sample dataset.");
        }

        renderInspection(data);
    } catch (err) {
        showError("Sample Load Error", err.message);
    }
}

// -----------------------------------------------------------------------------
// Inspection Rendering
// -----------------------------------------------------------------------------
function renderInspection(data) {
    currentUploadId = data.upload_id;

    document.getElementById("statFilename").textContent = data.filename;
    document.getElementById("statRows").textContent = Number(data.row_count).toLocaleString();
    document.getElementById("statCols").textContent = data.column_count;
    document.getElementById("statMissing").textContent = Number(data.missing_cells).toLocaleString();
    document.getElementById("statMissingPct").textContent = `${data.missing_percentage}%`;

    // Render Columns Table
    document.getElementById("colCountBadge").textContent = `${data.columns.length} columns`;
    const colTbody = document.getElementById("columnsTableBody");
    colTbody.innerHTML = "";
    data.columns.forEach(col => {
        const tr = document.createElement("tr");
        tr.innerHTML = `
            <td class="text-mono"><strong>${escapeHtml(col.name)}</strong></td>
            <td><span class="badge">${escapeHtml(col.dtype)}</span></td>
            <td>${col.missing_count}</td>
            <td>${col.missing_pct}%</td>
            <td>${col.unique_count}</td>
        `;
        colTbody.appendChild(tr);
    });

    // Render Preview Table
    const previewHead = document.getElementById("previewTableHead");
    const previewBody = document.getElementById("previewTableBody");
    previewHead.innerHTML = "";
    previewBody.innerHTML = "";

    if (data.preview_rows && data.preview_rows.length > 0) {
        const cols = Object.keys(data.preview_rows[0]);
        const trH = document.createElement("tr");
        cols.forEach(c => {
            const th = document.createElement("th");
            th.textContent = c;
            trH.appendChild(th);
        });
        previewHead.appendChild(trH);

        data.preview_rows.forEach(row => {
            const tr = document.createElement("tr");
            cols.forEach(c => {
                const td = document.createElement("td");
                td.textContent = String(row[c] !== null && row[c] !== undefined ? row[c] : "");
                tr.appendChild(td);
            });
            previewBody.appendChild(tr);
        });
    }

    document.getElementById("inspectionBox").classList.remove("hidden");
    document.getElementById("targetSection").classList.remove("hidden");

    // Populate Target Selector
    const targetSelect = document.getElementById("targetSelect");
    targetSelect.innerHTML = `<option value="">-- Choose Target Column --</option>`;
    data.columns.forEach(col => {
        const opt = document.createElement("option");
        opt.value = col.name;
        opt.textContent = col.name;
        if (data.suggested_target && col.name === data.suggested_target) {
            opt.selected = true;
        }
        targetSelect.appendChild(opt);
    });

    // Trigger problem detection if suggested target exists
    if (data.suggested_target) {
        if (data.detected_problem) {
            renderProblemDetection(data.suggested_target, data.detected_problem);
        } else {
            detectTargetProblem(data.suggested_target);
        }
    }
}

// -----------------------------------------------------------------------------
// Target & Problem Detection
// -----------------------------------------------------------------------------
function initTargetSelectListener() {
    const targetSelect = document.getElementById("targetSelect");
    targetSelect.addEventListener("change", (e) => {
        const val = e.target.value;
        if (val && currentUploadId) {
            detectTargetProblem(val);
        } else {
            document.getElementById("problemCard").classList.add("hidden");
        }
    });
}

async function detectTargetProblem(targetColumn) {
    if (!currentUploadId || !targetColumn) return;

    try {
        const resp = await fetch("/api/detect-target", {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({
                upload_id: currentUploadId,
                target_column: targetColumn,
            }),
        });

        const data = await resp.json();
        if (!resp.ok) {
            throw new Error(data.detail || "Problem detection failed.");
        }

        renderProblemDetection(targetColumn, data);
    } catch (err) {
        showError("Target Detection Error", err.message);
    }
}

function renderProblemDetection(targetColumn, pd) {
    const card = document.getElementById("problemCard");
    const badge = document.getElementById("problemTypeBadge");
    const conf = document.getElementById("problemConfidence");
    const reason = document.getElementById("problemReason");
    const distBox = document.getElementById("classDistBox");
    const chips = document.getElementById("classChips");

    card.classList.remove("hidden");
    const isClf = pd.problem_type.includes("classification");
    badge.textContent = pd.problem_type.replace("_", " ").toUpperCase();
    badge.className = `badge badge-lg ${isClf ? "badge-task" : "badge-indigo"}`;

    conf.textContent = `${(pd.confidence * 100).toFixed(0)}% (${pd.confidence.toFixed(2)})`;
    reason.textContent = pd.reason;

    if (pd.class_distribution && Object.keys(pd.class_distribution).length > 0) {
        distBox.classList.remove("hidden");
        chips.innerHTML = "";
        for (const [cls, count] of Object.entries(pd.class_distribution)) {
            const chip = document.createElement("span");
            chip.className = "class-chip";
            chip.textContent = `${cls}: ${count}`;
            chips.appendChild(chip);
        }
    } else {
        distBox.classList.add("hidden");
    }
}

// -----------------------------------------------------------------------------
// Engine Execution & Polling
// -----------------------------------------------------------------------------
async function startEngineRun() {
    closeErrorBanner();
    const targetSelect = document.getElementById("targetSelect");
    const targetColumn = targetSelect.value;
    const timeLimit = parseFloat(document.getElementById("timeLimitSelect").value || "30");

    if (!currentUploadId || !targetColumn) {
        showError("Configuration Missing", "Please select a target column before running the engine.");
        return;
    }

    const btn = document.getElementById("btnRunEngine");
    btn.disabled = true;
    btn.innerHTML = `<span>⏳ Initializing...</span>`;

    // Show Progress Section
    document.getElementById("progressSection").classList.remove("hidden");
    document.getElementById("resultsWrapper").classList.add("hidden");
    resetProgress();

    runStartTime = Date.now();
    startTimer();

    try {
        const resp = await fetch("/api/run", {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({
                upload_id: currentUploadId,
                target_column: targetColumn,
                time_limit_seconds: timeLimit,
            }),
        });

        const data = await resp.json();
        if (!resp.ok) {
            throw new Error(data.detail || "Failed to start engine.");
        }

        currentRunId = data.run_id;
        startPolling(currentRunId);
    } catch (err) {
        stopTimer();
        btn.disabled = false;
        btn.innerHTML = `<span>🚀 Run Universal ML Engine</span>`;
        document.getElementById("progressSection").classList.add("hidden");
        showError("Execution Error", err.message);
    }
}

function startTimer() {
    const timerElem = document.getElementById("elapsedTimer");
    timerInterval = setInterval(() => {
        const elapsed = ((Date.now() - runStartTime) / 1000).toFixed(1);
        timerElem.textContent = `${elapsed}s`;
    }, 100);
}

function stopTimer() {
    if (timerInterval) {
        clearInterval(timerInterval);
        timerInterval = null;
    }
}

function resetProgress() {
    for (let i = 1; i <= 8; i++) {
        const step = document.getElementById(`pstep-${i}`);
        if (step) {
            step.className = "step-item";
        }
    }
    document.getElementById("pstep-1").className = "step-item active";
    document.getElementById("progressBarInner").style.width = "10%";
}

function updateProgressStage(seconds) {
    // Dynamic progressive visual update
    const p1 = document.getElementById("pstep-1");
    const p2 = document.getElementById("pstep-2");
    const p3 = document.getElementById("pstep-3");
    const p4 = document.getElementById("pstep-4");
    const p5 = document.getElementById("pstep-5");
    const p6 = document.getElementById("pstep-6");
    const p7 = document.getElementById("pstep-7");
    const p8 = document.getElementById("pstep-8");
    const bar = document.getElementById("progressBarInner");

    if (seconds < 2) {
        p1.className = "step-item active";
        bar.style.width = "15%";
    } else if (seconds < 4) {
        p1.className = "step-item completed";
        p2.className = "step-item active";
        bar.style.width = "25%";
    } else if (seconds < 7) {
        p2.className = "step-item completed";
        p3.className = "step-item active";
        bar.style.width = "40%";
    } else if (seconds < 25) {
        p3.className = "step-item completed";
        p4.className = "step-item active";
        bar.style.width = "65%";
    } else if (seconds < 30) {
        p4.className = "step-item completed";
        p5.className = "step-item active";
        bar.style.width = "80%";
    } else {
        p5.className = "step-item completed";
        p6.className = "step-item active";
        p7.className = "step-item active";
        bar.style.width = "90%";
    }
}

function startPolling(runId) {
    if (pollingInterval) clearInterval(pollingInterval);

    pollingInterval = setInterval(async () => {
        try {
            const resp = await fetch(`/api/status/${runId}`);
            if (!resp.ok) return;

            const data = await resp.json();
            updateProgressStage(data.duration_seconds);
            document.getElementById("progressStatusText").textContent = data.stage || "Executing...";

            if (data.status === "completed") {
                clearInterval(pollingInterval);
                pollingInterval = null;
                stopTimer();
                markAllStepsCompleted();
                await fetchAndRenderResults(runId);
            } else if (data.status === "failed") {
                clearInterval(pollingInterval);
                pollingInterval = null;
                stopTimer();
                document.getElementById("progressSection").classList.add("hidden");
                const btn = document.getElementById("btnRunEngine");
                btn.disabled = false;
                btn.innerHTML = `<span>🚀 Run Universal ML Engine</span>`;
                showError("Engine Pipeline Failed", data.error_message || "Unknown execution error.");
            }
        } catch (e) {
            console.error("Polling error:", e);
        }
    }, 1000);
}

function markAllStepsCompleted() {
    for (let i = 1; i <= 8; i++) {
        const step = document.getElementById(`pstep-${i}`);
        if (step) step.className = "step-item completed";
    }
    document.getElementById("progressBarInner").style.width = "100%";
}

// -----------------------------------------------------------------------------
// Results Rendering (Champion, Validation vs Final Test, Visuals, Reports, Inference)
// -----------------------------------------------------------------------------
async function fetchAndRenderResults(runId) {
    try {
        const resp = await fetch(`/api/results/${runId}`);
        const res = await resp.json();
        if (!resp.ok) {
            throw new Error(res.detail || "Failed to load results.");
        }

        renderResults(res);
    } catch (err) {
        showError("Results Load Error", err.message);
    } finally {
        const btn = document.getElementById("btnRunEngine");
        btn.disabled = false;
        btn.innerHTML = `<span>🚀 Run Universal ML Engine</span>`;
    }
}

function renderResults(res) {
    currentResults = res;

    // Summary Card
    document.getElementById("resDatasetLabel").textContent = `Dataset: ${res.dataset_name} • Target Column: ${res.target_column}`;
    const taskBadge = document.getElementById("resTaskBadge");
    taskBadge.textContent = res.problem_type.replace("_", " ").toUpperCase();
    const isClf = res.problem_type.includes("classification");
    taskBadge.className = `badge badge-task ${isClf ? "" : "badge-indigo"}`;

    document.getElementById("resBackendBadge").textContent = `Backend: ${res.champion.backend_name.toUpperCase()}`;
    document.getElementById("resChampionName").textContent = res.champion.model_name;
    document.getElementById("resChampionBackend").textContent = res.champion.backend_name;
    document.getElementById("resValScore").textContent = `${res.champion.primary_metric.toUpperCase()}: ${formatNumber(res.champion.primary_val_score)}`;

    const p = res.partition_info;
    document.getElementById("resPartitionsText").textContent = `Train: ${p.train_rows} | Val: ${p.val_rows} | Test: ${p.test_rows} (Holdout)`;
    document.getElementById("resRationaleText").textContent = res.champion.selection_rationale;

    // Validation Metrics Panel (Model Selection)
    const valGrid = document.getElementById("valMetricsGrid");
    valGrid.innerHTML = "";
    for (const [mName, mVal] of Object.entries(res.champion.val_metrics || {})) {
        const chip = document.createElement("div");
        const isPrimary = (mName.toLowerCase() === res.champion.primary_metric.toLowerCase());
        chip.className = `metric-chip ${isPrimary ? "primary" : ""}`;
        chip.innerHTML = `
            <span class="chip-name">${escapeHtml(mName)} ${isPrimary ? "(Primary)" : ""}</span>
            <span class="chip-value">${formatNumber(mVal)}</span>
        `;
        valGrid.appendChild(chip);
    }

    // Final Test Metrics Panel (Untouched Holdout)
    const testGrid = document.getElementById("testMetricsGrid");
    testGrid.innerHTML = "";
    for (const [mName, mVal] of Object.entries(res.final_test.test_metrics || {})) {
        const chip = document.createElement("div");
        chip.className = "metric-chip";
        chip.innerHTML = `
            <span class="chip-name">${escapeHtml(mName)}</span>
            <span class="chip-value text-success">${formatNumber(mVal)}</span>
        `;
        testGrid.appendChild(chip);
    }

    if (res.final_test.test_sha256) {
        document.getElementById("testIsolationShaText").textContent =
            `Test SHA-256: ${res.final_test.test_sha256.substring(0, 16)}... • Strict Zero-Leakage`;
    }

    // All Candidates Comparison Table
    const candTbody = document.getElementById("candidatesTableBody");
    candTbody.innerHTML = "";
    if (res.champion.all_candidates && res.champion.all_candidates.length > 0) {
        document.getElementById("candidateCountBadge").textContent = `${res.champion.all_candidates.length} Models`;
        res.champion.all_candidates.forEach(cand => {
            const tr = document.createElement("tr");
            const isChamp = (cand.candidate_id === res.champion.candidate_id || cand.model_name === res.champion.model_name);
            tr.innerHTML = `
                <td class="text-mono">${escapeHtml(cand.candidate_id || cand.model_name)} ${isChamp ? '<span class="pill pill-success" style="margin-left:6px;">Champion</span>' : ''}</td>
                <td><span class="badge">${escapeHtml(cand.backend_name)}</span></td>
                <td><strong>${escapeHtml(cand.primary_metric || res.champion.primary_metric)}</strong></td>
                <td class="text-mono" style="font-weight:700;">${formatNumber(cand.primary_val_score)}</td>
                <td>${cand.fit_time_seconds ? cand.fit_time_seconds.toFixed(2) + 's' : '-'}</td>
                <td><span class="badge badge-success">Evaluated</span></td>
            `;
            candTbody.appendChild(tr);
        });
    }

    // Diagnostic Visuals Grid
    const plotsGrid = document.getElementById("plotsGrid");
    plotsGrid.innerHTML = "";
    if (res.plots && Object.keys(res.plots).length > 0) {
        for (const [pName, pUrl] of Object.entries(res.plots)) {
            const card = document.createElement("div");
            card.className = "plot-card";
            card.onclick = () => openPlotModal(formatPlotTitle(pName), pUrl);

            card.innerHTML = `
                <div class="plot-card-header">
                    <span class="plot-title">${escapeHtml(formatPlotTitle(pName))}</span>
                    <span class="badge">🔍 Enlarge</span>
                </div>
                <img src="${pUrl}" alt="${pName}" class="plot-img" loading="lazy">
            `;
            plotsGrid.appendChild(card);
        }
    }

    // Feature Importance Table
    const impTbody = document.getElementById("importanceTableBody");
    impTbody.innerHTML = "";
    if (res.importance_entries && res.importance_entries.length > 0) {
        res.importance_entries.forEach(entry => {
            const tr = document.createElement("tr");
            tr.innerHTML = `
                <td><strong>#${entry.rank}</strong></td>
                <td class="text-mono">${escapeHtml(entry.feature_name)}</td>
                <td class="text-mono">${entry.importance_score >= 0 ? '+' : ''}${formatNumber(entry.importance_score)}</td>
                <td>
                    <div style="display:flex; align-items:center; gap:8px;">
                        <div style="flex:1; height:6px; background:#e2e8f0; border-radius:4px; overflow:hidden;">
                            <div style="width:${Math.min(100, Math.max(0, entry.relative_importance_pct))}%; height:100%; background:#2563eb;"></div>
                        </div>
                        <span style="font-size:0.75rem; font-family:'JetBrains Mono';">${entry.relative_importance_pct.toFixed(1)}%</span>
                    </div>
                </td>
            `;
            impTbody.appendChild(tr);
        });
    }

    // Reports Download Links
    document.getElementById("btnDownloadMd").href = res.reports.markdown_url;
    document.getElementById("btnDownloadJson").href = res.reports.json_url;

    // Prediction Playground Setup
    document.getElementById("excludedTargetName").textContent = res.target_column;
    document.getElementById("featureCountBadge").textContent = `${res.features.length} Features`;
    const predTypeBadge = document.getElementById("predTypeBadge");
    predTypeBadge.textContent = res.problem_type.replace("_", " ").toUpperCase();
    predTypeBadge.className = `badge ${res.problem_type.includes("classification") ? "badge-task" : "badge-indigo"}`;

    renderSampleRowButtons(res.sample_test_rows);
    renderInferenceForm(res.features, res.feature_types, res.categorical_options);

    // Show Results Wrapper
    document.getElementById("resultsWrapper").classList.remove("hidden");
    document.getElementById("resultsWrapper").scrollIntoView({ behavior: "smooth" });
}

function formatPlotTitle(name) {
    return name
        .replace(/_/g, " ")
        .replace(/\b\w/g, c => c.toUpperCase());
}

// -----------------------------------------------------------------------------
// Prediction Playground Controller
// -----------------------------------------------------------------------------
function renderSampleRowButtons(sampleRows) {
    const wrap = document.getElementById("sampleButtonsWrap");
    wrap.innerHTML = "";

    if (!sampleRows || sampleRows.length === 0) {
        wrap.innerHTML = "<span style='font-size:0.75rem; color:var(--text-muted);'>Manual input mode</span>";
        return;
    }

    sampleRows.forEach((row, idx) => {
        const btn = document.createElement("button");
        btn.type = "button";
        btn.className = `btn-sample-chip ${idx === 0 ? "active" : ""}`;
        let label = row._row_label || `Example #${idx + 1}`;
        if (row._ground_truth !== undefined) {
            label += ` (Actual: ${row._ground_truth})`;
        }
        btn.textContent = label;
        btn.onclick = () => {
            document.querySelectorAll(".btn-sample-chip").forEach(b => b.classList.remove("active"));
            btn.classList.add("active");
            populateFormWithRow(row);
        };
        wrap.appendChild(btn);
    });
}

function renderInferenceForm(features, featureTypes, categoricalOptions) {
    const form = document.getElementById("inferenceForm");
    form.innerHTML = "";
    
    // Reset prediction output display
    document.getElementById("predEmptyState").classList.remove("hidden");
    document.getElementById("predActiveResult").classList.add("hidden");

    if (!features || features.length === 0) {
        form.innerHTML = "<p style='color:var(--text-muted); font-size:0.8rem;'>No features required.</p>";
        return;
    }

    features.forEach(feat => {
        const div = document.createElement("div");
        div.className = "inference-field";
        const fType = (featureTypes && featureTypes[feat]) ? featureTypes[feat].toLowerCase() : "float64";
        const isNum = fType.includes("int") || fType.includes("float");
        const options = (categoricalOptions && categoricalOptions[feat]) ? categoricalOptions[feat] : null;

        if (options && options.length > 0) {
            // Render select dropdown for categorical feature
            let optsHtml = "";
            options.forEach(optVal => {
                optsHtml += `<option value="${escapeHtml(optVal)}">${escapeHtml(optVal)}</option>`;
            });
            div.innerHTML = `
                <label class="field-label" title="${escapeHtml(feat)}">${escapeHtml(feat)}</label>
                <select id="inf_${escapeHtml(feat)}" class="field-select" name="${escapeHtml(feat)}">
                    ${optsHtml}
                </select>
            `;
        } else {
            // Render numeric or text input
            div.innerHTML = `
                <label class="field-label" title="${escapeHtml(feat)}">${escapeHtml(feat)}</label>
                <input type="${isNum ? 'number' : 'text'}" 
                       step="any"
                       id="inf_${escapeHtml(feat)}" 
                       class="field-input" 
                       placeholder="${isNum ? '0' : 'value'}"
                       name="${escapeHtml(feat)}">
            `;
        }
        form.appendChild(div);
    });

    // Populate with first sample row if available
    if (currentResults && currentResults.sample_test_rows && currentResults.sample_test_rows.length > 0) {
        populateFormWithRow(currentResults.sample_test_rows[0]);
    }
}

function populateFormWithRow(row) {
    if (!row) return;
    for (const [k, v] of Object.entries(row)) {
        if (k.startsWith("_")) continue; // skip metadata fields like _ground_truth
        const elem = document.getElementById(`inf_${k}`);
        if (elem) {
            elem.value = v;
        }
    }
}

async function runPrediction(e) {
    if (e) e.preventDefault();
    if (!currentResults) {
        showError("Model Missing", "Please load a verified run or run analysis before making predictions.");
        return;
    }

    closeErrorBanner();
    const btn = document.getElementById("btnRunPrediction");
    const originalBtnText = btn.innerHTML;
    btn.disabled = true;
    btn.innerHTML = `<span>⏳ Predicting with UnifiedPredictor...</span>`;

    const features = {};
    let hasValidationError = false;
    let validationMsg = "";

    currentResults.features.forEach(feat => {
        const elem = document.getElementById(`inf_${feat}`);
        if (elem) {
            const val = elem.value.trim();
            const fType = (currentResults.feature_types && currentResults.feature_types[feat]) ? currentResults.feature_types[feat].toLowerCase() : "";
            if (fType.includes("int") || fType.includes("float")) {
                if (val === "") {
                    features[feat] = 0;
                } else {
                    const num = parseFloat(val);
                    if (isNaN(num)) {
                        hasValidationError = true;
                        validationMsg = `Feature '${feat}' requires a valid numerical value.`;
                    } else {
                        features[feat] = num;
                    }
                }
            } else {
                features[feat] = val;
            }
        }
    });

    if (hasValidationError) {
        btn.disabled = false;
        btn.innerHTML = originalBtnText;
        showError("Input Validation Error", validationMsg);
        return;
    }

    try {
        const resp = await fetch("/api/predict", {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({
                run_id: currentResults.run_id,
                features: features,
            }),
        });

        const data = await resp.json();
        if (!resp.ok) {
            throw new Error(data.detail || "Prediction request failed.");
        }

        renderPredictionResult(data);
    } catch (err) {
        showError("Prediction Error", err.message);
    } finally {
        btn.disabled = false;
        btn.innerHTML = originalBtnText;
    }
}

function renderPredictionResult(data) {
    document.getElementById("predEmptyState").classList.add("hidden");
    document.getElementById("predActiveResult").classList.remove("hidden");

    const valElem = document.getElementById("predValueDisplay");
    const latElem = document.getElementById("predLatencyBadge");
    const titleElem = document.getElementById("predTargetTitle");
    const probContainer = document.getElementById("probBarContainer");
    const probBars = document.getElementById("probBars");
    const regSummaryBox = document.getElementById("regSummaryBox");

    latElem.textContent = `${data.latency_ms} ms (UnifiedPredictor)`;

    const isClf = data.problem_type && data.problem_type.includes("classification");
    if (isClf) {
        titleElem.textContent = `Predicted Class (${data.target_column || "Target"})`;
        regSummaryBox.classList.add("hidden");

        const predClassStr = String(data.prediction);
        let winningProbabilityPct = null;

        if (data.probabilities && data.probabilities.length > 0) {
            probContainer.classList.remove("hidden");
            probBars.innerHTML = "";
            const labels = data.class_labels || data.probabilities.map((_, i) => `Class ${i}`);

            data.probabilities.forEach((p, idx) => {
                const labelStr = String(labels[idx] !== undefined ? labels[idx] : idx);
                const isWinner = (labelStr === predClassStr || idx === data.prediction || String(idx) === predClassStr);
                const pctNum = (p * 100);
                const pct = pctNum.toFixed(1);

                if (isWinner) {
                    winningProbabilityPct = pct;
                }

                const row = document.createElement("div");
                row.className = `prob-bar-row ${isWinner ? "predicted-winner" : ""}`;
                row.innerHTML = `
                    <span class="prob-class-name" title="Class: ${escapeHtml(labelStr)}">${escapeHtml(labelStr)} ${isWinner ? "★" : ""}</span>
                    <div class="prob-bar-track">
                        <div class="prob-bar-fill" style="width: ${pct}%;"></div>
                    </div>
                    <span class="prob-bar-pct">${pct}%</span>
                `;
                probBars.appendChild(row);
            });
        } else {
            probContainer.classList.add("hidden");
        }

        if (winningProbabilityPct !== null) {
            valElem.innerHTML = `<span style="color:#2563eb;">${escapeHtml(predClassStr)}</span> <span style="font-size:1.1rem; color:var(--text-muted); font-weight:600;">(${winningProbabilityPct}% probability)</span>`;
        } else {
            valElem.innerHTML = `<span style="color:#2563eb;">${escapeHtml(predClassStr)}</span>`;
        }

    } else {
        // Regression formatting
        titleElem.textContent = `Predicted ${data.target_column || "Value"}`;
        probContainer.classList.add("hidden");
        regSummaryBox.classList.remove("hidden");

        const numVal = parseFloat(data.prediction);
        if (!isNaN(numVal)) {
            valElem.innerHTML = `<span style="color:#16a34a;">${numVal.toLocaleString(undefined, { minimumFractionDigits: 2, maximumFractionDigits: 2 })}</span>`;
        } else {
            valElem.innerHTML = `<span style="color:#16a34a;">${escapeHtml(String(data.prediction))}</span>`;
        }
    }
}

// -----------------------------------------------------------------------------
// Quick Benchmark Selector & Listing
// -----------------------------------------------------------------------------
function initBenchmarkSelector() {
    const select = document.getElementById("benchmarkSelect");
    select.addEventListener("change", (e) => {
        const runId = e.target.value;
        if (runId) {
            closeErrorBanner();
            fetchAndRenderResults(runId);
        }
    });
}

async function fetchRunsList() {
    try {
        const resp = await fetch("/api/runs");
        if (!resp.ok) return;
        const data = await resp.json();
        const select = document.getElementById("benchmarkSelect");
        select.innerHTML = `<option value="">-- Quick Load Verified Run --</option>`;

        if (data.runs && data.runs.length > 0) {
            data.runs.forEach(run => {
                const opt = document.createElement("option");
                opt.value = run.run_id;
                const pType = (run.problem_type && run.problem_type.includes("classification")) ? "Classification" : "Regression";
                opt.textContent = `${run.model_name} (${pType} - ${run.run_id})`;
                select.appendChild(opt);
            });
        }
    } catch (e) {
        console.error("Could not fetch runs list:", e);
    }
}

// -----------------------------------------------------------------------------
// Report Drawer & Plot Modal
// -----------------------------------------------------------------------------
async function toggleReportDrawer() {
    const drawer = document.getElementById("reportDrawer");
    if (!drawer.classList.contains("hidden")) {
        drawer.classList.add("hidden");
        return;
    }

    if (!currentResults || !currentResults.reports.markdown_url) return;

    try {
        const resp = await fetch(currentResults.reports.markdown_url);
        const mdText = await resp.text();
        document.getElementById("reportDrawerContent").textContent = mdText;
        drawer.classList.remove("hidden");
    } catch (err) {
        showError("Report Error", "Failed to load Markdown report content.");
    }
}

function openPlotModal(title, url) {
    document.getElementById("modalPlotTitle").textContent = title;
    document.getElementById("modalPlotImg").src = url;
    document.getElementById("plotModal").classList.remove("hidden");
}

function closePlotModal() {
    document.getElementById("plotModal").classList.add("hidden");
}

// -----------------------------------------------------------------------------
// Error Banner & Formatting Utilities
// -----------------------------------------------------------------------------
function showError(title, message) {
    const banner = document.getElementById("errorBanner");
    document.getElementById("errorTitle").textContent = title;
    document.getElementById("errorMessage").textContent = message;
    banner.classList.remove("hidden");
    banner.scrollIntoView({ behavior: "smooth" });
}

function closeErrorBanner() {
    document.getElementById("errorBanner").classList.add("hidden");
}

function formatNumber(val) {
    if (val === null || val === undefined) return "-";
    const num = Number(val);
    if (isNaN(num)) return String(val);
    if (Math.abs(num) >= 1000) {
        return num.toLocaleString(undefined, { maximumFractionDigits: 4 });
    }
    return num.toFixed(4);
}

function escapeHtml(str) {
    if (!str) return "";
    return String(str)
        .replace(/&/g, "&amp;")
        .replace(/</g, "&lt;")
        .replace(/>/g, "&gt;")
        .replace(/"/g, "&quot;")
        .replace(/'/g, "&#039;");
}
