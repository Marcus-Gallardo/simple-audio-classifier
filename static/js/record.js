let mediaRecorder;        
let stream = null;
let isRecording = false;
let stopwatchInterval = null;
let startTime = null;

let liveRecorder = null;
let aggregatedChunks = [];

document.addEventListener("DOMContentLoaded", () => {
    document.getElementById("recordButton").onclick = toggleRecording;
    populateMicOptions();
});

async function populateMicOptions() {
    const devices = await navigator.mediaDevices.enumerateDevices();
    const mics = devices.filter(d => d.kind === "audioinput");
    const select = document.getElementById("micSelect");

    mics.forEach(mic => {
        const opt = document.createElement("option");
        opt.value = mic.deviceId;
        opt.innerText = mic.label || `Microphone ${mic.deviceId}`;
        select.appendChild(opt);
    });
}

async function toggleRecording() {
    if (!isRecording) startRecording();
    else stopRecording();
}

async function startRecording() {
    const deviceId = document.getElementById("micSelect").value;

    stream = await navigator.mediaDevices.getUserMedia({
        audio: { deviceId: deviceId ? { exact: deviceId } : undefined }
    });

    isRecording = true;
    document.getElementById("recordButton").innerText = "Stop";
    document.getElementById("result").innerText = "Listening...";
    document.getElementById("probabilities").innerHTML = "";

    startTime = Date.now();
    stopwatchInterval = setInterval(updateStopwatch, 100);

    mediaRecorder = new MediaRecorder(stream);
    let fullChunks = [];

    mediaRecorder.ondataavailable = e => fullChunks.push(e.data);

    mediaRecorder.onstop = async () => {
        clearInterval(stopwatchInterval);
        document.getElementById("stopwatch").innerText = "00:00";

        let blob = new Blob(fullChunks, { type: 'audio/webm' });
        fullChunks = [];

        const formData = new FormData();
        formData.append("audio", blob, "final.webm");

        finalizePrediction(formData);
    };

    mediaRecorder.start();

    // Start UI mic level meter
    startAudioLevelMeter(stream);

    // Start continuous aggregated live predictions
    startLiveStreaming();
}

function stopRecording() {
    isRecording = false;

    document.getElementById("recordButton").innerText = "Record";

    if (mediaRecorder && mediaRecorder.state !== "inactive") {
        mediaRecorder.stop();
    }

    if (liveRecorder && liveRecorder.state !== "inactive") {
        liveRecorder.stop();
    }

    if (stopwatchInterval) clearInterval(stopwatchInterval);
}

async function startLiveStreaming() {
    liveRecorder = new MediaRecorder(stream);
    aggregatedChunks = [];   // reset the buffer

    liveRecorder.ondataavailable = async e => {
        if (!isRecording) return;

        // Add 1-second chunk to full session buffer
        aggregatedChunks.push(e.data);

        // Build one Blob containing ALL recording so far
        const fullBlob = new Blob(aggregatedChunks, { type: "audio/webm" });

        const formData = new FormData();
        formData.append("audio", fullBlob, "live_full.webm");

        try {
            const r = await fetch("/predict", { method: "POST", body: formData });
            const json = await r.json();

            if (!json.error) {
                document.getElementById("result").innerText =
                    "Live Prediction: " + json.prediction;

                displayProbabilities(json.probabilities);
            }
        } catch (err) {
            console.error("Live prediction error:", err);
        }

        // Continue the loop
        if (isRecording) {
            liveRecorder.start();
            setTimeout(() => liveRecorder.stop(), 1000);
        }
    };

    // Kick off first 1-second chunk
    liveRecorder.start();
    setTimeout(() => liveRecorder.stop(), 1000);
}

async function finalizePrediction(formData) {
    try {
        const response = await fetch("/predict", {
            method: "POST",
            body: formData
        });

        const json = await response.json();

        if (json.error) {
            document.getElementById("result").innerText = "Error: " + json.error;
            return;
        }

        document.getElementById("result").innerText =
            "Final Prediction: " + json.prediction;

        displayProbabilities(json.probabilities);

    } catch (err) {
        console.error(err);
        document.getElementById("result").innerText = "Error fetching prediction.";
    }
}

function updateStopwatch() {
    const elapsed = Date.now() - startTime;
    const sec = Math.floor(elapsed / 1000);
    const mins = Math.floor(sec / 60);
    const secs = sec % 60;

    document.getElementById("stopwatch").innerText =
        `${mins.toString().padStart(2,'0')}:${secs.toString().padStart(2,'0')}`;
}

function displayProbabilities(probabilities) {
    let container = document.getElementById("probabilities");
    container.innerHTML = "";

    let sorted = Object.entries(probabilities)
        .sort((a, b) => b[1] - a[1])
        .slice(0, 10);

    sorted.forEach(([label, prob]) => {
        const p = document.createElement("p");
        p.innerText = `${label}: ${(prob * 100).toFixed(1)}%`;
        container.appendChild(p);
    });
}

function startAudioLevelMeter(stream) {
    const ctx = new (window.AudioContext || window.webkitAudioContext)();
    const analyser = ctx.createAnalyser();
    const src = ctx.createMediaStreamSource(stream);

    analyser.fftSize = 256;
    src.connect(analyser);

    const data = new Uint8Array(analyser.fftSize);

    function updateMeter() {
        if (!isRecording) return;

        analyser.getByteTimeDomainData(data);

        let sum = 0;
        for (let i = 0; i < data.length; i++) {
            let v = (data[i] - 128) / 128;
            sum += v * v;
        }

        let rms = Math.sqrt(sum / data.length);
        document.getElementById("levelBar").style.width =
            `${Math.min(rms * 500, 100)}%`;

        requestAnimationFrame(updateMeter);
    }

    updateMeter();
}
