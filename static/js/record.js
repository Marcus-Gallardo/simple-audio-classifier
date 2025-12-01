let chunks = [];
let mediaRecorder;

document.addEventListener("DOMContentLoaded", () => {
    document.getElementById("recordButton").onclick = () => {
        const duration = parseInt(document.getElementById("duration").value) || 5;
        document.getElementById("result").innerText = ""
        document.getElementById("probabilities").innerText = ""
        recordAndProcessAudio(duration);
    };

    populateMicOptions();
});

async function populateMicOptions() {
    const devices = await navigator.mediaDevices.enumerateDevices();
    const mics = devices.filter(d => d.kind === "audioinput");
    const select = document.getElementById("micSelect");
    mics.forEach(mic => {
        const option = document.createElement("option");
        option.value = mic.deviceId;
        option.innerText = mic.label || `Microphone ${mic.deviceId}`;
        select.appendChild(option);
    });
}

async function recordAndProcessAudio(duration) {
    const select = document.getElementById("micSelect");
    const deviceId = select.value;

    const stream = await navigator.mediaDevices.getUserMedia({
        audio: { deviceId: deviceId ? { exact: deviceId } : undefined }
    });

    mediaRecorder = new MediaRecorder(stream);
    chunks = [];

    mediaRecorder.ondataavailable = e => chunks.push(e.data);

    mediaRecorder.onstop = async () => {
        let blob = new Blob(chunks, { type: chunks[0].type || 'audio/webm' });
        let formData = new FormData();
        formData.append("audio", blob, "recording.webm");

        try {
            const response = await fetch("/predict", { method: "POST", body: formData });
            const json = await response.json();

            if (json.error) {
                document.getElementById("result").innerText = "Error: " + json.error;
                return;
            }

            document.getElementById("result").innerText = "Prediction: " + json.prediction;

            // Display top probabilities
            let probsContainer = document.getElementById("probabilities");
            probsContainer.innerHTML = "";
            let sorted = Object.entries(json.probabilities).sort((a, b) => b[1] - a[1]);
            for (let [label, prob] of sorted) {
                let p = document.createElement("p");
                p.innerText = `${label}: ${(prob*100).toFixed(1)}%`;
                probsContainer.appendChild(p);
            }
        } catch (err) {
            document.getElementById("result").innerText = "Error fetching prediction";
            console.error(err);
        }
    };

    mediaRecorder.start();

    // Automatically stop after `duration` seconds
    setTimeout(() => mediaRecorder.stop(), duration * 1000);

    // Optional: add audio level meter
    const audioContext = new (window.AudioContext || window.webkitAudioContext)();
    const analyser = audioContext.createAnalyser();
    const source = audioContext.createMediaStreamSource(stream);
    source.connect(analyser);
    analyser.fftSize = 256;
    const dataArray = new Uint8Array(analyser.fftSize);

    function updateLevel() {
        analyser.getByteTimeDomainData(dataArray);
        let sum = 0;
        for (let i = 0; i < dataArray.length; i++) {
            let v = (dataArray[i] - 128) / 128;
            sum += v * v;
        }
        let rms = Math.sqrt(sum / dataArray.length);
        document.getElementById("levelBar").style.width = `${Math.min(rms*500, 100)}%`;
        if (mediaRecorder.state === "recording") requestAnimationFrame(updateLevel);
    }

    updateLevel();
}
