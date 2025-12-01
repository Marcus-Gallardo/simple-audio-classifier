import traceback

from flask import Flask, render_template, request, jsonify
from src.inference import load_model, predict_audio

app = Flask(__name__)

model_name = "Model v0"

model, label_map = load_model(model_name)

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    try:
        # Ensure audio is present
        if "audio" not in request.files:
            return jsonify({"error": "No file uploaded"}), 400

        audio_file = request.files["audio"]

        # Make prediction
        pred_label, labeled_probs = predict_audio(model, label_map, audio_file)
        
        return jsonify({
            "prediction": pred_label,
            "probabilities": labeled_probs
        })
    except Exception as e:
        print("[PREDICT ERROR]", e)
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    app.run(debug=True)
