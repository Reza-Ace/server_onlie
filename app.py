from flask import Flask, request, jsonify
import tensorflow as tf
import numpy as np
import io
import soundfile as sf

app = Flask(__name__)

# Load model saat server start
MODEL_PATH = "horn_detection_model1.h5"
model = tf.keras.models.load_model(MODEL_PATH)

@app.route("/", methods=["GET"])
def home():
    return "🚀 Klakson Detection Server Running"

@app.route("/predict", methods=["POST"])
def predict():
    try:
        # Ambil file audio dari ESP32
        if "audio" not in request.files:
            return jsonify({"error": "No audio file"}), 400
        
        file = request.files["audio"]
        
        # Baca audio (pastikan ESP32 kirim WAV atau PCM)
        data, samplerate = sf.read(io.BytesIO(file.read()))
        
        # Preprocessing sesuai training
        data = np.array(data, dtype=np.float32)
        data = np.expand_dims(data, axis=0)  # shape (1, n_samples)

        # Prediksi
        prediction = model.predict(data)
        score = float(prediction[0][0])  # asumsi output 1 neuron sigmoid
        
        return jsonify({
            "score": score,
            "klakson_detected": score > 0.8
        })
    
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
