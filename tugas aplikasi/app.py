from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np
import os

def classify_leaf(img_path):
    model_path = 'models/leaf_cnn.h5'
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model tidak ditemukan di {model_path}")

    cnn_model = load_model(model_path)

    img = image.load_img(img_path, target_size=(100, 100))
    img_array = image.img_to_array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    pred = cnn_model.predict(img_array)
    kelas_idx = np.argmax(pred)

    # Misal kelas disimpan saat training:
    class_indices = {'Mangga': 0, 'Jeruk': 1, 'Pepaya': 2}  
    idx_to_class = {v:k for k,v in class_indices.items()}

    print(f"Prediksi: {idx_to_class[kelas_idx]} dengan confidence {pred[0][kelas_idx]:.2f}")

if __name__ == "__main__":
    test_img_path = 'test/daun_test.jpg'  # ganti sesuai file gambar uji
    classify_leaf(test_img_path)
