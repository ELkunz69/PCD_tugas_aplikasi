import tkinter as tk
from tkinter import filedialog, messagebox
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np
import os
import json

class LeafClassifierApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Leaf Classifier")
        self.root.geometry("400x300")

        # Path model CNN
        self.model_path = 'models/leaf_cnn.h5'
        if not os.path.exists(self.model_path):
            messagebox.showerror("Error", f"Model tidak ditemukan di {self.model_path}")
            root.destroy()
            return

        self.model = load_model(self.model_path)

        # Mapping kelas (bisa diganti otomatis dari file JSON jika disimpan saat training)
        self.class_indices = {'Mangga': 1, 'Jeruk': 2, 'Pepaya': 0}
        self.idx_to_class = {v: k for k, v in self.class_indices.items()}

        # GUI Components
        self.label = tk.Label(root, text="Pilih gambar daun untuk klasifikasi", font=("Arial", 14))
        self.label.pack(pady=20)

        self.btn_load = tk.Button(root, text="Pilih Gambar", command=self.load_image)
        self.btn_load.pack()

        self.result_label = tk.Label(root, text="", font=("Arial", 12))
        self.result_label.pack(pady=20)

    def load_image(self):
        file_path = filedialog.askopenfilename(
            title="Pilih Gambar Daun",
            filetypes=[("Image Files", "*.jpg *.jpeg *.png")]
        )
        if file_path:
            try:
                self.predict_leaf(file_path)
            except Exception as e:
                messagebox.showerror("Error", f"Gagal memproses gambar:\n{e}")

    def predict_leaf(self, img_path):
        img = image.load_img(img_path, target_size=(100, 100))
        img_array = image.img_to_array(img) / 255.0
        img_array = np.expand_dims(img_array, axis=0)

        pred = self.model.predict(img_array)
        kelas_idx = np.argmax(pred[0])
        kelas = self.idx_to_class.get(kelas_idx, "Tidak diketahui")
        confidence = float(pred[0][kelas_idx])

        self.result_label.config(
            text=f"Hasil Prediksi:\n{kelas}\nConfidence: {confidence:.2f}"
        )

# Jalankan aplikasi
if __name__ == "__main__":
    root = tk.Tk()
    app = LeafClassifierApp(root)
    root.mainloop()
