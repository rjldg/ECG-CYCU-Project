import os
import tkinter as tk
from tkinter import filedialog, messagebox
from tkinter import ttk

import numpy as np
import wfdb
import matplotlib.pyplot as plt
from PIL import Image, ImageTk
import tensorflow as tf

# -------------------------
# Configuration
# -------------------------
MODEL_PATH_1D = "./JD_ecg_1d_cnn_9888.h5"
MODEL_PATH_2D = "./JD_ecg_2d_cnn_9866.h5"
MODEL_PATH_HYBRID = "./ecg_hybrid_model_batchsize64.h5"
IMG_SIZE = 128
CLASS_NAMES = ['N','e','j','L','R','A','a','J','S','V','E','F','/','f','Q']

# -------------------------
# Utility functions
# -------------------------
def normalize_signal(sig):
    sig = sig.astype(np.float32)
    std = sig.std() if sig.std() > 1e-6 else 1.0
    return (sig - sig.mean()) / std

def signal_to_image(signal, out_path="temp_ecg.png"):
    plt.figure(figsize=(IMG_SIZE/100, IMG_SIZE/100), dpi=100)
    ax = plt.axes([0,0,1,1])
    ax.set_axis_off()
    sig_norm = (signal - signal.min()) / (signal.max() - signal.min() + 1e-8)
    x = np.linspace(0, 1, len(sig_norm))
    ax.plot(x, sig_norm, color="blue")
    plt.savefig(out_path, dpi=100, bbox_inches='tight', pad_inches=0)
    plt.close()
    im = Image.open(out_path).convert("L")
    im = im.resize((IMG_SIZE, IMG_SIZE), Image.BICUBIC)
    return im

# -------------------------
# Application class
# -------------------------
class ECGApp:
    def __init__(self, root):
        self.root = root
        self.root.title("ECG Classifier")
        self.root.geometry("600x800")
        self.root.configure(bg="#f0f4f7")

        style = ttk.Style()
        style.configure("TButton", font=("Segoe UI", 12), padding=6)
        style.configure("TLabel", font=("Segoe UI", 12), background="#f0f4f7")

        # Load models
        self.model_1d = tf.keras.models.load_model(MODEL_PATH_1D, compile=False)
        self.model_2d = tf.keras.models.load_model(MODEL_PATH_2D, compile=False)
        self.model_hybrid = tf.keras.models.load_model(MODEL_PATH_HYBRID, compile=False)

        self.title_label = ttk.Label(root, text="ECG Classification App", font=("Segoe UI", 16, "bold"))
        self.title_label.pack(pady=10)

        self.upload_button = ttk.Button(root, text="Upload ECG Files (.hea, .dat, .atr)", command=self.load_files)
        self.upload_button.pack(pady=10)

        self.canvas = tk.Label(root, bg="#ffffff", width=400, height=400, relief="solid", bd=1)
        self.canvas.pack(pady=15)

        self.result_label = ttk.Label(root, text="", font=("Segoe UI", 12), wraplength=550, justify="left")
        self.result_label.pack(pady=15)

        self.reset_button = ttk.Button(root, text="Restart", command=self.reset)
        self.reset_button.pack(pady=10)

    def load_files(self):
        files = filedialog.askopenfilenames(
            title="Select ECG files", 
            filetypes=[("ECG files", "*.hea *.dat *.atr")]
        )
        if not files:
            return

        base = os.path.splitext(files[0])[0]

        try:
            record = wfdb.rdrecord(base, sampto=3000, channel_names=['MLII'])
            data = record.p_signal.flatten()

            ann = wfdb.rdann(base, 'atr')
            idx = ann.sample[0]
            segment = data[idx-99:idx+201] if idx > 99 else data[0:300]
            if len(segment) < 300:
                segment = np.pad(segment, (0, 300-len(segment)))

            sig = normalize_signal(segment)
            im = signal_to_image(sig)
            self.display_image(im)

            sig_in = sig.reshape(1, -1)
            img_in = np.array(im).astype(np.float32)/255.0
            img_in = np.expand_dims(img_in, 0)

            # --- Ask user choice ---
            choice = messagebox.askquestion("Model Selection", "Use separate 1D + 2D CNN for joint decision? (Yes) or Hybrid model? (No)")

            if choice == 'yes':
                # 1D model prediction
                pred_1d = self.model_1d.predict(sig_in)
                label_1d = CLASS_NAMES[np.argmax(pred_1d)]
                verdict_1d = "Normal" if label_1d == 'N' else "Abnormal"

                # 2D model prediction
                pred_2d = self.model_2d.predict(img_in)
                label_2d = CLASS_NAMES[np.argmax(pred_2d)]
                verdict_2d = "Normal" if label_2d == 'N' else "Abnormal"

                # Joint decision
                if verdict_1d == verdict_2d:
                    final_verdict = verdict_1d
                else:
                    final_verdict = "Conflict: Review needed"

                self.result_label.configure(text=f"1D CNN → {label_1d} ({verdict_1d})\n2D CNN → {label_2d} ({verdict_2d})\nFinal Decision: {final_verdict}")

            else:
                pred = self.model_hybrid.predict([sig_in, img_in])
                label = CLASS_NAMES[np.argmax(pred)]
                verdict = "Normal" if label == 'N' else "Abnormal"
                prob = np.max(pred) * 100

                self.result_label.configure(text=f"Hybrid Prediction: {label} ({prob:.2f}%) → {verdict}")

        except Exception as e:
            messagebox.showerror("Error", f"Failed to load ECG files: {e}")

        self.upload_button.state(["disabled"])

    def display_image(self, im):
        imtk = ImageTk.PhotoImage(im)
        self.canvas.configure(image=imtk)
        self.canvas.image = imtk

    def reset(self):
        self.canvas.configure(image="")
        self.canvas.image = None
        self.result_label.configure(text="")
        self.upload_button.state(["!disabled"])

if __name__ == "__main__":
    root = tk.Tk()
    app = ECGApp(root)
    root.mainloop()