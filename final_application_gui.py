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
MODEL_PATH = "./best_model_eval/ecg_hybrid_model_batchsize64.h5"
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
        self.root.title("ECG Hybrid Classifier")
        self.root.geometry("600x700")
        self.root.configure(bg="#f0f4f7")

        style = ttk.Style()
        style.configure("TButton", font=("Segoe UI", 12), padding=6)
        style.configure("TLabel", font=("Segoe UI", 12), background="#f0f4f7")

        self.model = tf.keras.models.load_model(MODEL_PATH, compile=False)

        self.model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
            loss=tf.keras.losses.SparseCategoricalCrossentropy(),
            metrics=['accuracy'])

        self.title_label = ttk.Label(root, text="ECG Hybrid Classification App", font=("Segoe UI", 16, "bold"))
        self.title_label.pack(pady=10)

        self.upload_button = ttk.Button(root, text="Upload ECG Files (.hea, .dat, .atr)", command=self.load_files)
        self.upload_button.pack(pady=10)

        self.canvas = tk.Label(root, bg="#ffffff", width=400, height=400, relief="solid", bd=1)
        self.canvas.pack(pady=15)

        self.result_label = ttk.Label(root, text="", font=("Segoe UI", 14))
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

        # Get base path (without extension)
        base = os.path.splitext(files[0])[0]
        folder = os.path.dirname(files[0])
        base_name = os.path.basename(base)

        try:
            # Load from local directory (no pn_dir!)
            record = wfdb.rdrecord(base, sampto=3000, channel_names=['MLII'])
            data = record.p_signal.flatten()

            ann = wfdb.rdann(base, 'atr')
            # take first beat for demo
            idx = ann.sample[0]
            segment = data[idx-99:idx+201] if idx > 99 else data[0:300]
            if len(segment) < 300:
                segment = np.pad(segment, (0, 300-len(segment)))

            sig = normalize_signal(segment)

            # generate 2D image
            im = signal_to_image(sig)
            self.display_image(im)

            # prepare for model
            sig_in = sig.reshape(1, -1)
            img_in = np.array(im).astype(np.float32) / 255.0
            img_in = np.expand_dims(img_in, 0)

            pred = self.model.predict([sig_in, img_in])
            label_idx = np.argmax(pred, axis=-1)[0]
            label = CLASS_NAMES[label_idx]
            prob = np.max(pred) * 100

            self.result_label.configure(text=f"Prediction: {label} ({prob:.2f}%)")

        except Exception as e:
            messagebox.showerror("Error", f"Failed to load ECG files: {e}")        
            files = filedialog.askopenfilenames(title="Select ECG files", filetypes=[("ECG files", "*.hea *.dat *.atr")])

            if not files:
                return
            base = os.path.splitext(os.path.basename(files[0]))[0]
            try:
                record = wfdb.rdrecord(base, pn_dir=os.path.dirname(files[0]), channel_names=['MLII'])
                data = record.p_signal.flatten()
                ann = wfdb.rdann(base, 'atr', pn_dir=os.path.dirname(files[0]))
                # take first beat for demo
                idx = ann.sample[0]
                segment = data[idx-99:idx+201] if idx > 99 else data[0:300]
                if len(segment) < 300:
                    segment = np.pad(segment, (0,300-len(segment)))
                sig = normalize_signal(segment)

                # generate 2D image
                im = signal_to_image(sig)
                self.display_image(im)

                # prepare for model
                sig_in = sig.reshape(1,-1)
                img_in = np.array(im).astype(np.float32)/255.0
                img_in = np.expand_dims(img_in,0)

                pred = self.model.predict([sig_in, img_in])
                label_idx = np.argmax(pred, axis=-1)[0]
                label = CLASS_NAMES[label_idx]
                prob = np.max(pred)*100

                self.result_label.configure(text=f"Prediction: {label} ({prob:.2f}%)")

            except Exception as e:
                messagebox.showerror("Error", f"Failed to load ECG files: {e}")
            
        self.result_label.configure(text=f"Prediction: {label} ({prob:.2f}%)")
        self.upload_button.state(["disabled"])

    def display_image(self, im):
        imtk = ImageTk.PhotoImage(im)
        self.canvas.configure(image=imtk)
        self.canvas.image = imtk

    def reset(self):
        self.canvas.configure(image="")
        self.canvas.image = None
        self.result_label.configure(text="")
        self.upload_button.state(["!disabled"])  # re-enable upload

if __name__ == "__main__":
    root = tk.Tk()
    app = ECGApp(root)
    root.mainloop()
