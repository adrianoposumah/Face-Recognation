import os
import cv2
import numpy as np
import face_recognition
from tkinter import *
from tkinter import filedialog, messagebox
from collections import defaultdict
import threading

# ======= Fungsi Utama Aplikasi =======

def load_target_encodings(targets_folder):
    encodings_by_label = defaultdict(list)
    allowed = {'.jpg', '.jpeg', '.png'}

    for fname in os.listdir(targets_folder):
        name, ext = os.path.splitext(fname)
        if ext.lower() not in allowed:
            continue

        path = os.path.join(targets_folder, fname)
        image = face_recognition.load_image_file(path)
        boxes = face_recognition.face_locations(image, model='hog')

        if len(boxes) == 0:
            print(f"[WARN] Tidak ada wajah ditemukan di {fname}")
            continue

        enc = face_recognition.face_encodings(image, boxes)[0]
        label = name.split('_')[0]
        encodings_by_label[label].append(enc)

    avg_encodings = {}
    for label, encs in encodings_by_label.items():
        avg_encodings[label] = np.mean(encs, axis=0)

    return avg_encodings


def distance_to_percent(dist, threshold=0.6):
    if dist <= 0:
        return 100
    if dist >= threshold:
        return 0
    return max(0, min(100, (1 - dist / threshold) * 100))


def annotate_frame(frame, loc, text):
    top, right, bottom, left = loc
    cv2.rectangle(frame, (left, top), (right, bottom), (0,255,0), 2)

    (w, h), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
    cv2.rectangle(frame, (left, top - h - 4), (left + w + 4, top), (0,255,0), -1)
    cv2.putText(frame, text, (left + 2, top - 2), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,0), 1)


def process_video(video_path, targets_folder):
    if not video_path or not targets_folder:
        messagebox.showerror("Error", "Video dan folder target harus dipilih!")
        return

    targets = load_target_encodings(targets_folder)
    if len(targets) == 0:
        messagebox.showerror("Error", "Tidak ada foto target valid!")
        return

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        messagebox.showerror("Error", "Gagal membuka video!")
        return

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        face_locations = face_recognition.face_locations(rgb, model='hog')
        face_encs = face_recognition.face_encodings(rgb, face_locations)

        for loc, enc in zip(face_locations, face_encs):
            best_label = "Unknown"
            best_percent = 0

            for label, target_enc in targets.items():
                dist = np.linalg.norm(target_enc - enc)
                percent = distance_to_percent(dist)

                if percent > best_percent:
                    best_percent = percent
                    best_label = label

            annotate_frame(frame, loc, f"{best_label} {best_percent:.1f}%")

        cv2.imshow("Face Identifier GUI", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()


# ======= GUI TKINTER =======

root = Tk()
root.title("Face Identifier App")
root.geometry("420x250")
root.resizable(False, False)

video_path = ""
targets_folder = ""

def pick_video():
    global video_path
    video_path = filedialog.askopenfilename(
        title="Pilih Video",
        filetypes=[("Video Files", "*.mp4 *.avi *.mov *.mkv")]
    )
    video_label.config(text="Video: " + (video_path if video_path else "Belum dipilih"))

def pick_targets():
    global targets_folder
    targets_folder = filedialog.askdirectory(title="Pilih Folder Foto Target")
    targets_label.config(text="Targets: " + (targets_folder if targets_folder else "Belum dipilih"))

def start_process():
    # jalankan di thread agar GUI tidak freeze
    threading.Thread(target=process_video, args=(video_path, targets_folder), daemon=True).start()

Label(root, text="Face Identifier", font=("Arial", 16, "bold")).pack(pady=10)

video_label = Label(root, text="Video: Belum dipilih")
video_label.pack()

Button(root, text="Pilih Video", width=20, command=pick_video).pack(pady=5)

targets_label = Label(root, text="Targets: Belum dipilih")
targets_label.pack()

Button(root, text="Pilih Folder Target", width=20, command=pick_targets).pack(pady=5)

Button(root, text="Mulai", width=20, bg="#4CAF50", fg="white", command=start_process).pack(pady=15)

Label(root, text="Tekan 'q' untuk menutup video", fg="gray").pack()

root.mainloop()
