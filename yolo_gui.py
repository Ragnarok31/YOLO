import tkinter as tk
from tkinter import messagebox
from PIL import Image, ImageTk
import cv2
from ultralytics import YOLO

# Load your trained YOLOv8 model
MODEL_PATH = r"C:\Users\Dell\Desktop\YOLO\runs\detect\train6\weights"






model = YOLO(MODEL_PATH)

# Global webcam + detection toggle
cap = None
is_detecting = False

class YOLOApp:
    def __init__(self, root):
        self.root = root
        self.root.title("YOLOv8 Smart Object Detector")
        self.root.geometry("900x700")

        # Start webcam
        self.start_cam_btn = tk.Button(root, text="Start Webcam", command=self.start_camera)
        self.start_cam_btn.pack(pady=10)

        # Toggle detection
        self.start_detect_btn = tk.Button(root, text="Start Detection", command=self.toggle_detection)
        self.start_detect_btn.pack(pady=10)

        # Output frame
        self.video_frame = tk.Label(root)
        self.video_frame.pack(pady=10)

    def start_camera(self):
        global cap
        if cap is None:
            cap = cv2.VideoCapture(0)
            if not cap.isOpened():
                messagebox.showerror("Error", "Webcam not accessible.")
                return
            self.update_frame()
        else:
            messagebox.showinfo("Info", "Webcam already started.")

    def toggle_detection(self):
        global is_detecting
        if not cap or not cap.isOpened():
            messagebox.showerror("Error", "Start the webcam first.")
            return
        is_detecting = not is_detecting
        self.start_detect_btn.config(text="Stop Detection" if is_detecting else "Start Detection")

    def update_frame(self):
        if cap:
            ret, frame = cap.read()
            if ret:
                if is_detecting:
                    results = model(frame, verbose=False)[0]
                    frame = results.plot()
                # Convert and display frame
                img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                img = Image.fromarray(img_rgb)
                imgtk = ImageTk.PhotoImage(image=img)
                self.video_frame.imgtk = imgtk
                self.video_frame.config(image=imgtk)

        self.root.after(10, self.update_frame)

    def on_close(self):
        global cap
        if cap:
            cap.release()
        self.root.destroy()

# Run the GUI
if __name__ == "__main__":
    root = tk.Tk()
    app = YOLOApp(root)
    root.protocol("WM_DELETE_WINDOW", app.on_close)
    root.mainloop()
