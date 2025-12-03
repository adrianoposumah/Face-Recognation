import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import cv2
import numpy as np
from PIL import Image, ImageTk
import os
from pathlib import Path
import threading
import queue
import time
from datetime import datetime
import insightface
from insightface.app import FaceAnalysis
import warnings
warnings.filterwarnings('ignore')

class FaceRecognitionApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Face Detection & Recognition System")
        self.root.geometry("1200x700")
        self.root.configure(bg='#2c3e50')
        
        # Initialize variables
        self.video_path = None
        self.target_folder = None
        self.processing = False
        self.video_writer = None
        self.cap = None
        self.output_video_path = None
        self.face_db = {}
        self.face_embeddings = []
        self.face_names = []
        self.current_frame = None
        
        # Model
        self.detector = None
        
        # Setup GUI
        self.setup_gui()
        
        # Queue for thread-safe GUI updates
        self.queue = queue.Queue()
        
    def setup_gui(self):
        # Main container
        main_frame = tk.Frame(self.root, bg='#2c3e50')
        main_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # Left Panel - Controls
        left_panel = tk.Frame(main_frame, bg='#34495e', width=300, relief=tk.RAISED, borderwidth=2)
        left_panel.pack(side=tk.LEFT, fill=tk.Y, padx=(0, 10))
        left_panel.pack_propagate(False)
        
        # Title
        title_label = tk.Label(left_panel, text="Face Recognition System", 
                              font=('Arial', 16, 'bold'), bg='#34495e', fg='#ecf0f1')
        title_label.pack(pady=20)
        
        # Separator
        ttk.Separator(left_panel, orient='horizontal').pack(fill=tk.X, padx=20, pady=5)
        
        # Video Selection
        video_frame = tk.Frame(left_panel, bg='#34495e')
        video_frame.pack(fill=tk.X, padx=20, pady=10)
        
        video_label = tk.Label(video_frame, text="Video File:", 
                              font=('Arial', 11, 'bold'), bg='#34495e', fg='#bdc3c7')
        video_label.pack(anchor=tk.W)
        
        self.video_path_var = tk.StringVar()
        video_entry = tk.Entry(video_frame, textvariable=self.video_path_var, 
                              state='readonly', width=30, font=('Arial', 10))
        video_entry.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=(0, 5))
        
        browse_video_btn = tk.Button(video_frame, text="Browse", command=self.browse_video,
                                    bg='#3498db', fg='white', font=('Arial', 10, 'bold'),
                                    relief=tk.RAISED)
        browse_video_btn.pack(side=tk.RIGHT)
        
        # Target Images Folder
        target_frame = tk.Frame(left_panel, bg='#34495e')
        target_frame.pack(fill=tk.X, padx=20, pady=10)
        
        target_label = tk.Label(target_frame, text="Target Images Folder:", 
                               font=('Arial', 11, 'bold'), bg='#34495e', fg='#bdc3c7')
        target_label.pack(anchor=tk.W)
        
        self.target_path_var = tk.StringVar()
        target_entry = tk.Entry(target_frame, textvariable=self.target_path_var, 
                               state='readonly', width=30, font=('Arial', 10))
        target_entry.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=(0, 5))
        
        browse_target_btn = tk.Button(target_frame, text="Browse", command=self.browse_target_folder,
                                     bg='#3498db', fg='white', font=('Arial', 10, 'bold'),
                                     relief=tk.RAISED)
        browse_target_btn.pack(side=tk.RIGHT)
        
        # Similarity Threshold Control
        threshold_frame = tk.Frame(left_panel, bg='#34495e')
        threshold_frame.pack(fill=tk.X, padx=20, pady=10)
        
        threshold_label = tk.Label(threshold_frame, text="Minimum Similarity (%):", 
                                  font=('Arial', 11, 'bold'), bg='#34495e', fg='#bdc3c7')
        threshold_label.pack(anchor=tk.W)
        
        self.threshold_var = tk.IntVar(value=70)
        threshold_scale = tk.Scale(threshold_frame, from_=50, to=95, resolution=5,
                                  orient=tk.HORIZONTAL, variable=self.threshold_var,
                                  bg='#34495e', fg='white', troughcolor='#2c3e50',
                                  length=250)
        threshold_scale.pack(fill=tk.X)
        
        threshold_value_label = tk.Label(threshold_frame, 
                                        text=f"Current: {self.threshold_var.get()}%",
                                        font=('Arial', 9), bg='#34495e', fg='#ecf0f1')
        threshold_value_label.pack()
        self.threshold_var.trace('w', lambda *args: threshold_value_label.config(
            text=f"Current: {self.threshold_var.get()}%"))
        
        # Detection Confidence
        conf_frame = tk.Frame(left_panel, bg='#34495e')
        conf_frame.pack(fill=tk.X, padx=20, pady=10)
        
        conf_label = tk.Label(conf_frame, text="Face Detection Confidence:", 
                             font=('Arial', 11, 'bold'), bg='#34495e', fg='#bdc3c7')
        conf_label.pack(anchor=tk.W)
        
        self.conf_var = tk.DoubleVar(value=0.5)
        conf_scale = tk.Scale(conf_frame, from_=0.1, to=1.0, resolution=0.05,
                             orient=tk.HORIZONTAL, variable=self.conf_var,
                             bg='#34495e', fg='white', troughcolor='#2c3e50',
                             length=250)
        conf_scale.pack(fill=tk.X)
        
        # Color Settings
        color_frame = tk.Frame(left_panel, bg='#34495e')
        color_frame.pack(fill=tk.X, padx=20, pady=10)
        
        color_label = tk.Label(color_frame, text="Display Colors:", 
                              font=('Arial', 11, 'bold'), bg='#34495e', fg='#bdc3c7')
        color_label.pack(anchor=tk.W)
        
        # Recognized color
        recog_frame = tk.Frame(color_frame, bg='#34495e')
        recog_frame.pack(fill=tk.X, pady=2)
        
        recog_color_label = tk.Label(recog_frame, text="Recognized: ", 
                                    font=('Arial', 9), bg='#34495e', fg='#ecf0f1')
        recog_color_label.pack(side=tk.LEFT)
        
        self.recog_color_canvas = tk.Canvas(recog_frame, width=20, height=20, bg='#00FF00')
        self.recog_color_canvas.pack(side=tk.LEFT, padx=(0, 10))
        
        recog_color_text = tk.Label(recog_frame, text="Green", 
                                   font=('Arial', 9), bg='#34495e', fg='#ecf0f1')
        recog_color_text.pack(side=tk.LEFT)
        
        # Unknown color
        unknown_frame = tk.Frame(color_frame, bg='#34495e')
        unknown_frame.pack(fill=tk.X, pady=2)
        
        unknown_color_label = tk.Label(unknown_frame, text="Unknown: ", 
                                      font=('Arial', 9), bg='#34495e', fg='#ecf0f1')
        unknown_color_label.pack(side=tk.LEFT)
        
        self.unknown_color_canvas = tk.Canvas(unknown_frame, width=20, height=20, bg='#FF0000')
        self.unknown_color_canvas.pack(side=tk.LEFT, padx=(0, 10))
        
        unknown_color_text = tk.Label(unknown_frame, text="Red", 
                                     font=('Arial', 9), bg='#34495e', fg='#ecf0f1')
        unknown_color_text.pack(side=tk.LEFT)
        
        # Separator
        ttk.Separator(left_panel, orient='horizontal').pack(fill=tk.X, padx=20, pady=10)
        
        # Control Buttons
        button_frame = tk.Frame(left_panel, bg='#34495e')
        button_frame.pack(fill=tk.X, padx=20, pady=10)
        
        self.process_btn = tk.Button(button_frame, text="Start Processing", command=self.start_processing,
                                    bg='#2ecc71', fg='white', font=('Arial', 12, 'bold'),
                                    height=2, width=15)
        self.process_btn.pack(pady=5)
        
        self.stop_btn = tk.Button(button_frame, text="Stop", command=self.stop_processing,
                                 bg='#e74c3c', fg='white', font=('Arial', 12, 'bold'),
                                 height=2, width=15, state=tk.DISABLED)
        self.stop_btn.pack(pady=5)
        
        # Progress Frame
        progress_frame = tk.Frame(left_panel, bg='#34495e')
        progress_frame.pack(fill=tk.X, padx=20, pady=10)
        
        self.progress_label = tk.Label(progress_frame, text="Ready", 
                                      font=('Arial', 10), bg='#34495e', fg='#ecf0f1')
        self.progress_label.pack(anchor=tk.W)
        
        self.progress_bar = ttk.Progressbar(progress_frame, length=250, mode='determinate')
        self.progress_bar.pack(fill=tk.X, pady=5)
        
        # Status Frame
        status_frame = tk.Frame(left_panel, bg='#34495e')
        status_frame.pack(fill=tk.X, padx=20, pady=10)
        
        self.status_label = tk.Label(status_frame, text="Status: Idle", 
                                    font=('Arial', 10), bg='#34495e', fg='#ecf0f1')
        self.status_label.pack(anchor=tk.W)
        
        # Right Panel - Video Display
        right_panel = tk.Frame(main_frame, bg='#2c3e50')
        right_panel.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True)
        
        # Video Display Frame
        display_frame = tk.Frame(right_panel, bg='#1a252f', relief=tk.SUNKEN, borderwidth=2)
        display_frame.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        self.video_label = tk.Label(display_frame, bg='#1a252f')
        self.video_label.pack(expand=True, fill=tk.BOTH)
        
        # Statistics Frame
        stats_frame = tk.Frame(right_panel, bg='#34495e')
        stats_frame.pack(fill=tk.X, padx=5, pady=5)
        
        # Stats labels
        stats_grid = tk.Frame(stats_frame, bg='#34495e')
        stats_grid.pack(padx=10, pady=5)
        
        self.total_faces_label = tk.Label(stats_grid, text="Total Faces Detected: 0", 
                                         font=('Arial', 10), bg='#34495e', fg='#ecf0f1')
        self.total_faces_label.grid(row=0, column=0, padx=10, pady=2, sticky=tk.W)
        
        self.recognized_faces_label = tk.Label(stats_grid, text="Recognized: 0", 
                                              font=('Arial', 10), bg='#34495e', fg='#00FF00')
        self.recognized_faces_label.grid(row=0, column=1, padx=10, pady=2, sticky=tk.W)
        
        self.unknown_faces_label = tk.Label(stats_grid, text="Unknown: 0", 
                                           font=('Arial', 10), bg='#34495e', fg='#FF0000')
        self.unknown_faces_label.grid(row=1, column=0, padx=10, pady=2, sticky=tk.W)
        
        self.target_persons_label = tk.Label(stats_grid, text="Target Persons: 0", 
                                            font=('Arial', 10), bg='#34495e', fg='#ecf0f1')
        self.target_persons_label.grid(row=1, column=1, padx=10, pady=2, sticky=tk.W)
        
        # Info Frame
        info_frame = tk.Frame(right_panel, bg='#2c3e50')
        info_frame.pack(fill=tk.X, padx=5, pady=5)
        
        info_text = """
        Instructions:
        1. Select a video file (.mp4, .avi, .mov)
        2. Select folder with target images
        3. Target images should be named: name_number.jpg
           Example: john_1.jpg, john_2.jpg, jane_1.jpg
        4. Set minimum similarity percentage (50-95%)
        5. Click 'Start Processing'
        
        Output will be saved in 'outputs' folder.
        Recognized faces show similarity percentage (70% = good match)
        """
        info_label = tk.Label(info_frame, text=info_text, font=('Arial', 9), 
                             bg='#2c3e50', fg='#ecf0f1', justify=tk.LEFT)
        info_label.pack(padx=10, pady=10)
        
        # Initialize statistics
        self.stats = {
            'total_faces': 0,
            'recognized_faces': 0,
            'unknown_faces': 0,
            'target_persons': 0
        }
        
    def browse_video(self):
        filetypes = [
            ('Video files', '*.mp4 *.avi *.mov *.mkv'),
            ('All files', '*.*')
        ]
        filename = filedialog.askopenfilename(filetypes=filetypes)
        if filename:
            self.video_path = filename
            self.video_path_var.set(Path(filename).name)
            
    def browse_target_folder(self):
        folder = filedialog.askdirectory()
        if folder:
            self.target_folder = folder
            self.target_path_var.set(Path(folder).name)
            self.update_target_count()
            
    def update_target_count(self):
        """Count and display number of target persons"""
        if self.target_folder and os.path.exists(self.target_folder):
            names_set = set()
            for img_file in os.listdir(self.target_folder):
                if img_file.lower().endswith(('.png', '.jpg', '.jpeg')):
                    name_parts = os.path.splitext(img_file)[0].split('_')
                    if len(name_parts) >= 1:
                        names_set.add(name_parts[0])
            
            self.stats['target_persons'] = len(names_set)
            self.target_persons_label.config(text=f"Target Persons: {len(names_set)}")
            
    def load_target_images(self):
        """Load target images and extract face embeddings"""
        if not self.target_folder:
            messagebox.showerror("Error", "Please select target folder first!")
            return False
            
        try:
            self.face_db = {}
            self.face_embeddings = []
            self.face_names = []
            
            # Initialize face analysis
            app = FaceAnalysis(name='buffalo_l', providers=['CUDAExecutionProvider', 'CPUExecutionProvider'])
            app.prepare(ctx_id=0, det_size=(640, 640))
            
            # Process each image in target folder
            processed_count = 0
            for img_file in os.listdir(self.target_folder):
                if img_file.lower().endswith(('.png', '.jpg', '.jpeg')):
                    # Parse name and number
                    name_parts = os.path.splitext(img_file)[0].split('_')
                    if len(name_parts) >= 2:
                        person_name = name_parts[0]
                        img_path = os.path.join(self.target_folder, img_file)
                        
                        # Load image
                        img = cv2.imread(img_path)
                        if img is None:
                            continue
                            
                        # Detect faces
                        faces = app.get(img)
                        if len(faces) > 0:
                            # Get the largest face (assuming it's the target person)
                            face = max(faces, key=lambda x: (x.bbox[2]-x.bbox[0])*(x.bbox[3]-x.bbox[1]))
                            embedding = face.normed_embedding
                            
                            # Store in database
                            if person_name not in self.face_db:
                                self.face_db[person_name] = []
                            self.face_db[person_name].append(embedding)
                            processed_count += 1
            
            # Create average embeddings for each person
            for name, embeddings in self.face_db.items():
                if embeddings:
                    avg_embedding = np.mean(embeddings, axis=0)
                    self.face_embeddings.append(avg_embedding)
                    self.face_names.append(name)
                    
            self.face_embeddings = np.array(self.face_embeddings)
            
            self.update_status(f"Loaded {processed_count} images for {len(self.face_names)} persons")
            return len(self.face_names) > 0
            
        except Exception as e:
            messagebox.showerror("Error", f"Failed to load target images: {str(e)}")
            return False
            
    def recognize_face(self, embedding, threshold_percent=70):
        """Recognize face using cosine similarity and return percentage"""
        if len(self.face_embeddings) == 0:
            return None, 0
            
        # Calculate cosine similarities
        embedding = embedding / np.linalg.norm(embedding)
        similarities = np.dot(self.face_embeddings, embedding)
        
        # Find best match
        best_idx = np.argmax(similarities)
        best_similarity = similarities[best_idx]
        
        # Convert similarity to percentage (0-100%)
        similarity_percent = (best_similarity + 1) * 50  # Convert from [-1,1] to [0,100]
        
        if similarity_percent >= threshold_percent:
            return self.face_names[best_idx], similarity_percent
        return None, similarity_percent
        
    def process_video(self):
        """Main video processing function"""
        try:
            # Load target images
            self.update_status("Loading target images...")
            if not self.load_target_images():
                return
                
            # Initialize models
            self.update_status("Initializing face detector...")
            
            # Initialize face detector
            self.detector = FaceAnalysis(name='buffalo_l', providers=['CUDAExecutionProvider', 'CPUExecutionProvider'])
            self.detector.prepare(ctx_id=0, det_size=(640, 640))
            
            # Open video
            self.cap = cv2.VideoCapture(self.video_path)
            if not self.cap.isOpened():
                messagebox.showerror("Error", "Failed to open video file!")
                return
                
            # Get video properties
            fps = int(self.cap.get(cv2.CAP_PROP_FPS))
            width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            total_frames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
            
            # Create output directory
            output_dir = "outputs"
            os.makedirs(output_dir, exist_ok=True)
            
            # Create output video path
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            video_name = Path(self.video_path).stem
            self.output_video_path = os.path.join(output_dir, f"{video_name}_processed_{timestamp}.mp4")
            
            # Initialize video writer
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            self.video_writer = cv2.VideoWriter(self.output_video_path, fourcc, fps, (width, height))
            
            frame_count = 0
            detection_threshold = self.conf_var.get()
            similarity_threshold = self.threshold_var.get()
            
            # Reset statistics
            self.stats = {
                'total_faces': 0,
                'recognized_faces': 0,
                'unknown_faces': 0,
                'target_persons': len(self.face_names)
            }
            
            self.update_status("Processing video...")
            
            while self.processing and self.cap.isOpened():
                ret, frame = self.cap.read()
                if not ret:
                    break
                    
                # Process frame
                processed_frame = self.process_frame(frame, detection_threshold, similarity_threshold)
                
                # Write to output video
                self.video_writer.write(processed_frame)
                
                # Update display
                if frame_count % 3 == 0:  # Update display every 3 frames for performance
                    self.update_display(processed_frame)
                    
                # Update progress
                frame_count += 1
                progress = (frame_count / total_frames) * 100
                self.update_progress(progress, f"Processing frame {frame_count}/{total_frames}")
                
                # Update statistics every 10 frames
                if frame_count % 10 == 0:
                    self.update_statistics()
                
            # Update final statistics
            self.update_statistics()
            
            # Cleanup
            if self.cap:
                self.cap.release()
            if self.video_writer:
                self.video_writer.release()
                
            if self.processing:  # Only show success if not stopped
                self.update_status("Processing completed!")
                self.update_progress(100, "Completed!")
                messagebox.showinfo("Success", 
                    f"Video processing completed!\n\n"
                    f"Statistics:\n"
                    f"- Total faces detected: {self.stats['total_faces']}\n"
                    f"- Recognized faces: {self.stats['recognized_faces']}\n"
                    f"- Unknown faces: {self.stats['unknown_faces']}\n"
                    f"- Target persons: {self.stats['target_persons']}\n\n"
                    f"Output saved to:\n{self.output_video_path}")
                
        except Exception as e:
            self.update_status(f"Error: {str(e)}")
            messagebox.showerror("Error", f"Processing failed: {str(e)}")
        finally:
            self.processing = False
            self.process_btn.config(state=tk.NORMAL)
            self.stop_btn.config(state=tk.DISABLED)
            
    def process_frame(self, frame, detection_threshold, similarity_threshold):
        """Process a single frame for face detection and recognition"""
        # Create a copy for drawing
        output_frame = frame.copy()
        
        try:
            # Detect faces
            faces = self.detector.get(frame)
            
            for face in faces:
                # Check detection confidence
                if face.det_score < detection_threshold:
                    continue
                    
                # Get bounding box
                bbox = face.bbox.astype(int)
                x1, y1, x2, y2 = bbox
                
                # Get face embedding
                embedding = face.normed_embedding
                
                # Recognize face and get percentage
                name, similarity_percent = self.recognize_face(embedding, similarity_threshold)
                
                # Update statistics
                self.stats['total_faces'] += 1
                if name:
                    self.stats['recognized_faces'] += 1
                else:
                    self.stats['unknown_faces'] += 1
                
                # Choose color based on recognition
                if name:
                    color = (0, 255, 0)  # Green for recognized
                else:
                    color = (0, 0, 255)  # Red for unknown
                
                # Draw rectangle
                cv2.rectangle(output_frame, (x1, y1), (x2, y2), color, 2)
                
                # Prepare label with percentage
                if name:
                    label = f"{name}: {similarity_percent:.1f}%"
                else:
                    label = f"Unknown: {similarity_percent:.1f}%"
                
                # Draw label background
                label_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)[0]
                cv2.rectangle(output_frame, (x1, y1 - label_size[1] - 10), 
                             (x1 + label_size[0], y1), color, -1)
                
                # Draw label text
                cv2.putText(output_frame, label, (x1, y1 - 5), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
                
                # Draw detection confidence (smaller, above the main label)
                det_label = f"Det: {face.det_score:.2f}"
                det_size = cv2.getTextSize(det_label, cv2.FONT_HERSHEY_SIMPLEX, 0.4, 1)[0]
                cv2.putText(output_frame, det_label, (x1, y1 - label_size[1] - 15), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.4, (200, 200, 200), 1)
                           
        except Exception as e:
            print(f"Error processing frame: {e}")
            
        return output_frame
        
    def update_display(self, frame):
        """Update the video display in GUI"""
        try:
            # Resize for display
            display_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            h, w = display_frame.shape[:2]
            aspect_ratio = w / h
            
            # Calculate new dimensions
            max_width = self.video_label.winfo_width() or 800
            max_height = self.video_label.winfo_height() or 600
            
            if w > max_width:
                w = max_width
                h = int(w / aspect_ratio)
                
            if h > max_height:
                h = max_height
                w = int(h * aspect_ratio)
                
            if w > 0 and h > 0:
                display_frame = cv2.resize(display_frame, (w, h))
                
                # Convert to ImageTk
                image = Image.fromarray(display_frame)
                photo = ImageTk.PhotoImage(image=image)
                
                # Update label
                self.video_label.config(image=photo)
                self.video_label.image = photo
                
        except Exception as e:
            print(f"Error updating display: {e}")
            
    def update_statistics(self):
        """Update statistics display"""
        def _update():
            self.total_faces_label.config(text=f"Total Faces Detected: {self.stats['total_faces']}")
            self.recognized_faces_label.config(text=f"Recognized: {self.stats['recognized_faces']}")
            self.unknown_faces_label.config(text=f"Unknown: {self.stats['unknown_faces']}")
            self.target_persons_label.config(text=f"Target Persons: {self.stats['target_persons']}")
        self.root.after(0, _update)
        
    def update_status(self, message):
        """Update status label"""
        def _update():
            self.status_label.config(text=f"Status: {message}")
        self.root.after(0, _update)
        
    def update_progress(self, value, message=None):
        """Update progress bar and label"""
        def _update():
            self.progress_bar['value'] = value
            if message:
                self.progress_label.config(text=message)
        self.root.after(0, _update)
        
    def start_processing(self):
        """Start video processing thread"""
        if not self.video_path:
            messagebox.showerror("Error", "Please select a video file!")
            return
            
        if not self.target_folder:
            messagebox.showerror("Error", "Please select target folder!")
            return
            
        self.processing = True
        self.process_btn.config(state=tk.DISABLED)
        self.stop_btn.config(state=tk.NORMAL)
        
        # Reset progress
        self.update_progress(0, "Starting...")
        
        # Start processing in separate thread
        thread = threading.Thread(target=self.process_video, daemon=True)
        thread.start()
        
    def stop_processing(self):
        """Stop video processing"""
        self.processing = False
        self.update_status("Stopping...")
        
    def on_closing(self):
        """Cleanup on window close"""
        self.processing = False
        if self.cap:
            self.cap.release()
        if self.video_writer:
            self.video_writer.release()
        self.root.destroy()

def main():
    root = tk.Tk()
    app = FaceRecognitionApp(root)
    root.protocol("WM_DELETE_WINDOW", app.on_closing)
    root.mainloop()

if __name__ == "__main__":
    main()