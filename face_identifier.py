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
from ultralytics import YOLO
import warnings
warnings.filterwarnings('ignore')

class FaceRecognitionApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Face Recognition System - YOLO + ArcFace")
        self.root.geometry("1400x800")
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
        
        # Models
        self.yolo_model = None  # YOLO for face detection
        self.arcface_model = None  # ArcFace for face recognition
        
        # Setup GUI
        self.setup_gui()
        
        # Queue for thread-safe GUI updates
        self.queue = queue.Queue()
        
    def setup_gui(self):
        # Create main container with proper layout
        main_container = tk.PanedWindow(self.root, orient=tk.HORIZONTAL, bg='#2c3e50', sashwidth=5)
        main_container.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        # Left Panel - Controls
        left_panel = tk.Frame(main_container, bg='#34495e', width=400)
        main_container.add(left_panel, minsize=400, stretch='never')
        
        # Right Panel - Video Display
        right_panel = tk.Frame(main_container, bg='#2c3e50')
        main_container.add(right_panel, minsize=800)
        
        # ========== LEFT PANEL CONTENT ==========
        
        # Title
        title_frame = tk.Frame(left_panel, bg='#34495e')
        title_frame.pack(fill=tk.X, padx=20, pady=20)
        
        title_label = tk.Label(title_frame, text="Face Recognition System", 
                              font=('Arial', 20, 'bold'), bg='#34495e', fg='#ecf0f1')
        title_label.pack()
        
        subtitle_label = tk.Label(title_frame, text="YOLO (Detection) + ArcFace (Recognition)", 
                                 font=('Arial', 12), bg='#34495e', fg='#bdc3c7')
        subtitle_label.pack(pady=(5, 0))
        
        # Separator
        ttk.Separator(left_panel, orient=tk.HORIZONTAL).pack(fill=tk.X, padx=20, pady=10)
        
        # File Selection Section
        file_section = tk.LabelFrame(left_panel, text=" File Selection ", font=('Arial', 12, 'bold'),
                                    bg='#34495e', fg='#ecf0f1', relief=tk.GROOVE, borderwidth=2)
        file_section.pack(fill=tk.X, padx=20, pady=10)
        
        # Video Selection
        video_frame = tk.Frame(file_section, bg='#34495e')
        video_frame.pack(fill=tk.X, padx=15, pady=10)
        
        video_label = tk.Label(video_frame, text="Video File:", 
                              font=('Arial', 11), bg='#34495e', fg='#bdc3c7', anchor=tk.W)
        video_label.pack(fill=tk.X)
        
        video_browse_frame = tk.Frame(video_frame, bg='#34495e')
        video_browse_frame.pack(fill=tk.X, pady=(5, 0))
        
        self.video_path_var = tk.StringVar()
        video_entry = tk.Entry(video_browse_frame, textvariable=self.video_path_var, 
                              state='readonly', font=('Arial', 10), width=30)
        video_entry.pack(side=tk.LEFT, fill=tk.X, expand=True)
        
        browse_video_btn = tk.Button(video_browse_frame, text="Browse", command=self.browse_video,
                                    bg='#3498db', fg='white', font=('Arial', 10, 'bold'),
                                    relief=tk.RAISED, width=10)
        browse_video_btn.pack(side=tk.RIGHT, padx=(5, 0))
        
        # Target Images Folder
        target_frame = tk.Frame(file_section, bg='#34495e')
        target_frame.pack(fill=tk.X, padx=15, pady=(0, 10))
        
        target_label = tk.Label(target_frame, text="Target Images Folder:", 
                               font=('Arial', 11), bg='#34495e', fg='#bdc3c7', anchor=tk.W)
        target_label.pack(fill=tk.X)
        
        target_browse_frame = tk.Frame(target_frame, bg='#34495e')
        target_browse_frame.pack(fill=tk.X, pady=(5, 0))
        
        self.target_path_var = tk.StringVar()
        target_entry = tk.Entry(target_browse_frame, textvariable=self.target_path_var, 
                               state='readonly', font=('Arial', 10), width=30)
        target_entry.pack(side=tk.LEFT, fill=tk.X, expand=True)
        
        browse_target_btn = tk.Button(target_browse_frame, text="Browse", command=self.browse_target_folder,
                                     bg='#3498db', fg='white', font=('Arial', 10, 'bold'),
                                     relief=tk.RAISED, width=10)
        browse_target_btn.pack(side=tk.RIGHT, padx=(5, 0))
        
        # Settings Section
        settings_section = tk.LabelFrame(left_panel, text=" Settings ", font=('Arial', 12, 'bold'),
                                        bg='#34495e', fg='#ecf0f1', relief=tk.GROOVE, borderwidth=2)
        settings_section.pack(fill=tk.X, padx=20, pady=10)
        
        # Similarity Threshold Control
        threshold_frame = tk.Frame(settings_section, bg='#34495e')
        threshold_frame.pack(fill=tk.X, padx=15, pady=10)
        
        threshold_label = tk.Label(threshold_frame, text="Recognition Similarity (%):", 
                                  font=('Arial', 11), bg='#34495e', fg='#bdc3c7', anchor=tk.W)
        threshold_label.pack(fill=tk.X)
        
        self.threshold_var = tk.IntVar(value=70)
        threshold_scale_frame = tk.Frame(threshold_frame, bg='#34495e')
        threshold_scale_frame.pack(fill=tk.X, pady=(5, 0))
        
        threshold_scale = tk.Scale(threshold_scale_frame, from_=50, to=95, resolution=5,
                                  orient=tk.HORIZONTAL, variable=self.threshold_var,
                                  bg='#34495e', fg='white', troughcolor='#2c3e50',
                                  length=280, sliderrelief=tk.RAISED)
        threshold_scale.pack(fill=tk.X)
        
        self.threshold_value_label = tk.Label(threshold_frame, 
                                             text=f"Current: {self.threshold_var.get()}%",
                                             font=('Arial', 10), bg='#34495e', fg='#ecf0f1')
        self.threshold_value_label.pack(pady=(5, 0))
        
        # Update threshold label when scale changes
        self.threshold_var.trace('w', self.update_threshold_label)
        
        # YOLO Detection Confidence
        conf_frame = tk.Frame(settings_section, bg='#34495e')
        conf_frame.pack(fill=tk.X, padx=15, pady=(0, 10))
        
        conf_label = tk.Label(conf_frame, text="YOLO Detection Confidence:", 
                             font=('Arial', 11), bg='#34495e', fg='#bdc3c7', anchor=tk.W)
        conf_label.pack(fill=tk.X)
        
        conf_scale_frame = tk.Frame(conf_frame, bg='#34495e')
        conf_scale_frame.pack(fill=tk.X, pady=(5, 0))
        
        self.conf_var = tk.DoubleVar(value=0.5)
        conf_scale = tk.Scale(conf_scale_frame, from_=0.1, to=1.0, resolution=0.05,
                             orient=tk.HORIZONTAL, variable=self.conf_var,
                             bg='#34495e', fg='white', troughcolor='#2c3e50',
                             length=280, sliderrelief=tk.RAISED)
        conf_scale.pack(fill=tk.X)
        
        self.conf_value_label = tk.Label(conf_frame, 
                                        text=f"Current: {self.conf_var.get():.2f}",
                                        font=('Arial', 10), bg='#34495e', fg='#ecf0f1')
        self.conf_value_label.pack(pady=(5, 0))
        
        # Update confidence label when scale changes
        self.conf_var.trace('w', self.update_confidence_label)
        
        # YOLO Model Selection
        model_frame = tk.Frame(settings_section, bg='#34495e')
        model_frame.pack(fill=tk.X, padx=15, pady=(0, 10))
        
        model_label = tk.Label(model_frame, text="YOLO Model Size:", 
                              font=('Arial', 11), bg='#34495e', fg='#bdc3c7', anchor=tk.W)
        model_label.pack(fill=tk.X)
        
        self.model_var = tk.StringVar(value="yolov8n-face.pt")
        model_options = ["yolov8n-face.pt", "yolov8s-face.pt", "yolov8m-face.pt", "yolov8l-face.pt"]
        model_combo = ttk.Combobox(model_frame, textvariable=self.model_var, 
                                  values=model_options, state="readonly",
                                  font=('Arial', 10))
        model_combo.pack(fill=tk.X, pady=(5, 0))
        
        # Control Buttons Section
        button_section = tk.Frame(left_panel, bg='#34495e')
        button_section.pack(fill=tk.X, padx=20, pady=20)
        
        self.process_btn = tk.Button(button_section, text="▶ START PROCESSING", command=self.start_processing,
                                    bg='#2ecc71', fg='white', font=('Arial', 14, 'bold'),
                                    height=2, width=20, relief=tk.RAISED, bd=3)
        self.process_btn.pack(pady=(0, 10))
        
        self.stop_btn = tk.Button(button_section, text="■ STOP", command=self.stop_processing,
                                 bg='#e74c3c', fg='white', font=('Arial', 14, 'bold'),
                                 height=2, width=20, relief=tk.RAISED, bd=3, state=tk.DISABLED)
        self.stop_btn.pack()
        
        # Progress Section
        progress_section = tk.LabelFrame(left_panel, text=" Progress ", font=('Arial', 12, 'bold'),
                                        bg='#34495e', fg='#ecf0f1', relief=tk.GROOVE, borderwidth=2)
        progress_section.pack(fill=tk.X, padx=20, pady=10)
        
        # Progress Label
        self.progress_label = tk.Label(progress_section, text="Ready to process", 
                                      font=('Arial', 11), bg='#34495e', fg='#ecf0f1', anchor=tk.W)
        self.progress_label.pack(fill=tk.X, padx=15, pady=(10, 5))
        
        # Progress Bar with percentage
        progress_bar_frame = tk.Frame(progress_section, bg='#34495e')
        progress_bar_frame.pack(fill=tk.X, padx=15, pady=(0, 10))
        
        self.progress_bar = ttk.Progressbar(progress_bar_frame, length=320, mode='determinate',
                                           style="green.Horizontal.TProgressbar")
        self.progress_bar.pack(fill=tk.X)
        
        self.progress_percentage = tk.Label(progress_bar_frame, text="0%", 
                                           font=('Arial', 10), bg='#34495e', fg='#ecf0f1')
        self.progress_percentage.pack(pady=(5, 0))
        
        # Status Section
        status_section = tk.LabelFrame(left_panel, text=" Status ", font=('Arial', 12, 'bold'),
                                      bg='#34495e', fg='#ecf0f1', relief=tk.GROOVE, borderwidth=2)
        status_section.pack(fill=tk.X, padx=20, pady=(10, 20))
        
        self.status_label = tk.Label(status_section, text="Idle", 
                                    font=('Arial', 11), bg='#34495e', fg='#ecf0f1', anchor=tk.W)
        self.status_label.pack(fill=tk.X, padx=15, pady=10)
        
        # ========== RIGHT PANEL CONTENT ==========
        
        # Video Display Frame
        display_container = tk.Frame(right_panel, bg='#1a252f')
        display_container.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # Video display with border
        display_border = tk.Frame(display_container, bg='#3498db', bd=2, relief=tk.SUNKEN)
        display_border.pack(fill=tk.BOTH, expand=True)
        
        self.video_label = tk.Label(display_border, bg='#1a252f')
        self.video_label.pack(fill=tk.BOTH, expand=True, padx=1, pady=1)
        
        # Default placeholder image
        self.show_placeholder()
        
        # Statistics Panel
        stats_panel = tk.Frame(right_panel, bg='#34495e', height=120)
        stats_panel.pack(fill=tk.X, padx=10, pady=(0, 10))
        stats_panel.pack_propagate(False)
        
        # Statistics Title
        stats_title = tk.Label(stats_panel, text="Processing Statistics", 
                              font=('Arial', 14, 'bold'), bg='#34495e', fg='#ecf0f1')
        stats_title.pack(pady=(10, 15))
        
        # Statistics Grid
        stats_grid = tk.Frame(stats_panel, bg='#34495e')
        stats_grid.pack(expand=True)
        
        # Row 1
        self.total_faces_label = tk.Label(stats_grid, text="Total Faces: 0", 
                                         font=('Arial', 12, 'bold'), bg='#34495e', fg='#3498db',
                                         width=20)
        self.total_faces_label.grid(row=0, column=0, padx=20, pady=5)
        
        self.recognized_faces_label = tk.Label(stats_grid, text="Recognized: 0", 
                                              font=('Arial', 12, 'bold'), bg='#34495e', fg='#2ecc71',
                                              width=20)
        self.recognized_faces_label.grid(row=0, column=1, padx=20, pady=5)
        
        # Row 2
        self.unknown_faces_label = tk.Label(stats_grid, text="Unknown: 0", 
                                           font=('Arial', 12, 'bold'), bg='#34495e', fg='#e74c3c',
                                           width=20)
        self.unknown_faces_label.grid(row=1, column=0, padx=20, pady=5)
        
        self.target_persons_label = tk.Label(stats_grid, text="Target Persons: 0", 
                                            font=('Arial', 12, 'bold'), bg='#34495e', fg='#9b59b6',
                                            width=20)
        self.target_persons_label.grid(row=1, column=1, padx=20, pady=5)
        
        # Model Info Panel
        model_info_panel = tk.Frame(right_panel, bg='#2c3e50')
        model_info_panel.pack(fill=tk.X, padx=10, pady=(0, 10))
        
        model_info_text = """
        MODEL ARCHITECTURE:
        • Face Detection: YOLOv8 (Real-time object detection)
        • Face Recognition: ArcFace (High-accuracy face recognition)
        
        YOLO Models Available:
        • yolov8n-face.pt: Fastest, lower accuracy
        • yolov8s-face.pt: Balanced speed/accuracy
        • yolov8m-face.pt: More accurate, slower
        • yolov8l-face.pt: Most accurate, slowest
        
        Similarity Threshold:
        • 70-80%: Good match
        • 80-90%: Very confident match
        • 90%+: Excellent match
        """
        
        model_info_label = tk.Label(model_info_panel, text=model_info_text, 
                                   font=('Arial', 9), bg='#2c3e50', fg='#ecf0f1', 
                                   justify=tk.LEFT, anchor=tk.W)
        model_info_label.pack(fill=tk.X, padx=10, pady=10)
        
        # Initialize statistics
        self.stats = {
            'total_faces': 0,
            'recognized_faces': 0,
            'unknown_faces': 0,
            'target_persons': 0
        }
        
        # Configure ttk style for progress bar
        style = ttk.Style()
        style.theme_use('default')
        style.configure("green.Horizontal.TProgressbar", 
                       background='#2ecc71',
                       troughcolor='#2c3e50',
                       bordercolor='#34495e',
                       lightcolor='#2ecc71',
                       darkcolor='#2ecc71')
        
    def show_placeholder(self):
        """Show a placeholder when no video is loaded"""
        placeholder = tk.Label(self.video_label, text="Video Preview\n\nSelect a video file to begin",
                              font=('Arial', 16), bg='#1a252f', fg='#7f8c8d')
        placeholder.place(relx=0.5, rely=0.5, anchor=tk.CENTER)
        
    def update_threshold_label(self, *args):
        """Update threshold label when scale changes"""
        self.threshold_value_label.config(text=f"Current: {self.threshold_var.get()}%")
        
    def update_confidence_label(self, *args):
        """Update confidence label when scale changes"""
        self.conf_value_label.config(text=f"Current: {self.conf_var.get():.2f}")
        
    def browse_video(self):
        filetypes = [
            ('Video files', '*.mp4 *.avi *.mov *.mkv *.flv *.wmv'),
            ('All files', '*.*')
        ]
        filename = filedialog.askopenfilename(filetypes=filetypes)
        if filename:
            self.video_path = filename
            self.video_path_var.set(Path(filename).name)
            self.status_label.config(text=f"Video loaded: {Path(filename).name}")
            
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
            image_count = 0
            for img_file in os.listdir(self.target_folder):
                if img_file.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp')):
                    name_parts = os.path.splitext(img_file)[0].split('_')
                    if len(name_parts) >= 1:
                        names_set.add(name_parts[0])
                        image_count += 1
            
            self.stats['target_persons'] = len(names_set)
            self.target_persons_label.config(text=f"Target Persons: {len(names_set)}")
            self.status_label.config(text=f"Loaded {image_count} images for {len(names_set)} persons")
            
    def load_target_images(self):
        """Load target images and extract face embeddings using ArcFace"""
        if not self.target_folder:
            messagebox.showerror("Error", "Please select target folder first!")
            return False
            
        try:
            self.face_db = {}
            self.face_embeddings = []
            self.face_names = []
            
            # Initialize ArcFace model for recognition
            self.arcface_model = FaceAnalysis(name='buffalo_l', providers=['CUDAExecutionProvider', 'CPUExecutionProvider'])
            self.arcface_model.prepare(ctx_id=0, det_size=(640, 640))
            
            # Process each image in target folder
            processed_count = 0
            for img_file in os.listdir(self.target_folder):
                if img_file.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp')):
                    # Parse name and number
                    name_parts = os.path.splitext(img_file)[0].split('_')
                    if len(name_parts) >= 2:
                        person_name = name_parts[0]
                        img_path = os.path.join(self.target_folder, img_file)
                        
                        # Load image
                        img = cv2.imread(img_path)
                        if img is None:
                            continue
                            
                        # Detect faces using ArcFace detector (for training only)
                        faces = self.arcface_model.get(img)
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
            
            self.update_status(f"ArcFace: Loaded {processed_count} images for {len(self.face_names)} persons")
            return len(self.face_names) > 0
            
        except Exception as e:
            messagebox.showerror("Error", f"Failed to load target images: {str(e)}")
            return False
            
    def recognize_face(self, embedding, threshold_percent=70):
        """Recognize face using ArcFace cosine similarity and return percentage"""
        if len(self.face_embeddings) == 0:
            return None, 0
            
        # Calculate cosine similarities using ArcFace embeddings
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
        """Main video processing function using YOLO for detection and ArcFace for recognition"""
        try:
            # Load target images with ArcFace
            self.update_status("Loading target images with ArcFace...")
            self.update_progress(5, "Loading target images...")
            if not self.load_target_images():
                return
                
            # Initialize YOLO model for face detection
            self.update_status(f"Loading YOLO model: {self.model_var.get()}...")
            self.update_progress(15, f"Loading YOLO {self.model_var.get()}...")
            
            try:
                # Try to load YOLO face detection model
                self.yolo_model = YOLO(self.model_var.get())
            except:
                # If model doesn't exist, download it
                self.update_status("Downloading YOLO face model...")
                model_name = self.model_var.get().replace('.pt', '')
                self.yolo_model = YOLO(model_name)
            
            # Initialize ArcFace model for recognition
            self.update_status("Initializing ArcFace model...")
            self.update_progress(25, "Initializing ArcFace...")
            self.arcface_model = FaceAnalysis(name='buffalo_l', providers=['CUDAExecutionProvider', 'CPUExecutionProvider'])
            self.arcface_model.prepare(ctx_id=0, det_size=(640, 640))
            
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
            
            self.update_status("Processing video with YOLO + ArcFace...")
            self.update_progress(30, "Starting video processing...")
            
            # Model information for display
            model_info = f"YOLO: {self.model_var.get()} | ArcFace | Similarity: {similarity_threshold}%"
            
            while self.processing and self.cap.isOpened():
                ret, frame = self.cap.read()
                if not ret:
                    break
                    
                # Process frame with YOLO + ArcFace
                processed_frame = self.process_frame(frame, detection_threshold, similarity_threshold)
                
                # Add model info to frame
                cv2.putText(processed_frame, model_info, (10, 30), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
                
                # Write to output video
                self.video_writer.write(processed_frame)
                
                # Update display
                if frame_count % 3 == 0:  # Update display every 3 frames for performance
                    self.update_display(processed_frame)
                    
                # Update progress
                frame_count += 1
                progress = 30 + (frame_count / total_frames) * 65  # Scale from 30% to 95%
                self.update_progress(progress, f"Processing frame {frame_count}/{total_frames}")
                
                # Update statistics every 10 frames
                if frame_count % 10 == 0:
                    self.update_statistics()
                
            # Update final progress and statistics
            self.update_progress(100, "Processing complete!")
            self.update_statistics()
            
            # Cleanup
            if self.cap:
                self.cap.release()
            if self.video_writer:
                self.video_writer.release()
                
            if self.processing:  # Only show success if not stopped
                self.update_status("Processing completed successfully!")
                messagebox.showinfo("Success", 
                    f"✅ Video processing completed!\n\n"
                    f"📊 Statistics:\n"
                    f"• Total faces detected: {self.stats['total_faces']}\n"
                    f"• Recognized faces: {self.stats['recognized_faces']}\n"
                    f"• Unknown faces: {self.stats['unknown_faces']}\n"
                    f"• Target persons: {self.stats['target_persons']}\n\n"
                    f"🔧 Models used:\n"
                    f"• Detection: YOLO ({self.model_var.get()})\n"
                    f"• Recognition: ArcFace\n\n"
                    f"💾 Output saved to:\n{self.output_video_path}")
                
        except Exception as e:
            self.update_status(f"Error: {str(e)}")
            self.update_progress(0, f"Error: {str(e)}")
            messagebox.showerror("Error", f"Processing failed: {str(e)}")
        finally:
            self.processing = False
            self.process_btn.config(state=tk.NORMAL)
            self.stop_btn.config(state=tk.DISABLED)
            
    def process_frame(self, frame, detection_threshold, similarity_threshold):
        """Process a single frame using YOLO for detection and ArcFace for recognition"""
        # Create a copy for drawing
        output_frame = frame.copy()
        
        try:
            # STEP 1: Detect faces using YOLO
            yolo_results = self.yolo_model(frame, conf=detection_threshold, verbose=False)
            
            # Extract face regions detected by YOLO
            face_regions = []
            for result in yolo_results:
                boxes = result.boxes
                if boxes is not None:
                    for box in boxes:
                        x1, y1, x2, y2 = box.xyxy[0].cpu().numpy().astype(int)
                        confidence = box.conf[0].cpu().numpy()
                        
                        # Only consider detections with sufficient confidence
                        if confidence >= detection_threshold:
                            face_regions.append({
                                'bbox': [x1, y1, x2, y2],
                                'confidence': confidence,
                                'region': frame[y1:y2, x1:x2] if y2 > y1 and x2 > x1 else None
                            })
            
            # STEP 2: Recognize each face using ArcFace
            for face in face_regions:
                x1, y1, x2, y2 = face['bbox']
                
                # Extract face region for recognition
                face_region = face['region']
                if face_region is None or face_region.size == 0:
                    continue
                
                # Get ArcFace embedding for this face
                arcface_faces = self.arcface_model.get(face_region)
                if len(arcface_faces) == 0:
                    continue
                
                # Get the main face embedding
                arcface_face = arcface_faces[0]
                embedding = arcface_face.normed_embedding
                
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
                
                # Draw YOLO detection rectangle
                cv2.rectangle(output_frame, (x1, y1), (x2, y2), color, 2)
                
                # Prepare label with YOLO confidence and ArcFace similarity
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
                
                # Draw YOLO detection confidence
                det_label = f"YOLO: {face['confidence']:.2f}"
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
                
                # Clear placeholder
                for widget in self.video_label.winfo_children():
                    widget.destroy()
                
                # Update label
                self.video_label.config(image=photo)
                self.video_label.image = photo
                
        except Exception as e:
            print(f"Error updating display: {e}")
            
    def update_statistics(self):
        """Update statistics display"""
        def _update():
            self.total_faces_label.config(text=f"Total Faces: {self.stats['total_faces']}")
            self.recognized_faces_label.config(text=f"Recognized: {self.stats['recognized_faces']}")
            self.unknown_faces_label.config(text=f"Unknown: {self.stats['unknown_faces']}")
            self.target_persons_label.config(text=f"Target Persons: {self.stats['target_persons']}")
        self.root.after(0, _update)
        
    def update_status(self, message):
        """Update status label"""
        def _update():
            self.status_label.config(text=message)
        self.root.after(0, _update)
        
    def update_progress(self, value, message=None):
        """Update progress bar and label"""
        def _update():
            self.progress_bar['value'] = value
            self.progress_percentage.config(text=f"{value:.1f}%")
            if message:
                self.progress_label.config(text=message)
        self.root.after(0, _update)
        
    def start_processing(self):
        """Start video processing thread"""
        if not self.video_path:
            messagebox.showerror("Error", "Please select a video file first!")
            return
            
        if not self.target_folder:
            messagebox.showerror("Error", "Please select target folder first!")
            return
            
        # Check if target folder has images
        if not any(f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp')) for f in os.listdir(self.target_folder)):
            messagebox.showerror("Error", "Target folder doesn't contain any valid images!")
            return
            
        self.processing = True
        self.process_btn.config(state=tk.DISABLED)
        self.stop_btn.config(state=tk.NORMAL)
        
        # Reset progress and statistics
        self.update_progress(0, "Initializing YOLO + ArcFace...")
        self.stats = {'total_faces': 0, 'recognized_faces': 0, 'unknown_faces': 0, 'target_persons': 0}
        self.update_statistics()
        
        # Start processing in separate thread
        thread = threading.Thread(target=self.process_video, daemon=True)
        thread.start()
        
    def stop_processing(self):
        """Stop video processing"""
        self.processing = False
        self.update_status("Stopping process...")
        
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