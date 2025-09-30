import os
import numpy as np
import tkinter as tk
from tkinter import ttk, messagebox, filedialog
from PIL import Image, ImageTk, ImageDraw
import matplotlib.cm as cm
import cv2
import csv
import time
import matplotlib.pyplot as plt

class SpectrogramYOLOLabeler:
    def __init__(self, root):
        self.root = root
        self.root.title("Drone RF Signal YOLO Labeler")
        
        # Start in fullscreen mode
        self.root.attributes('-fullscreen', True)
        self.fullscreen = True
        
        # Store screen dimensions for responsive layout
        self.screen_width = self.root.winfo_screenwidth()
        self.screen_height = self.root.winfo_screenheight()
        
        self.root.configure(bg="#2c3e50")
        
        # Set style for modern look
        self.setup_styles()
        
        # Create main frame
        self.main_frame = ttk.Frame(root, padding="10", style="Main.TFrame")
        self.main_frame.pack(fill=tk.BOTH, expand=True)
        
        # Header with exit fullscreen button
        header_frame = ttk.Frame(self.main_frame, style="Header.TFrame")
        header_frame.pack(fill=tk.X, pady=(0, 5))
        
        title_label = ttk.Label(
            header_frame, 
            text="Drone RF Signal YOLO Labeler", 
            font=("Arial", 20, "bold"),
            foreground="#ecf0f1",
            style="Header.TLabel"
        )
        title_label.pack(side=tk.LEFT, pady=5)
        
        # Exit fullscreen button
        self.fullscreen_btn = tk.Button(
            header_frame,
            text="Exit Fullscreen (F11)",
            font=("Arial", 9, "bold"),
            bg="#e67e22",
            fg="white",
            activebackground="#d35400",
            relief="raised",
            bd=2,
            cursor="hand2",
            command=self.toggle_fullscreen
        )
        self.fullscreen_btn.pack(side=tk.RIGHT, padx=5)
        
        # Status bar
        self.status_var = tk.StringVar()
        self.status_var.set("Ready to label")
        status_label = ttk.Label(
            self.main_frame, 
            textvariable=self.status_var, 
            font=("Arial", 10, "bold"),
            foreground="#bdc3c7",
            style="Status.TLabel"
        )
        status_label.pack(pady=(0, 5))
        
        # Progress bar
        self.progress_var = tk.DoubleVar()
        self.progress_bar = ttk.Progressbar(
            self.main_frame, 
            variable=self.progress_var, 
            maximum=100,
            mode='determinate',
            style="Custom.Horizontal.TProgressbar"
        )
        self.progress_bar.pack(fill=tk.X, pady=(0, 10))
        
        # Main content area - using paned window for resizable split
        self.content_paned = ttk.PanedWindow(self.main_frame, orient=tk.HORIZONTAL, style="Main.TFrame")
        self.content_paned.pack(fill=tk.BOTH, expand=True, pady=5)
        
        # Image display area (left side) - now takes 65% of space
        image_container = ttk.Frame(self.content_paned, style="ImageContainer.TFrame")
        self.content_paned.add(image_container, weight=6)  # 65% of space
        
        image_title = ttk.Label(
            image_container, 
            text="Current Spectrogram - Click and drag to draw bounding boxes", 
            font=("Arial", 12, "bold"),
            foreground="#ecf0f1",
            style="ImageTitle.TLabel"
        )
        image_title.pack(pady=(0, 5))
        
        self.image_frame = ttk.Frame(image_container, relief=tk.RAISED, borderwidth=2, style="Image.TFrame")
        self.image_frame.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        # Create a canvas for the image to allow better control
        self.image_canvas = tk.Canvas(self.image_frame, bg="#1a252f", highlightthickness=0)
        self.image_canvas.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        # Control area (right side) - now takes 35% of space
        control_frame = ttk.Frame(self.content_paned, style="ButtonsContainer.TFrame")
        self.content_paned.add(control_frame, weight=4)  # 35% of space
        
        # Class selection section
        class_title = ttk.Label(
            control_frame, 
            text="Class Selection (Press 0-9, A-N keys)", 
            font=("Arial", 11, "bold"),
            foreground="#ecf0f1",
            style="ButtonsTitle.TLabel"
        )
        class_title.pack(pady=(0, 5), fill=tk.X)
        
        # Create a frame for class buttons with 3 columns to give more horizontal space
        class_grid_frame = ttk.Frame(control_frame, style="Buttons.TFrame")
        class_grid_frame.pack(fill=tk.BOTH, expand=True, pady=2)
        
        # Configure grid for 3 columns and 8 rows (3x8 = 24 buttons)
        for i in range(3):
            class_grid_frame.columnconfigure(i, weight=1)
        for i in range(8):
            class_grid_frame.rowconfigure(i, weight=1)
        
        # Class names for drone detection
        self.CLASSES = [
            "Background, including WiFi and Bluetooth",
            "DJI Phantom 3",
            "DJI Phantom 4 Pro",
            "DJI MATRICE 200",
            "DJI MATRICE 100",
            "DJI Air 2S",
            "DJI Mini 3 Pro",
            "DJI Inspire 2",
            "DJI Mavic Pro",
            "DJI Mini 2",
            "DJI Mavic 3",
            "DJI MATRICE 300",
            "DJI Phantom 4 Pro RTK",
            "DJI MATRICE 30T",
            "DJI AVATA",
            "DJI DIY",
            "DJI MATRICE 600 Pro",
            "VBar",
            "FrSky X20",
            "Futaba T16IZ",
            "Taranis Plus",
            "RadioLink AT9S",
            "Futaba T14SG",
            "Skydroid"
        ]
        
        # Create class buttons with maximum text visibility
        self.class_vars = []
        self.class_buttons = []
        for i, class_name in enumerate(self.CLASSES):
            # Determine shortcut key
            if i < 10:
                shortcut = str(i)
            else:
                shortcut = chr(ord('A') + i - 10)
                
            var = tk.BooleanVar(value=(i == 0))
            
            # Create button text that shows as much as possible
            # Show the full name with smart line breaks
            if len(class_name) > 25:
                # Find a good break point
                words = class_name.split()
                if len(words) >= 2:
                    # Try to break after first word if it makes sense
                    first_part = words[0]
                    remaining = ' '.join(words[1:])
                    if len(first_part) > 5 and len(remaining) > 5:
                        compact_text = f"{i}: {first_part}\n{remaining}\n[{shortcut}]"
                    else:
                        # Just truncate with ellipsis
                        compact_text = f"{i}: {class_name[:30]}...\n[{shortcut}]"
                else:
                    compact_text = f"{i}: {class_name[:30]}...\n[{shortcut}]"
            else:
                compact_text = f"{i}: {class_name}\n[{shortcut}]"
            
            btn = tk.Button(
                class_grid_frame, 
                text=compact_text,
                wraplength=180,  # Much wider to show more text
                justify=tk.LEFT,  # Left justify for better readability
                height=2,
                font=("Arial", 8, "bold"),
                bg="#3498db",
                fg="white",
                activebackground="#2980b9",
                activeforeground="white",
                relief="raised",
                bd=1,
                cursor="hand2",
                command=lambda i=i: self.set_class(i)
            )
            row = i // 3  # 3 columns now for wider buttons
            col = i % 3
            btn.grid(row=row, column=col, padx=1, pady=1, sticky="nsew")
            self.class_buttons.append(btn)
            self.class_vars.append(var)
        
        # Action buttons section - compact layout
        actions_title = ttk.Label(
            control_frame, 
            text="Actions", 
            font=("Arial", 11, "bold"),
            foreground="#ecf0f1",
            style="ButtonsTitle.TLabel"
        )
        actions_title.pack(pady=(8, 3), fill=tk.X)
        
        # Action buttons in a compact grid
        actions_grid = ttk.Frame(control_frame, style="SpecialButtons.TFrame")
        actions_grid.pack(fill=tk.X, pady=3, padx=2)
        
        # Configure grid for 3 columns
        for i in range(3):
            actions_grid.columnconfigure(i, weight=1)
        
        button_style = {
            "font": ("Arial", 9, "bold"),
            "width": 14,
            "height": 2,
            "relief": "raised",
            "bd": 2,
            "cursor": "hand2"
        }
        
        # Row 1
        self.load_btn = tk.Button(
            actions_grid, 
            text="Load", 
            bg="#3498db",
            fg="white",
            activebackground="#2980b9",
            **button_style,
            command=self.load_spectrogram
        )
        self.load_btn.grid(row=0, column=0, padx=2, pady=2, sticky="nsew")
        
        self.save_btn = tk.Button(
            actions_grid, 
            text="Save (S)", 
            bg="#2ecc71",
            fg="white",
            activebackground="#27ae60",
            **button_style,
            command=self.save_labels
        )
        self.save_btn.grid(row=0, column=1, padx=2, pady=2, sticky="nsew")
        
        self.clear_btn = tk.Button(
            actions_grid, 
            text="Clear (C)", 
            bg="#e74c3c",
            fg="white",
            activebackground="#c0392b",
            **button_style,
            command=self.clear_bboxes
        )
        self.clear_btn.grid(row=0, column=2, padx=2, pady=2, sticky="nsew")
        
        # Row 2
        self.next_btn = tk.Button(
            actions_grid, 
            text="Next (N)", 
            bg="#f39c12",
            fg="white",
            activebackground="#d35400",
            **button_style,
            command=self.next_file
        )
        self.next_btn.grid(row=1, column=0, padx=2, pady=2, sticky="nsew")
        
        self.prev_btn = tk.Button(
            actions_grid, 
            text="Prev (P)", 
            bg="#9b59b6",
            fg="white",
            activebackground="#8e44ad",
            **button_style,
            command=self.prev_file
        )
        self.prev_btn.grid(row=1, column=1, padx=2, pady=2, sticky="nsew")
        
        self.help_btn = tk.Button(
            actions_grid, 
            text="Help (H)", 
            bg="#1abc9c",
            fg="white",
            activebackground="#16a085",
            **button_style,
            command=self.show_help
        )
        self.help_btn.grid(row=1, column=2, padx=2, pady=2, sticky="nsew")
        
        # Row 3
        self.quit_btn = tk.Button(
            actions_grid, 
            text="Quit (Q)", 
            bg="#2c3e50",
            fg="white",
            activebackground="#34495e",
            **button_style,
            command=self.quit_app
        )
        self.quit_btn.grid(row=2, column=0, columnspan=3, padx=2, pady=2, sticky="nsew")
        
        # Current bboxes display
        bboxes_frame = ttk.Frame(control_frame, style="SpecialButtons.TFrame")
        bboxes_frame.pack(fill=tk.BOTH, expand=True, pady=8)
        
        bboxes_title = ttk.Label(
            bboxes_frame, 
            text="Current Bounding Boxes:", 
            font=("Arial", 11, "bold"),
            foreground="#ecf0f1",
            style="ButtonsTitle.TLabel"
        )
        bboxes_title.pack(pady=(0, 3))
        
        self.bboxes_text = tk.Text(
            bboxes_frame, 
            height=6, 
            font=("Arial", 8),
            bg="#2c3e50",
            fg="#ecf0f1",
            relief="flat",
            wrap=tk.WORD
        )
        self.bboxes_text.pack(fill=tk.BOTH, expand=True, padx=3, pady=3)
        
        # Keyboard bindings
        self.root.focus_set()
        
        # Bind number keys 0-9
        for i in range(10):
            self.root.bind(str(i), lambda e, idx=i: self.set_class(idx))
        
        # Bind letter keys A-N for classes 10-23
        for i in range(10, len(self.CLASSES)):
            key = chr(ord('A') + i - 10)
            self.root.bind(key, lambda e, idx=i: self.set_class(idx))
            self.root.bind(key.lower(), lambda e, idx=i: self.set_class(idx))
        
        # Special key bindings
        self.root.bind('<s>', lambda e: self.save_labels())
        self.root.bind('<S>', lambda e: self.save_labels())
        self.root.bind('<c>', lambda e: self.clear_bboxes())
        self.root.bind('<C>', lambda e: self.clear_bboxes())
        self.root.bind('<n>', lambda e: self.next_file())
        self.root.bind('<N>', lambda e: self.next_file())
        self.root.bind('<p>', lambda e: self.prev_file())
        self.root.bind('<P>', lambda e: self.prev_file())
        self.root.bind('<h>', lambda e: self.show_help())
        self.root.bind('<H>', lambda e: self.show_help())
        self.root.bind('<q>', lambda e: self.quit_app())
        self.root.bind('<Q>', lambda e: self.quit_app())
        self.root.bind('<F11>', lambda e: self.toggle_fullscreen())
        self.root.bind('<Escape>', lambda e: self.quit_app() if not self.fullscreen else self.toggle_fullscreen())
        
        # Mouse interaction for drawing
        self.is_drawing = False
        self.start_x = None
        self.start_y = None
        self.current_bbox = None
        self.bboxes = []  # Store bounding boxes as (class_id, x_min, y_min, x_max, y_max)
        self.current_image_id = None
        self.current_pil_image = None
        self.current_file_path = None
        # Store display information for coordinate conversion
        self.display_info = {
            'canvas_width': 0,
            'canvas_height': 0,
            'img_x': 0,
            'img_y': 0,
            'display_width': 0,
            'display_height': 0
        }
        
        self.image_canvas.bind("<ButtonPress-1>", self.start_drawing)
        self.image_canvas.bind("<B1-Motion>", self.update_drawing)
        self.image_canvas.bind("<ButtonRelease-1>", self.finish_drawing)
        
        # Initialize variables
        self.current_index = 0
        self.unlabeled_files = []
        self.log_file = os.path.join("../labeled_data", "labeling_log.csv")
        
        # Create directories
        os.makedirs("../labeled_data", exist_ok=True)
        os.makedirs("../unlabeled_data", exist_ok=True)
        
        # Setup audit log
        if not os.path.exists(self.log_file):
            with open(self.log_file, 'w', newline='') as f:
                writer = csv.writer(f)
                writer.writerow(["timestamp", "file_path", "class_index", "class_name", "x_min", "y_min", "x_max", "y_max"])
        
        # Load unlabeled files
        self.load_unlabeled_files()
        
        # Update layout based on screen size
        self.update_layout()
        
        # Bind to configure event to handle resizing
        self.image_frame.bind('<Configure>', self.on_image_frame_resize)
        
        # Wait for UI to be fully rendered before loading first image
        self.root.after(100, self.initialize_display)
    
    def setup_styles(self):
        """Configure modern styles for ttk widgets"""
        style = ttk.Style()
        
        # Configure styles for different elements
        style.configure("Main.TFrame", background="#34495e")
        style.configure("Header.TFrame", background="#2c3e50")
        style.configure("Header.TLabel", background="#2c3e50")
        style.configure("Status.TLabel", background="#34495e")
        style.configure("ImageContainer.TFrame", background="#34495e")
        style.configure("ImageTitle.TLabel", background="#34495e")
        style.configure("Image.TFrame", background="#1a252f")
        style.configure("ButtonsContainer.TFrame", background="#34495e")
        style.configure("ButtonsTitle.TLabel", background="#34495e")
        style.configure("Buttons.TFrame", background="#34495e")
        style.configure("SpecialButtons.TFrame", background="#34495e")
        
        # Configure progress bar style
        style.configure("Custom.Horizontal.TProgressbar",
                       background="#1abc9c",
                       troughcolor="#34495e",
                       borderwidth=0,
                       lightcolor="#1abc9c",
                       darkcolor="#1abc9c")
    
    def on_image_frame_resize(self, event):
        """Handle image frame resizing to update the image display"""
        if hasattr(self, 'current_pil_image') and self.current_pil_image is not None:
            self.update_image_display()
    
    def update_image_display(self):
        """Update the image display to fit the current canvas size"""
        if self.current_pil_image is None:
            return
            
        # Get current canvas dimensions
        canvas_width = self.image_canvas.winfo_width()
        canvas_height = self.image_canvas.winfo_height()
        
        # If canvas hasn't been rendered yet, use estimated dimensions
        if canvas_width <= 1 or canvas_height <= 1:
            canvas_width = self.image_frame.winfo_width() - 20
            canvas_height = self.image_frame.winfo_height() - 20
            
            if canvas_width <= 1 or canvas_height <= 1:
                canvas_width = int(self.screen_width * 0.65) - 100
                canvas_height = self.screen_height - 200
        
        # Ensure minimum dimensions
        canvas_width = max(canvas_width, 100)
        canvas_height = max(canvas_height, 100)
        
        # Calculate new dimensions maintaining aspect ratio
        img_width, img_height = self.current_pil_image.size
        canvas_ratio = canvas_width / canvas_height
        img_ratio = img_width / img_height
        
        if canvas_ratio > img_ratio:
            new_height = canvas_height - 20  # Padding
            new_width = int(new_height * img_ratio)
        else:
            new_width = canvas_width - 20  # Padding
            new_height = int(new_width / img_ratio)
        
        # Ensure minimum size
        new_width = max(new_width, 100)
        new_height = max(new_height, 100)
        
        # Calculate image position (centered)
        img_x = (canvas_width - new_width) // 2
        img_y = (canvas_height - new_height) // 2
        
        # Store display information for coordinate conversion
        self.display_info = {
            'canvas_width': canvas_width,
            'canvas_height': canvas_height,
            'img_x': img_x,
            'img_y': img_y,
            'display_width': new_width,
            'display_height': new_height,
            'original_width': img_width,
            'original_height': img_height
        }
        
        # Resize image
        resized_image = self.current_pil_image.resize((new_width, new_height), Image.LANCZOS)
        self.current_photo = ImageTk.PhotoImage(resized_image)
        
        # Clear canvas and display new image
        self.image_canvas.delete("all")
        self.current_image_id = self.image_canvas.create_image(img_x, img_y, anchor=tk.NW, image=self.current_photo)
        
        # Draw bounding boxes
        self._draw_bboxes()
        # Update bboxes text
        self._update_bboxes_text()
    
    def _canvas_to_original_coords(self, x, y):
        """Convert canvas coordinates to original image coordinates"""
        if not self.display_info:
            return x, y
            
        # Convert from canvas coordinates to display image coordinates
        display_x = x - self.display_info['img_x']
        display_y = y - self.display_info['img_y']
        
        # Convert from display coordinates to original image coordinates
        if self.display_info['display_width'] > 0 and self.display_info['display_height'] > 0:
            original_x = (display_x / self.display_info['display_width']) * self.display_info['original_width']
            original_y = (display_y / self.display_info['display_height']) * self.display_info['original_height']
            return original_x, original_y
        else:
            return x, y
    
    def _original_to_canvas_coords(self, x, y):
        """Convert original image coordinates to canvas coordinates"""
        if not self.display_info:
            return x, y
            
        # Convert from original image coordinates to display coordinates
        if self.display_info['original_width'] > 0 and self.display_info['original_height'] > 0:
            display_x = (x / self.display_info['original_width']) * self.display_info['display_width']
            display_y = (y / self.display_info['original_height']) * self.display_info['display_height']
            
            # Convert from display coordinates to canvas coordinates
            canvas_x = display_x + self.display_info['img_x']
            canvas_y = display_y + self.display_info['img_y']
            return canvas_x, canvas_y
        else:
            return x, y
    
    def _update_bboxes_text(self):
        """Update the bounding boxes text display"""
        self.bboxes_text.delete(1.0, tk.END)
        if not self.bboxes:
            self.bboxes_text.insert(tk.END, "No bounding boxes added yet.\n\nClick and drag on the image to add bounding boxes.")
            return
            
        for i, (class_id, x_min, y_min, x_max, y_max) in enumerate(self.bboxes):
            class_name = self.CLASSES[class_id]
            # Convert to original coordinates for display
            orig_x_min, orig_y_min = self._canvas_to_original_coords(x_min, y_min)
            orig_x_max, orig_y_max = self._canvas_to_original_coords(x_max, y_max)
            
            self.bboxes_text.insert(tk.END, f"Box {i+1}: {class_name}\n")
            self.bboxes_text.insert(tk.END, f"  Canvas: ({x_min:.0f}, {y_min:.0f}) to ({x_max:.0f}, {y_max:.0f})\n")
            self.bboxes_text.insert(tk.END, f"  Original: ({orig_x_min:.0f}, {orig_y_min:.0f}) to ({orig_x_max:.0f}, {orig_y_max:.0f})\n")
            self.bboxes_text.insert(tk.END, f"  Size: {orig_x_max-orig_x_min:.0f} x {orig_y_max-orig_y_min:.0f} pixels\n\n")
    
    def update_layout(self):
        """Update widget sizes based on current window size"""
        self.progress_bar.config(length=self.screen_width - 100)
    
    def toggle_fullscreen(self):
        """Toggle between fullscreen and windowed mode"""
        self.fullscreen = not self.fullscreen
        self.root.attributes('-fullscreen', self.fullscreen)
        
        if not self.fullscreen:
            self.root.geometry("1200x900")
            self.fullscreen_btn.config(text="Enter Fullscreen (F11)")
        else:
            self.fullscreen_btn.config(text="Exit Fullscreen (F11)")
        
        self.update_layout()
        if self.current_pil_image is not None:
            self.root.after(100, self.update_image_display)
    
    def set_class(self, class_idx):
        for i, var in enumerate(self.class_vars):
            var.set(i == class_idx)
            # Update button appearance to show selection
            if i == class_idx:
                self.class_buttons[i].config(bg="#e74c3c", activebackground="#c0392b")  # Highlight selected
            else:
                self.class_buttons[i].config(bg="#3498db", activebackground="#2980b9")  # Default color
        self.status_var.set(f"Selected class: {class_idx} - {self.CLASSES[class_idx]}")
    
    def load_unlabeled_files(self):
        self.unlabeled_files = []
        for root, _, files in os.walk("../unlabeled_data"):
            for file in files:
                if file.endswith('.npy'):
                    self.unlabeled_files.append(os.path.join(root, file))
        self.status_var.set(f"Found {len(self.unlabeled_files)} unlabeled files")
        self.update_progress()
    
    def update_progress(self):
        if len(self.unlabeled_files) > 0:
            progress = (self.current_index / len(self.unlabeled_files)) * 100
            self.progress_var.set(min(progress, 100))
            filename = os.path.basename(self.unlabeled_files[self.current_index])
            self.status_var.set(f"File {self.current_index+1} of {len(self.unlabeled_files)}: {filename}")
        else:
            self.progress_var.set(0)
            self.status_var.set("No files to label")
    
    def load_spectrogram(self):
        file_path = filedialog.askopenfilename(
            title="Select Spectrogram File",
            filetypes=[("Numpy files", "*.npy"), ("All files", "*.*")]
        )
        if not file_path:
            return
            
        self.current_file_path = file_path
        if file_path in self.unlabeled_files:
            self.current_index = self.unlabeled_files.index(file_path)
        self.display_file(self.current_index)
    
    def display_file(self, index):
        if index >= len(self.unlabeled_files):
            self.status_var.set("All files labeled!")
            messagebox.showinfo("Complete", "All files have been labeled!")
            return
            
        file_path = self.unlabeled_files[index]
        self.current_file_path = file_path
        
        try:
            # Load and preprocess spectrogram
            spec = np.load(file_path)
            spec = cv2.resize(spec.astype(np.float32), (256, 256))
            spec = (spec - np.min(spec)) / (np.max(spec) - np.min(spec) + 1e-8)
            
            # Convert to RGB using viridis colormap
            cmap = cm.get_cmap('viridis')
            rgba = cmap(spec)
            rgb = (rgba[:, :, :3] * 255).astype(np.uint8)
            img = Image.fromarray(rgb)
            
            # Store the original image for resizing
            self.current_pil_image = img
            self.bboxes = []
            
            # Load existing labels if they exist
            label_path = os.path.splitext(file_path)[0] + ".txt"
            if os.path.exists(label_path):
                with open(label_path, 'r') as f:
                    for line in f:
                        parts = line.strip().split()
                        if len(parts) >= 5:
                            class_id = int(parts[0])
                            x_center = float(parts[1])
                            y_center = float(parts[2])
                            width = float(parts[3])
                            height = float(parts[4])
                            
                            # Convert YOLO format to original image coordinates
                            img_width, img_height = img.size
                            orig_x_min = (x_center - width/2) * img_width
                            orig_y_min = (y_center - height/2) * img_height
                            orig_x_max = (x_center + width/2) * img_width
                            orig_y_max = (y_center + height/2) * img_height
                            
                            # Convert original coordinates to canvas coordinates for display
                            canvas_x_min, canvas_y_min = self._original_to_canvas_coords(orig_x_min, orig_y_min)
                            canvas_x_max, canvas_y_max = self._original_to_canvas_coords(orig_x_max, orig_y_max)
                            
                            self.bboxes.append((class_id, canvas_x_min, canvas_y_min, canvas_x_max, canvas_y_max))
            
            # Update the image display with a small delay to ensure UI is ready
            self.root.after(50, self.update_image_display)
            
            # Update status
            self.update_progress()
        except Exception as e:
            self.status_var.set(f"Error loading file: {str(e)}")
            messagebox.showerror("Error", f"Could not load file: {file_path}\nError: {str(e)}")
            # Skip problematic file
            self.current_index += 1
            if self.current_index < len(self.unlabeled_files):
                self.root.after(100, lambda: self.display_file(self.current_index))
    
    def start_drawing(self, event):
        self.is_drawing = True
        self.start_x = event.x
        self.start_y = event.y
        self.current_bbox = None
    
    def update_drawing(self, event):
        if not self.is_drawing or self.current_pil_image is None:
            return
            
        if self.current_bbox:
            self.image_canvas.delete(self.current_bbox)
        
        self.current_bbox = self.image_canvas.create_rectangle(
            self.start_x, self.start_y, event.x, event.y,
            outline="red", width=2
        )
    
    def finish_drawing(self, event):
        if not self.is_drawing or self.current_pil_image is None:
            return
            
        self.is_drawing = False
        x_min = min(self.start_x, event.x)
        y_min = min(self.start_y, event.y)
        x_max = max(self.start_x, event.x)
        y_max = max(self.start_y, event.y)
        
        # Add to bboxes list (store in canvas coordinates for display)
        class_id = None
        for i, var in enumerate(self.class_vars):
            if var.get():
                class_id = i
                break
        
        if class_id is not None:
            self.bboxes.append((class_id, x_min, y_min, x_max, y_max))
            self._draw_bboxes()
            self._update_bboxes_text()
    
    def _draw_bboxes(self):
        if self.current_pil_image is None:
            return
            
        # Clear existing bounding boxes
        self.image_canvas.delete("bbox")
        
        # Draw new bounding boxes (already in canvas coordinates)
        for class_id, x_min, y_min, x_max, y_max in self.bboxes:
            color = self._get_class_color(class_id)
            self.image_canvas.create_rectangle(
                x_min, y_min, x_max, y_max,
                outline=color, width=2,
                tags="bbox"
            )
            self.image_canvas.create_text(
                x_min + 5, y_min + 5,
                text=str(class_id),
                fill=color,
                anchor=tk.NW,
                tags="bbox"
            )
    
    def _get_class_color(self, class_id):
        colors = [
            "#FFFFFF", "#FF0000", "#00FF00", "#0000FF",
            "#FFFF00", "#FF00FF", "#00FFFF", "#800000",
            "#008000", "#000080", "#808000", "#800080",
            "#008080", "#808080", "#C00000", "#00C000",
            "#0000C0", "#C0C000", "#C000C0", "#00C0C0",
            "#C0C0C0", "#400000", "#004000", "#000040"
        ]
        return colors[class_id % len(colors)]
    
    def save_labels(self):
        if self.current_pil_image is None:
            messagebox.showwarning("Warning", "No image to save")
            return
            
        # Save image (optional)
        img_path = os.path.splitext(self.current_file_path)[0] + ".png"
        self.current_pil_image.save(img_path)
        
        # Save YOLO labels
        label_path = os.path.splitext(self.current_file_path)[0] + ".txt"
        with open(label_path, 'w') as f:
            for class_id, x_min, y_min, x_max, y_max in self.bboxes:
                # Convert canvas coordinates to original image coordinates
                orig_x_min, orig_y_min = self._canvas_to_original_coords(x_min, y_min)
                orig_x_max, orig_y_max = self._canvas_to_original_coords(x_max, y_max)
                
                img_width, img_height = self.current_pil_image.size
                
                # Convert to YOLO format (normalized center coordinates and dimensions)
                x_center = (orig_x_min + orig_x_max) / (2 * img_width)
                y_center = (orig_y_min + orig_y_max) / (2 * img_height)
                width = (orig_x_max - orig_x_min) / img_width
                height = (orig_y_max - orig_y_min) / img_height
                
                # Ensure values are within [0, 1] range
                x_center = max(0.0, min(1.0, x_center))
                y_center = max(0.0, min(1.0, y_center))
                width = max(0.0, min(1.0, width))
                height = max(0.0, min(1.0, height))
                
                f.write(f"{class_id} {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}\n")
        
        # Log decision
        with open(self.log_file, 'a', newline='') as f:
            writer = csv.writer(f)
            for class_id, x_min, y_min, x_max, y_max in self.bboxes:
                # Convert to original coordinates for logging
                orig_x_min, orig_y_min = self._canvas_to_original_coords(x_min, y_min)
                orig_x_max, orig_y_max = self._canvas_to_original_coords(x_max, y_max)
                
                writer.writerow([
                    time.strftime("%Y-%m-%d %H:%M:%S"),
                    self.current_file_path,
                    class_id,
                    self.CLASSES[class_id],
                    orig_x_min,
                    orig_y_min,
                    orig_x_max,
                    orig_y_max
                ])
        
        self.status_var.set(f"Labels saved to {label_path}")
    
    def clear_bboxes(self):
        self.bboxes = []
        self._draw_bboxes()
        self._update_bboxes_text()
        self.status_var.set("All bounding boxes cleared")
    
    def next_file(self):
        self.save_labels()
        if self.current_index < len(self.unlabeled_files) - 1:
            self.current_index += 1
            self.display_file(self.current_index)
        else:
            self.status_var.set("All files labeled!")
            messagebox.showinfo("Complete", "All files have been labeled!")
    
    def prev_file(self):
        self.save_labels()
        if self.current_index > 0:
            self.current_index -= 1
            self.display_file(self.current_index)
        else:
            self.status_var.set("Already at first file")
    
    def show_help(self):
        """Show a custom help window with proper width"""
        help_window = tk.Toplevel(self.root)
        help_window.title("Keyboard Shortcuts")
        help_window.geometry("1000x1000")  # Wider window
        help_window.configure(bg="#2c3e50")
        help_window.resizable(True, True)
        
        # Center the help window
        help_window.transient(self.root)
        help_window.grab_set()
        
        # Help text with better formatting
        help_text = """KEYBOARD SHORTCUTS:

CLASS SELECTION:
0-9: Classes 0-9
A-N: Classes 10-23

ACTIONS:
S: Save Labels for current file
C: Clear all bounding boxes
N: Next file (saves current labels first)
P: Previous file (saves current labels first)
H: Show this help window
F11: Toggle Fullscreen mode
Esc: Exit Fullscreen or Quit application
Q: Quit application

MOUSE CONTROLS:
• Click and drag on the spectrogram to draw bounding boxes
• Make sure to select a class first before drawing
• The selected class will be highlighted in red

WORKFLOW:
1. Select a class from the right panel or use keyboard shortcuts
2. Click and drag on the spectrogram to draw bounding boxes
3. Use Save (S) to save labels, or they auto-save when navigating
4. Use Next (N) and Previous (P) to navigate between files
5. All labels are saved in YOLO format alongside the spectrograms"""
        
        # Create a text widget with scrollbar for the help
        text_frame = ttk.Frame(help_window, style="Main.TFrame")
        text_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        help_text_widget = tk.Text(
            text_frame,
            wrap=tk.WORD,
            font=("Arial", 10),
            bg="#34495e",
            fg="#ecf0f1",
            padx=10,
            pady=10,
            relief="flat"
        )
        help_text_widget.insert(1.0, help_text)
        help_text_widget.config(state=tk.DISABLED)  # Make it read-only
        
        scrollbar = ttk.Scrollbar(text_frame, orient=tk.VERTICAL, command=help_text_widget.yview)
        help_text_widget.configure(yscrollcommand=scrollbar.set)
        
        help_text_widget.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        
        # Close button
        close_btn = tk.Button(
            help_window,
            text="Close",
            font=("Arial", 10, "bold"),
            bg="#e74c3c",
            fg="white",
            activebackground="#c0392b",
            relief="raised",
            bd=2,
            cursor="hand2",
            command=help_window.destroy
        )
        close_btn.pack(pady=10)
        
        # Make the text widget take focus so scrolling works immediately
        help_text_widget.focus_set()
    
    def quit_app(self):
        self.save_labels()
        if messagebox.askyesno("Confirm Quit", "Are you sure you want to quit?"):
            self.root.destroy()
    
    def initialize_display(self):
        """Initialize the display after the UI is fully rendered"""
        if self.unlabeled_files:
            self.display_file(0)
        else:
            self.status_var.set("No unlabeled files found in ../unlabeled_data")
            messagebox.showinfo("Information", "No unlabeled files found in ../unlabeled_data")

if __name__ == "__main__":
    root = tk.Tk()
    app = SpectrogramYOLOLabeler(root)
    root.mainloop()