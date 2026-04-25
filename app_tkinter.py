# ==============================================
# 📱 MOBILE PRICE CLASSIFICATION - TKINTER APP
# ==============================================

import tkinter as tk
from tkinter import ttk, messagebox
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression

# ==============================================
# LOAD AND TRAIN MODEL
# ==============================================
print("Loading and training model...")
train_df = pd.read_csv('data/train.csv')
X = train_df.drop(columns=['price_range'])
y = train_df['price_range']

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

model = LogisticRegression(max_iter=1000, random_state=42)
model.fit(X_scaled, y)
print(f"Model trained! Accuracy: {model.score(X_scaled, y)*100:.1f}%")

# ==============================================
# PRICE RANGE INFO
# ==============================================
PRICE_INFO = {
    0: {'label': 'Low Cost', 'emoji': '💰', 'color': '#28a745', 'description': 'Budget-friendly phone with basic features'},
    1: {'label': 'Medium Cost', 'emoji': '💵', 'color': '#17a2b8', 'description': 'Mid-range phone with good features'},
    2: {'label': 'High Cost', 'emoji': '💎', 'color': '#ffc107', 'description': 'Premium phone with advanced features'},
    3: {'label': 'Very High Cost', 'emoji': '👑', 'color': '#dc3545', 'description': 'Flagship phone with top-tier specifications'}
}

# Feature list in correct order for the model
FEATURE_ORDER = ['battery_power', 'blue', 'clock_speed', 'dual_sim', 'fc',
                 'four_g', 'int_memory', 'm_dep', 'mobile_wt', 'n_cores', 'pc',
                 'px_height', 'px_width', 'ram', 'sc_h', 'sc_w', 'talk_time',
                 'three_g', 'touch_screen', 'wifi']

# ==============================================
# MAIN APPLICATION CLASS
# ==============================================
class MobilePriceApp:
    def __init__(self, root):
        self.root = root
        self.root.title("📱 Mobile Price Predictor")
        self.root.geometry("900x750")
        self.root.resizable(True, True)
        self.root.configure(bg='#f0f2f6')
        
        # Configure style
        self.style = ttk.Style()
        self.style.theme_use('clam')
        self.style.configure('TLabel', background='#f0f2f6', font=('Segoe UI', 10))
        self.style.configure('TFrame', background='#f0f2f6')
        self.style.configure('Title.TLabel', font=('Segoe UI', 22, 'bold'), foreground='#1f77b4')
        self.style.configure('Subtitle.TLabel', font=('Segoe UI', 11), foreground='#666')
        self.style.configure('Predict.TButton', font=('Segoe UI', 14, 'bold'), background='#1f77b4')
        self.style.configure('Reset.TButton', font=('Segoe UI', 10))
        self.style.configure('Section.TLabel', font=('Segoe UI', 12, 'bold'), foreground='#333')
        
        # Store all input variables
        self.vars = {}
        
        # Build the UI
        self.build_ui()
    
    # -------------------------------------------------
    # BUILD USER INTERFACE
    # -------------------------------------------------
    def build_ui(self):
        # --- MAIN CONTAINER ---
        main_frame = ttk.Frame(self.root, padding=20)
        main_frame.pack(fill='both', expand=True)
        
        # --- TITLE ---
        title_label = ttk.Label(main_frame, text="📱 Mobile Price Predictor", style='Title.TLabel')
        title_label.pack(pady=(0, 5))
        
        subtitle_label = ttk.Label(main_frame, 
            text="Enter the specifications of a mobile phone and predict its price range",
            style='Subtitle.TLabel')
        subtitle_label.pack(pady=(0, 20))
        
        # --- NOTEBOOK (TABS) FOR ORGANIZED INPUT ---
        notebook = ttk.Notebook(main_frame)
        notebook.pack(fill='both', expand=True, pady=(0, 15))
        
        # Tab 1: Core Specifications
        core_frame = ttk.Frame(notebook, padding=15)
        notebook.add(core_frame, text="  📱 Core Specs  ")
        self.build_core_tab(core_frame)
        
        # Tab 2: Display & Camera
        display_frame = ttk.Frame(notebook, padding=15)
        notebook.add(display_frame, text="  📸 Display & Camera  ")
        self.build_display_tab(display_frame)
        
        # Tab 3: Physical & Connectivity
        physical_frame = ttk.Frame(notebook, padding=15)
        notebook.add(physical_frame, text="  ⚙️ Physical & Connectivity  ")
        self.build_physical_tab(physical_frame)
        
        # --- BUTTONS ---
        button_frame = ttk.Frame(main_frame)
        button_frame.pack(fill='x', pady=10)
        
        predict_btn = ttk.Button(button_frame, text="🔮 Predict Price", 
                                style='Predict.TButton', command=self.predict)
        predict_btn.pack(side='left', padx=5, ipadx=30, ipady=5)
        
        reset_btn = ttk.Button(button_frame, text="🔄 Reset to Default", 
                              style='Reset.TButton', command=self.reset)
        reset_btn.pack(side='left', padx=5)
        
        # --- RESULT FRAME ---
        self.result_frame = ttk.Frame(main_frame, padding=15)
        self.result_frame.pack(fill='x', pady=10)
        
        self.result_label = ttk.Label(self.result_frame, text="", 
                                       font=('Segoe UI', 14), anchor='center')
        self.result_label.pack(fill='x')
        
        self.prob_frame = ttk.Frame(self.result_frame)
        self.prob_frame.pack(fill='x', pady=10)
        
        # Footer
        footer_label = ttk.Label(main_frame, 
            text="Model: Logistic Regression | Accuracy: 96.50% | Key Predictor: RAM",
            font=('Segoe UI', 9), foreground='#999')
        footer_label.pack(side='bottom', pady=(10, 0))
    
    # -------------------------------------------------
    # TAB 1: CORE SPECIFICATIONS
    # -------------------------------------------------
    def build_core_tab(self, parent):
        # RAM (most important)
        self.create_slider_row(parent, "🚀 RAM (MB):", 'ram', 256, 4000, 2000, 
                              "(Most important predictor of price!)")
        
        ttk.Separator(parent, orient='horizontal').pack(fill='x', pady=10)
        
        self.create_slider_row(parent, "🔋 Battery Power (mAh):", 'battery_power', 500, 2000, 1250)
        self.create_slider_row(parent, "⚡ Clock Speed (GHz):", 'clock_speed', 0.5, 3.0, 1.5, 
                              step=0.1, is_float=True)
        self.create_slider_row(parent, "🧠 Processor Cores:", 'n_cores', 1, 8, 4)
        self.create_slider_row(parent, "💾 Internal Memory (GB):", 'int_memory', 2, 64, 32)
    
    # -------------------------------------------------
    # TAB 2: DISPLAY & CAMERA
    # -------------------------------------------------
    def build_display_tab(self, parent):
        self.create_slider_row(parent, "📸 Primary Camera (MP):", 'pc', 0, 20, 10)
        self.create_slider_row(parent, "🤳 Front Camera (MP):", 'fc', 0, 20, 8)
        self.create_slider_row(parent, "🖥️ Pixel Height:", 'px_height', 0, 2000, 1000)
        self.create_slider_row(parent, "🖥️ Pixel Width:", 'px_width', 500, 2000, 1250)
        self.create_slider_row(parent, "📱 Screen Height (cm):", 'sc_h', 5, 19, 12)
        self.create_slider_row(parent, "📱 Screen Width (cm):", 'sc_w', 0, 18, 7)
    
    # -------------------------------------------------
    # TAB 3: PHYSICAL & CONNECTIVITY
    # -------------------------------------------------
    def build_physical_tab(self, parent):
        # Physical
        ttk.Label(parent, text="Physical Specifications", style='Section.TLabel').pack(anchor='w', pady=(0, 10))
        
        self.create_slider_row(parent, "⚖️ Weight (g):", 'mobile_wt', 80, 200, 140)
        self.create_slider_row(parent, "📏 Mobile Depth (cm):", 'm_dep', 0.1, 1.0, 0.5, 
                              step=0.1, is_float=True)
        self.create_slider_row(parent, "🗣️ Talk Time (hours):", 'talk_time', 2, 20, 10)
        
        ttk.Separator(parent, orient='horizontal').pack(fill='x', pady=15)
        
        # Connectivity (checkboxes)
        ttk.Label(parent, text="Connectivity & Features", style='Section.TLabel').pack(anchor='w', pady=(0, 10))
        
        checkbox_frame = ttk.Frame(parent)
        checkbox_frame.pack(fill='x')
        
        self.create_checkbox_row(checkbox_frame, "🔵 Bluetooth", 'blue', 0)
        self.create_checkbox_row(checkbox_frame, "📶 WiFi", 'wifi', 1)
        self.create_checkbox_row(checkbox_frame, "📡 4G", 'four_g', 1)
        self.create_checkbox_row(checkbox_frame, "📶 3G", 'three_g', 1)
        self.create_checkbox_row(checkbox_frame, "📱 Dual SIM", 'dual_sim', 1)
        self.create_checkbox_row(checkbox_frame, "👆 Touch Screen", 'touch_screen', 1)
    
    # -------------------------------------------------
    # CREATE SLIDER ROW (Helper Method)
    # -------------------------------------------------
    def create_slider_row(self, parent, label_text, var_name, min_val, max_val, default_val, 
                         help_text="", step=None, is_float=False):
        frame = ttk.Frame(parent)
        frame.pack(fill='x', pady=5)
        
        # Label
        label = ttk.Label(frame, text=label_text, width=25)
        label.pack(side='left')
        
        # Value display
        if is_float:
            self.vars[var_name] = tk.DoubleVar(value=default_val)
        else:
            self.vars[var_name] = tk.IntVar(value=default_val)
        
        value_label = ttk.Label(frame, textvariable=self.vars[var_name], width=8, 
                                font=('Segoe UI', 10, 'bold'))
        value_label.pack(side='right')
        
        # Slider
        if step is None:
            step = 0.1 if is_float else 1
        
        slider = ttk.Scale(frame, from_=min_val, to=max_val, orient='horizontal',
                          variable=self.vars[var_name], length=300)
        slider.pack(side='right', padx=10)
        
        # Help text
        if help_text:
            help_label = ttk.Label(parent, text=f"    {help_text}", 
                                  font=('Segoe UI', 8), foreground='#999')
            help_label.pack(anchor='w')
    
    # -------------------------------------------------
    # CREATE CHECKBOX ROW (Helper Method)
    # -------------------------------------------------
    def create_checkbox_row(self, parent, label_text, var_name, default_val):
        self.vars[var_name] = tk.IntVar(value=default_val)
        cb = ttk.Checkbutton(parent, text=label_text, variable=self.vars[var_name])
        cb.pack(side='left', padx=10, pady=5)
    
    # -------------------------------------------------
    # PREDICT PRICE
    # -------------------------------------------------
    def predict(self):
        # Collect inputs in correct order
        input_values = []
        for feature in FEATURE_ORDER:
            input_values.append(self.vars[feature].get())
        
        input_array = np.array([input_values])
        
        # Scale and predict
        input_scaled = scaler.transform(input_array)
        prediction = model.predict(input_scaled)[0]
        probabilities = model.predict_proba(input_scaled)[0]
        
        # Get price info
        info = PRICE_INFO[prediction]
        
        # --- DISPLAY RESULT ---
        # Clear previous results
        for widget in self.result_frame.winfo_children():
            widget.destroy()
        
        # Create result display
        result_container = ttk.Frame(self.result_frame, padding=20)
        result_container.pack(fill='x')
        
        # Main prediction
        result_label = ttk.Label(result_container, 
            text=f"{info['emoji']}  Price Range: {prediction} - {info['label']}",
            font=('Segoe UI', 20, 'bold'), anchor='center')
        result_label.pack(fill='x', pady=(0, 5))
        
        desc_label = ttk.Label(result_container, text=info['description'],
                               font=('Segoe UI', 12), foreground='#666', anchor='center')
        desc_label.pack(fill='x', pady=(0, 15))
        
        # Probability bars
        prob_title = ttk.Label(result_container, text="Confidence Levels:",
                              font=('Segoe UI', 12, 'bold'))
        prob_title.pack(anchor='w', pady=(0, 10))
        
        for i in range(4):
            prob = probabilities[i]
            info_i = PRICE_INFO[i]
            
            prob_row = ttk.Frame(result_container)
            prob_row.pack(fill='x', pady=3)
            
            # Emoji + label
            label = ttk.Label(prob_row, text=f"{info_i['emoji']} {info_i['label']}", width=25)
            label.pack(side='left')
            
            # Progress bar
            bar = ttk.Progressbar(prob_row, length=300, mode='determinate', 
                                 value=prob * 100)
            bar.pack(side='left', padx=10)
            
            # Percentage
            pct_label = ttk.Label(prob_row, text=f"{prob*100:.1f}%", 
                                 font=('Segoe UI', 10, 'bold'), width=8)
            pct_label.pack(side='left')
    
    # -------------------------------------------------
    # RESET TO DEFAULTS
    # -------------------------------------------------
    def reset(self):
        defaults = {
            'ram': 2000, 'battery_power': 1250, 'clock_speed': 1.5, 'n_cores': 4,
            'int_memory': 32, 'pc': 10, 'fc': 8, 'px_height': 1000, 'px_width': 1250,
            'sc_h': 12, 'sc_w': 7, 'mobile_wt': 140, 'm_dep': 0.5, 'talk_time': 10,
            'blue': 0, 'wifi': 1, 'four_g': 1, 'three_g': 1, 'dual_sim': 1, 'touch_screen': 1
        }
        for key, val in defaults.items():
            self.vars[key].set(val)
        
        # Clear result
        for widget in self.result_frame.winfo_children():
            widget.destroy()
        
        messagebox.showinfo("Reset", "All values have been reset to default!")

# ==============================================
# RUN THE APPLICATION
# ==============================================
if __name__ == "__main__":
    root = tk.Tk()
    app = MobilePriceApp(root)
    root.mainloop()