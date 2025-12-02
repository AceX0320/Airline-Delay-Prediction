import tkinter as tk
from tkinter import ttk, messagebox
import pandas as pd
import numpy as np
import joblib
import json
import os

# Configuration
L1_MODEL_FILE = 'logistic_l1_model.pkl'
CONFIG_FILE = 'model_config.json'

class DelayPredictionApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Airline Delay Prediction Dashboard")
        self.root.geometry("1100x750")
        self.root.configure(bg="#f0f2f5")
        
        self.model = None
        self.config = None
        self.load_artifacts()
        
        self.scenarios = self.get_scenarios()
        self.predictions_cache = {} 
        self.last_hovered = None # Track hovered item
        
        self.create_widgets()
        self.populate_dashboard()
        
    def load_artifacts(self):
        try:
            if not os.path.exists(L1_MODEL_FILE) or not os.path.exists(CONFIG_FILE):
                messagebox.showerror("Error", "Model files not found! Please run airline_delay_prediction.py first.")
                self.root.destroy()
                return
                
            self.model = joblib.load(L1_MODEL_FILE)
            with open(CONFIG_FILE, 'r') as f:
                self.config = json.load(f)
        except Exception as e:
            messagebox.showerror("Error", f"Failed to load model: {str(e)}")
            self.root.destroy()

    def get_scenarios(self):
        # List of scenarios with Weather added
        return [
            {
                'name': 'High Risk: Winter Holiday',
                'weather': 'Snow',
                'month': 12, 'day_of_month': 24, 'day_of_week': 7, 
                'crs_dep_time': 1800, 'op_unique_carrier': 'UA', 
                'origin': 'ORD', 'dest': 'JFK', 'distance': 740.0
            },
            {
                'name': 'Low Risk: Spring Mid-Week',
                'weather': 'Clear',
                'month': 4, 'day_of_month': 15, 'day_of_week': 3, 
                'crs_dep_time': 900, 'op_unique_carrier': 'WN', 
                'origin': 'PHX', 'dest': 'LAS', 'distance': 255.0
            },
            {
                'name': 'Medium Risk: Summer Afternoon',
                'weather': 'Rain',
                'month': 7, 'day_of_month': 10, 'day_of_week': 5, 
                'crs_dep_time': 1400, 'op_unique_carrier': 'DL', 
                'origin': 'ATL', 'dest': 'MCO', 'distance': 404.0
            },
            {
                'name': 'Long Haul: Coast to Coast',
                'weather': 'Clear',
                'month': 6, 'day_of_month': 1, 'day_of_week': 6, 
                'crs_dep_time': 800, 'op_unique_carrier': 'AA', 
                'origin': 'LAX', 'dest': 'JFK', 'distance': 2475.0
            },
            {
                'name': 'Short Haul: Northeast Corridor',
                'weather': 'Storm',
                'month': 11, 'day_of_month': 27, 'day_of_week': 4, 
                'crs_dep_time': 1700, 'op_unique_carrier': 'B6', 
                'origin': 'BOS', 'dest': 'LGA', 'distance': 184.0
            },
            {
                'name': 'Early Morning Business',
                'weather': 'Fog',
                'month': 9, 'day_of_month': 12, 'day_of_week': 2, 
                'crs_dep_time': 600, 'op_unique_carrier': 'AS', 
                'origin': 'SFO', 'dest': 'SEA', 'distance': 679.0
            },
            {
                'name': 'Late Night Red-Eye',
                'weather': 'Clear',
                'month': 3, 'day_of_month': 5, 'day_of_week': 1, 
                'crs_dep_time': 2300, 'op_unique_carrier': 'NK', 
                'origin': 'LAS', 'dest': 'MIA', 'distance': 2174.0
            },
            {
                'name': 'Holiday Rush',
                'weather': 'Snow',
                'month': 11, 'day_of_month': 26, 'day_of_week': 3, 
                'crs_dep_time': 1630, 'op_unique_carrier': 'UA', 
                'origin': 'DEN', 'dest': 'ORD', 'distance': 888.0
            },
            {
                'name': 'Regional Connector',
                'weather': 'Rain',
                'month': 5, 'day_of_month': 20, 'day_of_week': 4, 
                'crs_dep_time': 1000, 'op_unique_carrier': 'AA', 
                'origin': 'CLT', 'dest': 'RDU', 'distance': 130.0
            },
            {
                'name': 'Island Hopper',
                'weather': 'Clear',
                'month': 1, 'day_of_month': 15, 'day_of_week': 1, 
                'crs_dep_time': 1200, 'op_unique_carrier': 'HA', 
                'origin': 'HNL', 'dest': 'OGG', 'distance': 100.0
            }
        ]

    def create_widgets(self):
        # --- STYLING (The "CSS" of Tkinter) ---
        style = ttk.Style()
        style.theme_use('clam')
        
        # NEW Professional Color Palette (Cool Blue Theme)
        BG_COLOR = "#e8efff"        # Cool Blue Background
        HEADER_BG = "#1a237e"       # Deep Indigo Header
        HEADER_FG = "#ffffff"       # White Text
        CARD_BG = "#ffffff"         # White Card Background
        ACCENT_COLOR = "#3949ab"    # Indigo Accent
        TEXT_COLOR = "#2c3e50"      # Dark Grey Text
        SUBTEXT_COLOR = "#546e7a"   # Blue Grey Text
        SUCCESS_COLOR = "#00c853"   # Vibrant Green
        DANGER_COLOR = "#d50000"    # Vibrant Red
        ROW_ALT_COLOR = "#f5f7fa"   # Very Light Grey
        HOVER_COLOR = "#e3f2fd"     # Light Blue for Hover
        
        # Configure Styles
        style.configure(".", background=BG_COLOR, foreground=TEXT_COLOR, font=("Segoe UI", 10))
        style.configure("TLabel", background=BG_COLOR, foreground=TEXT_COLOR)
        style.configure("TFrame", background=BG_COLOR)
        
        # Header Style
        style.configure("Header.TFrame", background=HEADER_BG)
        style.configure("Header.TLabel", background=HEADER_BG, foreground=HEADER_FG, font=("Segoe UI", 26, "bold"))
        style.configure("SubHeader.TLabel", font=("Segoe UI", 12, "bold"), foreground=ACCENT_COLOR, background=CARD_BG)
        style.configure("Card.TFrame", background=CARD_BG, relief="raised", borderwidth=1) 
        style.configure("CardLabel.TLabel", background=CARD_BG)
        
        # Treeview Style
        style.configure("Treeview", 
                        background=CARD_BG, 
                        foreground=TEXT_COLOR, 
                        fieldbackground=CARD_BG,
                        font=("Segoe UI", 10),
                        rowheight=35,
                        borderwidth=0)
        style.configure("Treeview.Heading", 
                        background="#c5cae9", 
                        foreground="#1a237e", 
                        font=("Segoe UI", 10, "bold"))
        style.map("Treeview", 
                  background=[('selected', ACCENT_COLOR)], 
                  foreground=[('selected', 'white')])
        
        # Main Container
        main_container = ttk.Frame(self.root)
        main_container.pack(fill=tk.BOTH, expand=True)
        
        # --- HEADER SECTION ---
        header_frame = ttk.Frame(main_container, style="Header.TFrame", padding="25 20")
        header_frame.pack(fill=tk.X)
        
        ttk.Label(header_frame, text="✈️ Airline Delay Prediction Dashboard", style="Header.TLabel").pack(side=tk.LEFT)
        
        # --- CONTENT SECTION ---
        content_frame = ttk.Frame(main_container, padding="25")
        content_frame.pack(fill=tk.BOTH, expand=True)
        
        # --- TOP PANEL: DASHBOARD TABLE ---
        table_card = ttk.Frame(content_frame, style="Card.TFrame", padding="2") 
        table_card.pack(fill=tk.BOTH, expand=True, pady=(0, 25))
        
        table_inner = ttk.Frame(table_card, style="Card.TFrame", padding="20")
        table_inner.pack(fill=tk.BOTH, expand=True)
        
        ttk.Label(table_inner, text="Flight Simulation Scenarios", font=("Segoe UI", 16, "bold"), style="CardLabel.TLabel", foreground=ACCENT_COLOR).pack(anchor="w", pady=(0, 15))
        
        columns = ("flight", "date", "weather", "prediction", "prob")
        self.table = ttk.Treeview(table_inner, columns=columns, show="headings", height=8)
        
        self.table.heading("flight", text="Flight Route")
        self.table.heading("date", text="Date")
        self.table.heading("weather", text="Weather")
        self.table.heading("prediction", text="Prediction")
        self.table.heading("prob", text="Delay Probability")
        
        self.table.column("flight", width=250)
        self.table.column("date", width=100, anchor="center")
        self.table.column("weather", width=100, anchor="center")
        self.table.column("prediction", width=100, anchor="center")
        self.table.column("prob", width=150, anchor="center")
        
        # Scrollbar
        scrollbar = ttk.Scrollbar(table_inner, orient=tk.VERTICAL, command=self.table.yview)
        self.table.configure(yscroll=scrollbar.set)
        
        self.table.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        
        # Bind selection
        self.table.bind("<<TreeviewSelect>>", self.on_scenario_select)
        self.table.bind("<Motion>", self.highlight_row)
        
        # Configure Tags for Table Colors
        self.table.tag_configure("delayed", foreground=DANGER_COLOR, font=("Segoe UI", 10, "bold"))
        self.table.tag_configure("ontime", foreground=SUCCESS_COLOR, font=("Segoe UI", 10, "bold"))
        self.table.tag_configure("evenrow", background=ROW_ALT_COLOR)
        self.table.tag_configure("hover", background=HOVER_COLOR)
        
        # --- BOTTOM PANEL: ANALYSIS ---
        analysis_card = ttk.Frame(content_frame, style="Card.TFrame", padding="2")
        analysis_card.pack(fill=tk.BOTH, expand=True)
        
        analysis_inner = ttk.Frame(analysis_card, style="Card.TFrame", padding="20")
        analysis_inner.pack(fill=tk.BOTH, expand=True)
        
        # Left side: Selected Flight Details
        details_frame = ttk.Frame(analysis_inner, style="Card.TFrame")
        details_frame.pack(side=tk.LEFT, fill=tk.Y, padx=(0, 40))
        
        ttk.Label(details_frame, text="Prediction Analysis", font=("Segoe UI", 16, "bold"), style="CardLabel.TLabel", foreground=ACCENT_COLOR).pack(anchor="w", pady=(0, 20))
        
        self.lbl_scenario = ttk.Label(details_frame, text="Select a flight above...", font=("Segoe UI", 12), style="CardLabel.TLabel", foreground=SUBTEXT_COLOR)
        self.lbl_scenario.pack(anchor="w", pady=(0, 15))
        
        self.lbl_prediction = ttk.Label(details_frame, text="", font=("Segoe UI", 28, "bold"), style="CardLabel.TLabel")
        self.lbl_prediction.pack(anchor="w", pady=(0, 10))
        
        self.lbl_prob = ttk.Label(details_frame, text="", font=("Segoe UI", 14), style="CardLabel.TLabel")
        self.lbl_prob.pack(anchor="w")
        
        # Right side: Reasoning Table
        reasoning_frame = ttk.Frame(analysis_inner, style="Card.TFrame")
        reasoning_frame.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True)
        
        ttk.Label(reasoning_frame, text="Top Contributing Factors", style="SubHeader.TLabel").pack(anchor="w", pady=(0, 5))
        ttk.Label(reasoning_frame, text="Impact Score: (+) Increases Risk, (-) Decreases Risk", font=("Segoe UI", 9), foreground=SUBTEXT_COLOR, style="CardLabel.TLabel").pack(anchor="w", pady=(0, 10))
        
        r_columns = ("feature", "effect", "impact", "desc")
        self.reasoning_tree = ttk.Treeview(reasoning_frame, columns=r_columns, show="headings", height=6)
        
        self.reasoning_tree.heading("feature", text="Factor")
        self.reasoning_tree.heading("effect", text="Effect")
        self.reasoning_tree.heading("impact", text="Impact Score")
        self.reasoning_tree.heading("desc", text="Explanation")
        
        self.reasoning_tree.column("feature", width=200)
        self.reasoning_tree.column("effect", width=120)
        self.reasoning_tree.column("impact", width=80, anchor="center")
        self.reasoning_tree.column("desc", width=250)
        
        self.reasoning_tree.pack(fill=tk.BOTH, expand=True)
        
        # Configure Tags for Reasoning Colors
        self.reasoning_tree.tag_configure("increase_risk", foreground=DANGER_COLOR)
        self.reasoning_tree.tag_configure("decrease_risk", foreground=SUCCESS_COLOR)

    def highlight_row(self, event):
        item = self.table.identify_row(event.y)
        
        if item != self.last_hovered:
            # Remove hover from previous
            if self.last_hovered:
                tags = list(self.table.item(self.last_hovered, "tags"))
                if "hover" in tags:
                    tags.remove("hover")
                    self.table.item(self.last_hovered, tags=tuple(tags))
            
            # Add hover to new
            if item:
                tags = list(self.table.item(item, "tags"))
                if "hover" not in tags:
                    tags.append("hover")
                    self.table.item(item, tags=tuple(tags))
            
            self.last_hovered = item

    def populate_dashboard(self):
        for i, scenario in enumerate(self.scenarios):
            # Prepare data for prediction
            data = {k: v for k, v in scenario.items() if k != 'name' and k != 'weather'}
            df = pd.DataFrame([data])
            
            # Predict Base Probability
            base_prob = self.model.predict_proba(df)[0, 1]
            
            # Apply Weather Impact (Heuristic)
            weather = scenario.get('weather', 'Clear')
            weather_impact = 0.0
            if weather == 'Rain': weather_impact = 0.15
            elif weather == 'Snow': weather_impact = 0.30
            elif weather == 'Storm': weather_impact = 0.45
            elif weather == 'Fog': weather_impact = 0.10
            
            final_prob = min(1.0, base_prob + weather_impact)
            prediction = "DELAYED" if final_prob > 0.5 else "ON-TIME"
            
            # Store result
            self.predictions_cache[i] = {
                'data': data,
                'prob': final_prob,
                'prediction': prediction,
                'weather': weather,
                'weather_impact': weather_impact
            }
            
            # Format display strings
            flight_str = f"{data['op_unique_carrier']} {data['origin']}->{data['dest']}"
            date_str = f"M{data['month']}/D{data['day_of_month']}"
            prob_str = f"{final_prob*100:.1f}%"
            
            # Insert into table
            # Tag rows for color
            pred_tag = "delayed" if prediction == "DELAYED" else "ontime"
            row_tag = "evenrow" if i % 2 == 0 else "oddrow"
            
            self.table.insert("", tk.END, iid=i, values=(flight_str, date_str, weather, prediction, prob_str), tags=(pred_tag, row_tag))

    def get_readable_feature_name(self, feature_name):
        """Convert technical feature names to readable text."""
        if feature_name == "month": return "Month of Year"
        if feature_name == "day_of_month": return "Day of Month"
        if feature_name == "day_of_week": return "Day of Week"
        if feature_name == "crs_dep_time": return "Scheduled Departure Time"
        if feature_name == "distance": return "Flight Distance"
        
        if feature_name.startswith("op_unique_carrier_"):
            return f"Airline: {feature_name.split('_')[-1]}"
        if feature_name.startswith("origin_"):
            return f"Origin Airport: {feature_name.split('_')[-1]}"
        if feature_name.startswith("dest_"):
            return f"Destination Airport: {feature_name.split('_')[-1]}"
            
        return feature_name

    def on_scenario_select(self, event):
        selected_items = self.table.selection()
        if not selected_items:
            return
            
        index = int(selected_items[0])
        result = self.predictions_cache[index]
        scenario = self.scenarios[index]
        
        # Update Details
        self.lbl_scenario.config(text=f"{scenario['name']} ({result['weather']})")
        
        color = "#c0392b" if result['prediction'] == "DELAYED" else "#27ae60"
        self.lbl_prediction.config(text=result['prediction'], foreground=color)
        self.lbl_prob.config(text=f"Probability: {result['prob']*100:.2f}%")
        
        # Update Reasoning
        self.update_reasoning(result['data'], result['weather'], result['weather_impact'])

    def update_reasoning(self, data, weather, weather_impact):
        # Clear tree
        for item in self.reasoning_tree.get_children():
            self.reasoning_tree.delete(item)
            
        # Calculate explanation
        preprocessor = self.model.named_steps['preprocessor']
        row_df = pd.DataFrame([data])
        row_transformed = preprocessor.transform(row_df)
        
        classifier = self.model.named_steps['classifier']
        coefs = classifier.coef_[0]
        
        if hasattr(row_transformed, "toarray"):
            row_values = row_transformed.toarray()[0]
        else:
            row_values = row_transformed[0]
            
        contributions = row_values * coefs
        
        cat_features = (preprocessor.named_transformers_['cat']
                       .named_steps['onehot']
                       .get_feature_names_out(['op_unique_carrier', 'origin', 'dest']))
        num_features = ['month', 'day_of_month', 'day_of_week', 'crs_dep_time', 'distance']
        all_features = np.concatenate([num_features, cat_features])
        
        explanation = pd.DataFrame({
            'Feature': all_features,
            'Contribution': contributions
        })
        
        explanation['Abs_Contribution'] = explanation['Contribution'].abs()
        explanation = explanation.sort_values(by='Abs_Contribution', ascending=False)
        
        # Combine with Weather
        top_features = explanation.head(5).copy()
        
        # Add Weather Row if impact is significant
        if weather_impact > 0:
            new_row = pd.DataFrame({
                'Feature': [f"Weather: {weather}"],
                'Contribution': [weather_impact],
                'Abs_Contribution': [weather_impact]
            })
            top_features = pd.concat([new_row, top_features], ignore_index=True)
            
        # Sort again
        top_features = top_features.sort_values(by='Abs_Contribution', ascending=False).head(5)
        
        # Show in tree
        for _, row in top_features.iterrows():
            feature_raw = row['Feature']
            
            # Determine Description
            description = ""
            if "Weather" in feature_raw:
                readable_name = feature_raw
                description = "Adverse weather conditions reduce capacity."
            else:
                readable_name = self.get_readable_feature_name(feature_raw)
                if "Month" in readable_name: description = "Seasonal traffic patterns."
                elif "Time" in readable_name: description = "Traffic congestion varies by time of day."
                elif "Airline" in readable_name: description = "Carrier operational efficiency."
                elif "Origin" in readable_name: description = "Airport congestion and infrastructure."
                elif "Destination" in readable_name: description = "Arrival airport capacity/weather."
                elif "Distance" in readable_name: description = "Longer flights have more recovery time."
                else: description = "Historical delay factor."

            if row['Contribution'] > 0:
                effect = "Increases Risk ⚠️"
                tag = "increase_risk"
            else:
                effect = "Decreases Risk ✅"
                tag = "decrease_risk"
                
            self.reasoning_tree.insert("", tk.END, values=(readable_name, effect, f"{row['Contribution']:.4f}", description), tags=(tag,))

if __name__ == "__main__":
    root = tk.Tk()
    app = DelayPredictionApp(root)
    root.mainloop()
