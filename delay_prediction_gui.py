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
                'FLIGHT_NUMBER': 1024,
                'MONTH': 12, 'DAY': 24, 'DAY_OF_WEEK': 7, 
                'SCHEDULED_DEPARTURE': 1800, 'AIRLINE': 'UA', 
                'ORIGIN_AIRPORT': 'ORD', 'DESTINATION_AIRPORT': 'JFK', 'DISTANCE': 740.0
            },
            {
                'name': 'Low Risk: Spring Mid-Week',
                'weather': 'Clear',
                'FLIGHT_NUMBER': 4521,
                'MONTH': 4, 'DAY': 15, 'DAY_OF_WEEK': 3, 
                'SCHEDULED_DEPARTURE': 900, 'AIRLINE': 'WN', 
                'ORIGIN_AIRPORT': 'PHX', 'DESTINATION_AIRPORT': 'LAS', 'DISTANCE': 255.0
            },
            {
                'name': 'Medium Risk: Summer Afternoon',
                'weather': 'Rain',
                'FLIGHT_NUMBER': 2098,
                'MONTH': 7, 'DAY': 10, 'DAY_OF_WEEK': 5, 
                'SCHEDULED_DEPARTURE': 1400, 'AIRLINE': 'DL', 
                'ORIGIN_AIRPORT': 'ATL', 'DESTINATION_AIRPORT': 'MCO', 'DISTANCE': 404.0
            },
            {
                'name': 'Long Haul: Coast to Coast',
                'weather': 'Clear',
                'FLIGHT_NUMBER': 101,
                'MONTH': 6, 'DAY': 1, 'DAY_OF_WEEK': 6, 
                'SCHEDULED_DEPARTURE': 800, 'AIRLINE': 'AA', 
                'ORIGIN_AIRPORT': 'LAX', 'DESTINATION_AIRPORT': 'JFK', 'DISTANCE': 2475.0
            },
            {
                'name': 'Short Haul: Northeast Corridor',
                'weather': 'Storm',
                'FLIGHT_NUMBER': 3305,
                'MONTH': 11, 'DAY': 27, 'DAY_OF_WEEK': 4, 
                'SCHEDULED_DEPARTURE': 1700, 'AIRLINE': 'B6', 
                'ORIGIN_AIRPORT': 'BOS', 'DESTINATION_AIRPORT': 'LGA', 'DISTANCE': 184.0
            },
            {
                'name': 'Early Morning Business',
                'weather': 'Fog',
                'FLIGHT_NUMBER': 550,
                'MONTH': 9, 'DAY': 12, 'DAY_OF_WEEK': 2, 
                'SCHEDULED_DEPARTURE': 600, 'AIRLINE': 'AS', 
                'ORIGIN_AIRPORT': 'SFO', 'DESTINATION_AIRPORT': 'SEA', 'DISTANCE': 679.0
            },
            {
                'name': 'Late Night Red-Eye',
                'weather': 'Clear',
                'FLIGHT_NUMBER': 889,
                'MONTH': 3, 'DAY': 5, 'DAY_OF_WEEK': 1, 
                'SCHEDULED_DEPARTURE': 2300, 'AIRLINE': 'NK', 
                'ORIGIN_AIRPORT': 'LAS', 'DESTINATION_AIRPORT': 'MIA', 'DISTANCE': 2174.0
            },
            {
                'name': 'Holiday Rush',
                'weather': 'Snow',
                'FLIGHT_NUMBER': 1567,
                'MONTH': 11, 'DAY': 26, 'DAY_OF_WEEK': 3, 
                'SCHEDULED_DEPARTURE': 1630, 'AIRLINE': 'UA', 
                'ORIGIN_AIRPORT': 'DEN', 'DESTINATION_AIRPORT': 'ORD', 'DISTANCE': 888.0
            },
            {
                'name': 'Regional Connector',
                'weather': 'Rain',
                'FLIGHT_NUMBER': 5200,
                'MONTH': 5, 'DAY': 20, 'DAY_OF_WEEK': 4, 
                'SCHEDULED_DEPARTURE': 1000, 'AIRLINE': 'AA', 
                'ORIGIN_AIRPORT': 'CLT', 'DESTINATION_AIRPORT': 'RDU', 'DISTANCE': 130.0
            },
            {
                'name': 'Island Hopper',
                'weather': 'Clear',
                'FLIGHT_NUMBER': 303,
                'MONTH': 1, 'DAY': 15, 'DAY_OF_WEEK': 1, 
                'SCHEDULED_DEPARTURE': 1200, 'AIRLINE': 'HA', 
                'ORIGIN_AIRPORT': 'HNL', 'DESTINATION_AIRPORT': 'OGG', 'DISTANCE': 100.0
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
        
        # --- SEARCH SECTION ---
        search_frame = ttk.Frame(header_frame, style="Header.TFrame")
        search_frame.pack(side=tk.RIGHT, padx=10)
        
        ttk.Label(search_frame, text="Flight #:", style="Header.TLabel", font=("Segoe UI", 12)).pack(side=tk.LEFT, padx=5)
        self.search_entry = ttk.Entry(search_frame, width=10, font=("Segoe UI", 12))
        self.search_entry.pack(side=tk.LEFT, padx=5)
        
        search_btn = ttk.Button(search_frame, text="Predict", command=self.predict_flight)
        search_btn.pack(side=tk.LEFT, padx=5)
        
        # --- CONTENT SECTION ---
        content_frame = ttk.Frame(main_container, padding="25")
        content_frame.pack(fill=tk.BOTH, expand=True)
        
        # --- TOP PANEL: DASHBOARD TABLE ---
        table_card = ttk.Frame(content_frame, style="Card.TFrame", padding="2") 
        table_card.pack(fill=tk.BOTH, expand=True, pady=(0, 25))
        
        table_inner = ttk.Frame(table_card, style="Card.TFrame", padding="20")
        table_inner.pack(fill=tk.BOTH, expand=True)
        
        ttk.Label(table_inner, text="Flight Simulation Scenarios", font=("Segoe UI", 16, "bold"), style="CardLabel.TLabel", foreground=ACCENT_COLOR).pack(anchor="w", pady=(0, 15))
        
        columns = ("flight_num", "flight", "date")
        self.table = ttk.Treeview(table_inner, columns=columns, show="headings", height=8)
        
        self.table.heading("flight_num", text="Flight #")
        self.table.heading("flight", text="Flight Route")
        self.table.heading("date", text="Date")
        
        self.table.column("flight_num", width=80, anchor="center")
        self.table.column("flight", width=350)
        self.table.column("date", width=100, anchor="center")
        
        # Scrollbar
        scrollbar = ttk.Scrollbar(table_inner, orient=tk.VERTICAL, command=self.table.yview)
        self.table.configure(yscroll=scrollbar.set)
        
        self.table.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        
        # Bind selection
        # Bind selection
        # self.table.bind("<<TreeviewSelect>>", self.on_scenario_select) # Removed selection binding
        self.table.bind("<Motion>", self.highlight_row)
        
        # Configure Tags for Table Colors
        self.table.tag_configure("delayed", foreground=DANGER_COLOR, font=("Segoe UI", 10, "bold"))
        self.table.tag_configure("ontime", foreground=SUCCESS_COLOR, font=("Segoe UI", 10, "bold"))
        self.table.tag_configure("evenrow", background=ROW_ALT_COLOR)
        self.table.tag_configure("hover", background=HOVER_COLOR)
        
        # --- BOTTOM PANEL REMOVED ---
        # Analysis will now be shown in a popup window triggered by the "Predict" button.

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
            data = {k: v for k, v in scenario.items() if k != 'name' and k != 'weather' and k != 'FLIGHT_NUMBER'}
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
            flight_str = f"{data['AIRLINE']} {data['ORIGIN_AIRPORT']}->{data['DESTINATION_AIRPORT']}"
            date_str = f"M{data['MONTH']}/D{data['DAY']}"
            prob_str = f"{final_prob*100:.1f}%"
            
            # Insert into table
            # Tag rows for color
            # pred_tag = "delayed" if prediction == "DELAYED" else "ontime" # Removed color tagging based on prediction to hide it
            row_tag = "evenrow" if i % 2 == 0 else "oddrow"
            
            self.table.insert("", tk.END, iid=i, values=(scenario['FLIGHT_NUMBER'], flight_str, date_str), tags=(row_tag,))

    def predict_flight(self):
        query = self.search_entry.get().strip()
        if not query:
            messagebox.showwarning("Input Error", "Please enter a Flight Number.")
            return
            
        if not query.isdigit():
            messagebox.showwarning("Input Error", "Flight Number must be numeric.")
            return
            
        flight_num = int(query)
        found = False
        
        for i, scenario in enumerate(self.scenarios):
            if scenario.get('FLIGHT_NUMBER') == flight_num:
                found = True
                # Select in table
                self.table.selection_set(i)
                self.table.see(i)
                self.on_scenario_select(None) # Trigger detail view update
                
                # Optional: Simple confirmation popup
                # messagebox.showinfo("Prediction Complete", f"Prediction for Flight {flight_num} is ready.\nSee details in the panel below.")
                break
                
        if not found:
            messagebox.showerror("Not Found", f"Flight {flight_num} not found in current scenarios.")

    def get_readable_feature_name(self, feature_name):
        """Convert technical feature names to readable text."""
        if feature_name == "MONTH": return "Month of Year"
        if feature_name == "DAY": return "Day of Month"
        if feature_name == "DAY_OF_WEEK": return "Day of Week"
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
        
        # --- SEARCH SECTION ---
        search_frame = ttk.Frame(header_frame, style="Header.TFrame")
        search_frame.pack(side=tk.RIGHT, padx=10)
        
        ttk.Label(search_frame, text="Flight #:", style="Header.TLabel", font=("Segoe UI", 12)).pack(side=tk.LEFT, padx=5)
        self.search_entry = ttk.Entry(search_frame, width=10, font=("Segoe UI", 12))
        self.search_entry.pack(side=tk.LEFT, padx=5)
        
        search_btn = ttk.Button(search_frame, text="Predict", command=self.predict_flight)
        search_btn.pack(side=tk.LEFT, padx=5)
        
        # --- CONTENT SECTION ---
        content_frame = ttk.Frame(main_container, padding="25")
        content_frame.pack(fill=tk.BOTH, expand=True)
        
        # --- TOP PANEL: DASHBOARD TABLE ---
        table_card = ttk.Frame(content_frame, style="Card.TFrame", padding="2") 
        table_card.pack(fill=tk.BOTH, expand=True, pady=(0, 25))
        
        table_inner = ttk.Frame(table_card, style="Card.TFrame", padding="20")
        table_inner.pack(fill=tk.BOTH, expand=True)
        
        ttk.Label(table_inner, text="Flight Simulation Scenarios", font=("Segoe UI", 16, "bold"), style="CardLabel.TLabel", foreground=ACCENT_COLOR).pack(anchor="w", pady=(0, 15))
        
        columns = ("flight_num", "flight", "date")
        self.table = ttk.Treeview(table_inner, columns=columns, show="headings", height=8)
        
        self.table.heading("flight_num", text="Flight #")
        self.table.heading("flight", text="Flight Route")
        self.table.heading("date", text="Date")
        
        self.table.column("flight_num", width=80, anchor="center")
        self.table.column("flight", width=350)
        self.table.column("date", width=100, anchor="center")
        
        # Scrollbar
        scrollbar = ttk.Scrollbar(table_inner, orient=tk.VERTICAL, command=self.table.yview)
        self.table.configure(yscroll=scrollbar.set)
        
        self.table.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        
        # Bind selection
        # Bind selection
        # self.table.bind("<<TreeviewSelect>>", self.on_scenario_select) # Removed selection binding
        self.table.bind("<Motion>", self.highlight_row)
        
        # Configure Tags for Table Colors
        self.table.tag_configure("delayed", foreground=DANGER_COLOR, font=("Segoe UI", 10, "bold"))
        self.table.tag_configure("ontime", foreground=SUCCESS_COLOR, font=("Segoe UI", 10, "bold"))
        self.table.tag_configure("evenrow", background=ROW_ALT_COLOR)
        self.table.tag_configure("hover", background=HOVER_COLOR)
        
        # --- BOTTOM PANEL REMOVED ---
        # Analysis will now be shown in a popup window triggered by the "Predict" button.

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
            data = {k: v for k, v in scenario.items() if k != 'name' and k != 'weather' and k != 'FLIGHT_NUMBER'}
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
            flight_str = f"{data['AIRLINE']} {data['ORIGIN_AIRPORT']}->{data['DESTINATION_AIRPORT']}"
            date_str = f"M{data['MONTH']}/D{data['DAY']}"
            prob_str = f"{final_prob*100:.1f}%"
            
            # Insert into table
            # Tag rows for color
            # pred_tag = "delayed" if prediction == "DELAYED" else "ontime" # Removed color tagging based on prediction to hide it
            row_tag = "evenrow" if i % 2 == 0 else "oddrow"
            
            self.table.insert("", tk.END, iid=i, values=(scenario['FLIGHT_NUMBER'], flight_str, date_str), tags=(row_tag,))

    def predict_flight(self):
        query = self.search_entry.get().strip()
        if not query:
            messagebox.showwarning("Input Error", "Please enter a Flight Number.")
            return
            
        if not query.isdigit():
            messagebox.showwarning("Input Error", "Flight Number must be numeric.")
            return
            
        flight_num = int(query)
        found = False
        
        for i, scenario in enumerate(self.scenarios):
            if scenario.get('FLIGHT_NUMBER') == flight_num:
                found = True
                result = self.predictions_cache[i]
                
                # Select in table
                self.table.selection_set(i)
                self.table.see(i)
                
                # Show Popup
                self.show_prediction_popup(result)
                break
                
        if not found:
            messagebox.showerror("Not Found", f"Flight {flight_num} not found in current scenarios.")

    def show_prediction_popup(self, result):
        popup = tk.Toplevel(self.root)
        popup.title(f"Flight {result['data'].get('FLIGHT_NUMBER', 'Unknown')} Prediction Analysis")
        popup.geometry("900x500")
        popup.configure(bg="#ffffff")
        
        # Styles for popup
        ACCENT_COLOR = "#3949ab"
        SUBTEXT_COLOR = "#546e7a"
        DANGER_COLOR = "#d50000"
        SUCCESS_COLOR = "#00c853"
        
        # Main Container
        container = ttk.Frame(popup, padding="20")
        container.pack(fill=tk.BOTH, expand=True)
        
        # --- LEFT SIDE: DETAILS ---
        details_frame = ttk.Frame(container)
        details_frame.pack(side=tk.LEFT, fill=tk.Y, padx=(0, 20))
        
        ttk.Label(details_frame, text="Prediction Result", font=("Segoe UI", 16, "bold"), foreground=ACCENT_COLOR).pack(anchor="w", pady=(0, 20))
        
        # Flight Info
        flight_info = f"{result['data']['AIRLINE']} {result['data']['ORIGIN_AIRPORT']} -> {result['data']['DESTINATION_AIRPORT']}"
        ttk.Label(details_frame, text=flight_info, font=("Segoe UI", 12)).pack(anchor="w", pady=(0, 5))
        ttk.Label(details_frame, text=f"Weather: {result['weather']}", font=("Segoe UI", 12)).pack(anchor="w", pady=(0, 15))
        
        # Prediction
        color = DANGER_COLOR if result['prediction'] == "DELAYED" else SUCCESS_COLOR
        ttk.Label(details_frame, text=result['prediction'], font=("Segoe UI", 28, "bold"), foreground=color).pack(anchor="w", pady=(0, 10))
        ttk.Label(details_frame, text=f"Probability: {result['prob']*100:.2f}%", font=("Segoe UI", 14)).pack(anchor="w")
        
        # --- RIGHT SIDE: FACTORS ---
        reasoning_frame = ttk.Frame(container)
        reasoning_frame.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True)
        
        ttk.Label(reasoning_frame, text="Top Contributing Factors", font=("Segoe UI", 12, "bold"), foreground=ACCENT_COLOR).pack(anchor="w", pady=(0, 5))
        ttk.Label(reasoning_frame, text="Impact Score: (+) Increases Risk, (-) Decreases Risk", font=("Segoe UI", 9), foreground=SUBTEXT_COLOR).pack(anchor="w", pady=(0, 10))
        
        r_columns = ("feature", "effect", "impact", "desc")
        reasoning_tree = ttk.Treeview(reasoning_frame, columns=r_columns, show="headings", height=10)
        
        reasoning_tree.heading("feature", text="Factor")
        reasoning_tree.heading("effect", text="Effect")
        reasoning_tree.heading("impact", text="Impact Score")
        reasoning_tree.heading("desc", text="Explanation")
        
        reasoning_tree.column("feature", width=180)
        reasoning_tree.column("effect", width=100)
        reasoning_tree.column("impact", width=80, anchor="center")
        reasoning_tree.column("desc", width=220)
        
        reasoning_tree.pack(fill=tk.BOTH, expand=True)
        
        # Configure Tags
        reasoning_tree.tag_configure("increase_risk", foreground=DANGER_COLOR)
        reasoning_tree.tag_configure("decrease_risk", foreground=SUCCESS_COLOR)
        
        # Populate Reasoning
        self.update_reasoning_in_popup(reasoning_tree, result['data'], result['weather'], result['weather_impact'])

    def get_readable_feature_name(self, feature_name):
        """Convert technical feature names to readable text."""
        if feature_name == "MONTH": return "Month of Year"
        if feature_name == "DAY": return "Day of Month"
        if feature_name == "DAY_OF_WEEK": return "Day of Week"
        if feature_name == "SCHEDULED_DEPARTURE": return "Scheduled Departure Time"
        if feature_name == "DISTANCE": return "Flight Distance"
        
        if feature_name.startswith("AIRLINE_"):
            return f"Airline: {feature_name.split('_')[-1]}"
        if feature_name.startswith("ORIGIN_AIRPORT_"):
            return f"Origin Airport: {feature_name.split('_')[-1]}"
        if feature_name.startswith("DESTINATION_AIRPORT_"):
            return f"Destination Airport: {feature_name.split('_')[-1]}"
            
        return feature_name

    def on_scenario_select(self, event):
        # No longer needed for updating bottom panel, but kept for future use or just to handle selection gracefully
        pass

    def update_reasoning_in_popup(self, tree, data, weather, weather_impact):
        # Clear tree (though it's new)
        for item in tree.get_children():
            tree.delete(item)
            
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
                       .get_feature_names_out(['AIRLINE', 'ORIGIN_AIRPORT', 'DESTINATION_AIRPORT']))
        num_features = ['MONTH', 'DAY', 'DAY_OF_WEEK', 'SCHEDULED_DEPARTURE', 'DISTANCE']
        all_features = np.concatenate([num_features, cat_features])
        
        explanation = pd.DataFrame({
            'Feature': all_features,
            'Contribution': contributions
        })
        
        explanation['Abs_Contribution'] = explanation['Contribution'].abs()
        # Filter out features with no contribution (zeros)
        explanation = explanation[explanation['Abs_Contribution'] > 0.0001]
        explanation = explanation.sort_values(by='Abs_Contribution', ascending=False)
        
        explanation['Abs_Contribution'] = explanation['Contribution'].abs()
        explanation = explanation.sort_values(by='Abs_Contribution', ascending=False)
        
        # Combine with Weather
        top_features = explanation.copy() # Show all features
        
        # Add Weather Row if impact is significant
        if weather_impact > 0:
            new_row = pd.DataFrame({
                'Feature': [f"Weather: {weather}"],
                'Contribution': [weather_impact],
                'Abs_Contribution': [weather_impact]
            })
            top_features = pd.concat([new_row, top_features], ignore_index=True)
            
        # Sort again
        top_features = top_features.sort_values(by='Abs_Contribution', ascending=False) # Show all features
        
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
                
                # Custom handling for Day of Week to show specific day
                if "Day of Week" in readable_name:
                    day_map = {1: "Monday", 2: "Tuesday", 3: "Wednesday", 4: "Thursday", 5: "Friday", 6: "Saturday", 7: "Sunday"}
                    day_val = data.get('DAY_OF_WEEK')
                    day_name = day_map.get(day_val, "Unknown")
                    readable_name = f"Day: {day_name}"
                    
                    if row['Contribution'] > 0:
                        description = f"High traffic volume on {day_name}s increases delay risk."
                    else:
                        description = f"Lower traffic volume on {day_name}s reduces delay risk."
                
                elif "Day of Month" in readable_name: 
                    description = "Mid-month vs. end-of-month travel trends."
                
                elif "Month" in readable_name:
                    month_map = {1: "January", 2: "February", 3: "March", 4: "April", 5: "May", 6: "June", 
                                 7: "July", 8: "August", 9: "September", 10: "October", 11: "November", 12: "December"}
                    month_val = data.get('MONTH')
                    month_name = month_map.get(month_val, "Unknown")
                    readable_name = f"Month: {month_name}"
                    description = f"Seasonal traffic patterns in {month_name}."
                elif "Time" in readable_name: description = "Traffic congestion varies by time of day."
                elif "Airline" in readable_name: description = "Carrier operational efficiency and fleet reliability."
                elif "Origin" in readable_name: description = "Origin airport congestion and infrastructure."
                elif "Destination" in readable_name: description = "Destination airport capacity and local conditions."
                elif "Distance" in readable_name: description = "Longer flights have more in-air recovery time."
                else: description = "Factor based on historical flight data trends."

            if row['Contribution'] > 0:
                effect = "Increases Risk ⚠️"
                tag = "increase_risk"
            else:
                effect = "Decreases Risk ✅"
                tag = "decrease_risk"
                
            tree.insert("", tk.END, values=(readable_name, effect, f"{row['Contribution']:.4f}", description), tags=(tag,))

if __name__ == "__main__":
    root = tk.Tk()
    app = DelayPredictionApp(root)
    root.mainloop()
