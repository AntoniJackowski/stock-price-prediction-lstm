import os
import time
import ctypes
import threading
import tkinter as tk
from tkinter import ttk, messagebox, filedialog
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import matplotlib.ticker as ticker
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from pandas.tseries.offsets import BDay

import joblib
from tensorflow.keras.models import load_model
import data_prep

# ==========================================
# CONFIGURATION & CONSTANTS
# ==========================================
BASE_DIR = Path(__file__).resolve().parent.parent
DATA_PATH = BASE_DIR / "data" / "Gold_features.csv"

FEATURE_COLUMNS = [
    "Close", "Volume", "RSI", "MACD",
    "MACD_Signal", "EMA_20", "BB_High", "BB_Low", "ATR"
]
TIME_STEPS = 30
DATA_UPDATE_INTERVAL = 86400  # 24 hours in seconds

# UI Colors
BG = "#0f172a"
PANEL = "#111827"
CARD = "#1e293b"
ACCENT = "#38bdf8"
ACCENT_GREEN = "#22c55e"
TEXT = "#f8fafc"
MUTED = "#94a3b8"
BUTTON = "#2563eb"
BUTTON_HOVER = "#1d4ed8"


class GoldForecastApp:
    """
    Main application class for the Gold Forecast AI.
    Handles UI rendering, user interactions, data loading, and LSTM inference.
    """

    def __init__(self, root: tk.Tk):
        self.root = root
        self.root.title("Gold Forecast AI - LSTM")

        # Set a safe fallback size for smaller laptop screens
        self.root.geometry("1100x600")
        self.root.minsize(1024, 600)

        # Automatically launch the application in a maximized window state.
        try:
            self.root.state('zoomed')
        except tk.TclError:
            self.root.geometry("1100x600")

        # Define path to the assets folder and set the custom window icon
        icon_path = BASE_DIR / "assets" / "icon.ico"
        try:
            self.root.iconbitmap(icon_path)
        except Exception:
            # Fails silently if the icon file is missing or the OS doesn't support .ico
            pass

        self.root.configure(bg=BG)

        # Application state variables
        self.gold_data = None
        self.chart_canvas = None
        self.current_figure = None
        self.layout_state = "expanded"  # Tracks current layout density (expanded/compact)

        # Timer used to debounce rapid window resize events
        self._resize_timer = None

        self._configure_styles()
        self._build_ui()

        # Catch window resizing events to toggle responsive layouts dynamically
        self.root.bind("<Configure>", self._adjust_vertical_layout)

    def _configure_styles(self):
        """Configures ttk styles for consistent UI appearance."""
        style = ttk.Style()
        style.theme_use("clam")
        style.configure(
            "TCombobox",
            fieldbackground="white",
            background="white",
            foreground="black",
            arrowcolor="black",
            selectbackground="white",
            selectforeground="black"
        )

    # ==========================================
    # UI CONSTRUCTION
    # ==========================================

    def _build_ui(self):
        """Assembles all main sections of the graphical user interface."""
        self._build_header()

        self.main_frame = tk.Frame(self.root, bg=BG)
        self.main_frame.pack(fill="both", expand=True, padx=30, pady=20)

        self._build_left_panel()
        self._build_right_panel()

    def _build_header(self):
        """Constructs the application header with titles."""
        header = tk.Frame(self.root, bg=BG)
        # Reduced vertical padding to save screen space
        header.pack(fill="x", padx=25, pady=(15, 5))

        title = tk.Label(
            header,
            text="Gold Forecast AI",
            font=("Segoe UI", 24, "bold"),  # Smaller font size (was 30)
            fg=TEXT,
            bg=BG
        )
        title.pack(anchor="w")

        self.subtitle = tk.Label(
            header,
            text="Gold price prediction using an LSTM network and advanced "
                 "indicators (RSI, MACD, EMA, Bollinger Bands, ATR)",
            font=("Segoe UI", 11),  # Smaller subtitle font
            fg=MUTED,
            bg=BG
        )
        self.subtitle.pack(anchor="w", pady=(2, 0))

    def _build_left_panel(self):
        """Constructs the sidebar containing inputs, controls, and metric cards."""
        self.left_panel = tk.Frame(self.main_frame, bg=PANEL, width=300)
        self.left_panel.pack(side="left", fill="y", padx=(0, 15))
        self.left_panel.pack_propagate(False)

        panel_title = tk.Label(
            self.left_panel,
            text="Investor Panel",
            font=("Segoe UI", 18, "bold"),
            fg=TEXT,
            bg=PANEL
        )
        # Reduced vertical padding below the title
        panel_title.pack(anchor="w", padx=25, pady=(15, 10))

        # Inputs
        self._create_sidebar_label("PRICE HISTORY (LAST DAYS)")
        self.history_entry = self._create_sidebar_entry("50")

        self._create_sidebar_label("FORECAST (UPCOMING DAYS)")
        self.forecast_entry = self._create_sidebar_entry("20")

        # Action Buttons (tightened spacing)
        self.show_button = tk.Button(
            self.left_panel, text="Show Historical Data",
            font=("Segoe UI", 11, "bold"),
            bg=BUTTON, fg="white", activebackground=BUTTON_HOVER,
            activeforeground="white",
            relief="flat", cursor="hand2", height=2,
            command=self.show_historical_chart
        )
        self.show_button.pack(fill="x", padx=25, pady=(0, 8))

        self.forecast_button = tk.Button(
            self.left_panel, text="Generate LSTM Forecast",
            font=("Segoe UI", 11, "bold"),
            bg=ACCENT_GREEN, fg="#052e16", activebackground="#16a34a",
            activeforeground="white",
            relief="flat", cursor="hand2", height=2,
            command=self.start_forecast_thread
        )
        self.forecast_button.pack(fill="x", padx=25)

        self.save_chart_button = tk.Button(
            self.left_panel, text="Save Chart as PNG",
            font=("Segoe UI", 11, "bold"),
            bg="#f59e0b", fg="#111827", activebackground="#d97706",
            activeforeground="white",
            relief="flat", cursor="hand2", height=2, command=self.save_chart
        )
        self.save_chart_button.pack(fill="x", padx=25, pady=(8, 0))

        # Status & Metric Cards container
        self.cards_frame = tk.Frame(self.left_panel, bg=PANEL)
        self.cards_frame.pack(fill="x", padx=25, pady=(18, 5))

        # Unpack the frame, value label, and title label from the helper method
        self.current_price_card, self.current_price_label, self.current_title_lbl = self._create_metric_card(
            "CURRENT PRICE", "-", ACCENT
        )
        self.forecast_price_card, self.forecast_price_label, self.forecast_title_lbl = self._create_metric_card(
            "FORECASTED PRICE", "-", ACCENT_GREEN
        )

        # Status Label (anchored to the absolute bottom of the left panel)
        self.status_label = tk.Label(
            self.left_panel, text="Status: Waiting for data",
            font=("Segoe UI", 10), fg=MUTED, bg=PANEL
        )
        # side="bottom" forces the widget to stick to the lower edge,
        # pushing all available empty space above it.
        self.status_label.pack(side="bottom", anchor="w", padx=25,
                               pady=(0, 25))

    def _build_right_panel(self):
        """Constructs the main dashboard area for charts and key details."""
        self.right_panel = tk.Frame(self.main_frame, bg=BG)
        self.right_panel.pack(side="right", fill="both", expand=True)

        # Top Information Cards
        top_cards = tk.Frame(self.right_panel, bg=BG)
        top_cards.pack(fill="x", pady=(0, 20))

        self._create_small_metric(top_cards, "MODEL", "LSTM", pad_x=(0, 16))

        # Save a reference to the indicators label specifically for responsive resizing
        self.indicators_label = self._create_small_metric(
            top_cards, "INDICATORS", "RSI, MACD +3", pad_x=(0, 16)
        )

        self._create_small_metric(top_cards, "ASSET", "GOLD", pad_x=(0, 16))
        self._create_small_metric(top_cards, "DATA", "REAL-TIME", pad_x=(0, 0))

        # Bind the resize event (<Configure>) to the top cards container
        # This will trigger the text adjustment whenever the window width changes
        top_cards.bind("<Configure>", self._update_indicators_text)
        
        # Chart Container
        self.chart_panel = tk.Frame(self.right_panel, bg=CARD)
        self.chart_panel.pack(fill="both", expand=True)

        chart_title = tk.Label(
            self.chart_panel, text="Gold Price Chart and Forecast",
            font=("Segoe UI", 18, "bold"), fg=TEXT, bg=CARD
        )
        chart_title.pack(anchor="w", padx=25, pady=(25, 10))

        self.chart_placeholder = tk.Label(
            self.chart_panel,
            text=(
                "The chart will be displayed here:\n\n"
                "• Historical gold price trends\n"
                "• Autoregressive LSTM predictions for upcoming days\n"
                "• Visual comparison of market data and AI forecast\n\n"
                "Select parameters on the left to begin analysis."
            ),
            font=("Segoe UI", 15), fg=MUTED, bg=CARD, justify="center"
        )
        self.chart_placeholder.pack(expand=True)

    def _update_indicators_text(self, event):
        """
        Dynamically adjusts the indicators text based on the available width
        of the top cards container. Acts like a CSS media query.
        """
        w = event.width

        # Depending on the pixel width of the container, we show more or fewer indicators
        if w > 1200:
            self.indicators_label.config(text="RSI, MACD, EMA, BB, ATR")
        elif w > 950:
            self.indicators_label.config(text="RSI, MACD, EMA +2")
        elif w > 750:
            self.indicators_label.config(text="RSI, MACD +3")
        else:
            self.indicators_label.config(text="RSI, +4")

    def _adjust_vertical_layout(self, event):
        """
        Catches the resize event but delays the actual UI update (Debouncing).
        This prevents Tkinter from lagging and getting trapped in an infinite layout loop.
        """
        if event.widget == self.root:
            # Cancel the previous timer if the user is still resizing the window
            if self._resize_timer is not None:
                self.root.after_cancel(self._resize_timer)

            # Wait 150ms after the resizing stops before applying heavy UI changes
            self._resize_timer = self.root.after(
                150,
                lambda: self._apply_layout_changes(self.root.winfo_height())
            )

    def _apply_layout_changes(self, h):
        """
        Performs the actual heavy lifting of hiding/showing widgets based on height.
        """
        if h < 750  :
            if self.layout_state != "compact":
                self.layout_state = "compact"

                if self.subtitle.winfo_ismapped():
                    self.subtitle.pack_forget()

                self.current_price_card.pack_forget()
                self.forecast_price_card.pack_forget()

                self.current_title_lbl.config(text="CURRENT",
                                              font=("Segoe UI", 8, "bold"))
                self.forecast_title_lbl.config(text="FORECAST",
                                               font=("Segoe UI", 8, "bold"))
                self.current_price_label.config(font=("Segoe UI", 13, "bold"))
                self.forecast_price_label.config(font=("Segoe UI", 13, "bold"))

                self.current_price_card.pack(side="left", fill="both",
                                             expand=True, padx=(0, 4))
                self.forecast_price_card.pack(side="left", fill="both",
                                              expand=True, padx=(4, 0))

                self.cards_frame.pack_configure(pady=(12, 5))
                self.status_label.pack_configure(pady=(0, 10))
        else:
            if self.layout_state != "expanded":
                self.layout_state = "expanded"

                if not self.subtitle.winfo_ismapped():
                    self.subtitle.pack(anchor="w", pady=(2, 0))

                self.current_price_card.pack_forget()
                self.forecast_price_card.pack_forget()

                self.current_title_lbl.config(text="CURRENT PRICE",
                                              font=("Segoe UI", 9, "bold"))
                self.forecast_title_lbl.config(text="FORECASTED PRICE",
                                               font=("Segoe UI", 9, "bold"))
                self.current_price_label.config(font=("Segoe UI", 16, "bold"))
                self.forecast_price_label.config(font=("Segoe UI", 16, "bold"))

                self.current_price_card.pack(fill="x", pady=4)
                self.forecast_price_card.pack(fill="x", pady=4)

                self.cards_frame.pack_configure(pady=(22, 10))
                self.status_label.pack_configure(pady=(0, 25))

    # UI Helpers
    def _create_sidebar_label(self, text: str):
        tk.Label(self.left_panel, text=text, font=("Segoe UI", 10, "bold"),
                 fg=MUTED, bg=PANEL).pack(anchor="w", padx=25)

    def _create_sidebar_entry(self, default_val: str) -> tk.Entry:
        """Helper to create text entry fields with tighter vertical spacing."""
        entry = tk.Entry(self.left_panel, font=("Segoe UI", 11), bg=CARD,
                         fg=TEXT, insertbackground=TEXT, relief="flat")
        entry.insert(0, default_val)
        # Reduced ipady from 8 to 5, and bottom margin from 20 to 10
        entry.pack(fill="x", padx=25, pady=(2, 10), ipady=5)
        return entry

    def _create_metric_card(self, title_text: str, value_text: str,
                            color: str):
        """Helper to create bottom data cards. Returns frame, value, and title widgets."""
        frame = tk.Frame(self.cards_frame, bg=CARD)
        frame.pack(fill="x", pady=4)

        title_lbl = tk.Label(frame, text=title_text,
                             font=("Segoe UI", 9, "bold"),
                             fg=MUTED, bg=CARD)
        title_lbl.pack(anchor="w", padx=15, pady=(8, 0))

        val_label = tk.Label(frame, text=value_text,
                             font=("Segoe UI", 16, "bold"), fg=color, bg=CARD)
        val_label.pack(anchor="w", padx=15, pady=(0, 8))

        return frame, val_label, title_lbl

    def _create_small_metric(self, parent: tk.Frame, title_text: str,
                             value_text: str, pad_x: tuple = (8, 8)):
        """
        Creates a small metric card with customizable horizontal padding.
        Returns the value label widget so it can be updated dynamically later.
        """
        frame = tk.Frame(parent, bg=CARD, height=70)

        frame.pack(side="left", fill="both", expand=True, padx=pad_x)
        frame.pack_propagate(False)

        tk.Label(
            frame, text=title_text, font=("Segoe UI", 8, "bold"), fg=MUTED,
            bg=CARD
        ).pack(anchor="w", padx=15, pady=(10, 0))

        # Assign the label to the 'val_label' variable before packing it
        val_label = tk.Label(
            frame, text=value_text, font=("Segoe UI", 16, "bold"), fg=TEXT,
            bg=CARD
        )
        val_label.pack(anchor="w", padx=15, pady=(2, 0))

        # Return the label reference to allow dynamic text updates
        return val_label

    # ==========================================
    # BUSINESS LOGIC & DATA HANDLING
    # ==========================================

    def load_gold_data(self):
        """
        Loads gold dataset from the CSV file. If the file is missing or older
        than the update interval (24h), it triggers the data_prep script.
        """
        file_exists = DATA_PATH.exists()
        needs_update = False

        if file_exists:
            file_age = time.time() - os.path.getmtime(DATA_PATH)
            if file_age > DATA_UPDATE_INTERVAL:
                needs_update = True

        if not file_exists or needs_update:
            self.status_label.config(
                text="Status: Updating data from Yahoo Finance...")
            self.root.update_idletasks()
            try:
                data_prep.prepare_gold_data()
            except Exception as e:
                messagebox.showerror("Download Error",
                                     f"Failed to update data:\n{e}")
                self.status_label.config(text="Status: Update failed")
                return

        try:
            self.gold_data = pd.read_csv(DATA_PATH)
            self.gold_data["Date"] = pd.to_datetime(self.gold_data["Date"])
            self.gold_data.sort_values("Date", inplace=True)
            self.gold_data.dropna(inplace=True)

            last_price = self.gold_data["Close"].iloc[-1]
            self.current_price_label.config(text=f"${last_price:.2f}")
            self.status_label.config(
                text=f"Status: Loaded {len(self.gold_data)} market records")
        except Exception as e:
            messagebox.showerror("File Error",
                                 f"Could not load historical data:\n{e}")
            self.status_label.config(text="Status: Data load error")

    # ==========================================
    # CHART RENDERING & TOOLTIPS
    # ==========================================

    def _clear_chart_area(self):
        """Removes placeholder text and clears the previous canvas."""
        self.chart_placeholder.pack_forget()
        if self.chart_canvas is not None:
            self.chart_canvas.get_tk_widget().destroy()

    def _render_canvas(self, fig):
        """Embeds the matplotlib figure into the Tkinter window."""
        fig.tight_layout()
        self.current_figure = fig
        self.chart_canvas = FigureCanvasTkAgg(fig, master=self.chart_panel)
        self.chart_canvas.draw()
        self.chart_canvas.get_tk_widget().pack(fill="both", expand=True,
                                               padx=25, pady=20)

    def show_historical_chart(self):
        """Validates input and displays the historical price chart."""
        if self.gold_data is None:
            self.load_gold_data()
            if self.gold_data is None:
                return

        try:
            days = int(self.history_entry.get())
            if not (2 <= days <= 1800):
                raise ValueError("Out of bounds")
        except ValueError:
            messagebox.showerror("Validation Error",
                                 "The number of history days must be an integer between 2 and 1800.")
            return

        filtered_data = self.gold_data.tail(days)
        self.current_price_label.config(
            text=f"${filtered_data['Close'].iloc[-1]:.2f}")
        self.status_label.config(
            text=f"Status: Displaying last {len(filtered_data)} trading days")

        self._clear_chart_area()

        fig, ax = plt.subplots(figsize=(9, 4))
        fig.patch.set_facecolor(CARD)
        ax.set_facecolor(CARD)

        ax.plot(filtered_data["Date"], filtered_data["Close"], linewidth=2,
                label="Close Price", color=ACCENT)

        self._style_axes(ax,
                         f"Historical Gold Price — Last {len(filtered_data)} Days")

        self._render_canvas(fig)
        self.add_precise_tooltip(fig, ax, [("Close Price",
                                            filtered_data["Date"].to_numpy(),
                                            filtered_data[
                                                "Close"].to_numpy())])

    def show_forecast_chart(self, future_predictions):
        """Renders both historical data and the newly generated AI forecast."""
        try:
            history_days = int(self.history_entry.get())
        except ValueError:
            history_days = 50

        historical_data = self.gold_data.tail(history_days)
        last_date = historical_data["Date"].iloc[-1]

        # Generate future business days (excluding weekends)
        future_dates = pd.bdate_range(start=last_date + BDay(1),
                                      periods=len(future_predictions))

        self._clear_chart_area()

        fig, ax = plt.subplots(figsize=(9, 4))
        fig.patch.set_facecolor(CARD)
        ax.set_facecolor(CARD)

        # Plot historical line
        ax.plot(historical_data["Date"], historical_data["Close"], linewidth=2,
                label="Historical Price", color=ACCENT)
        # Plot forecast line
        ax.plot(future_dates, future_predictions, linewidth=2, linestyle="--",
                label="LSTM Forecast", color=ACCENT_GREEN)

        self._style_axes(ax, "Gold Price: Historical Data vs LSTM Forecast")

        self._render_canvas(fig)
        self.add_precise_tooltip(fig, ax, [
            ("Historical Price", historical_data["Date"].to_numpy(),
             historical_data["Close"].to_numpy()),
            ("LSTM Forecast", future_dates.to_numpy(),
             np.array(future_predictions))
        ])

    def _style_axes(self, ax, title: str):
        """Applies consistent styling to matplotlib axes."""
        ax.set_title(title, color=TEXT, fontsize=14, fontweight="bold")

        ax.set_xlabel("Date", color=MUTED, labelpad=12)
        ax.set_ylabel("Price (USD)", color=MUTED, labelpad=12)

        # Enforce a strict single-line date format for major labels
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))

        # Disable minor locators and formatters to prevent rendering artifacts
        ax.xaxis.set_minor_locator(ticker.NullLocator())
        ax.xaxis.set_minor_formatter(ticker.NullFormatter())

        # Style major ticks: remove physical lines and adjust text padding
        ax.tick_params(axis="x", which="major", colors=MUTED, rotation=30,
                       length=0, pad=5)
        ax.tick_params(axis="y", which="major", colors=MUTED, length=0, pad=5)

        ax.grid(True, alpha=0.2)

        # Remove top and right spines for a cleaner, modern web-like appearance
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

        # Subtly color the remaining left and bottom spines
        ax.spines["left"].set_color(MUTED)
        ax.spines["bottom"].set_color(MUTED)

        legend = ax.legend()
        legend.get_frame().set_facecolor(CARD)
        legend.get_frame().set_edgecolor(CARD)
        for text in legend.get_texts():
            text.set_color(TEXT)

    def add_precise_tooltip(self, fig, ax, series_list):
        """Attaches an interactive tooltip that tracks the mouse over plot lines."""

        # Tooltip with a dark background, but with border and arrow matched to the Save button
        tooltip = ax.annotate(
            "", xy=(0, 0), xytext=(15, 15), textcoords="offset points",
            bbox=dict(
                boxstyle="round,pad=0.5",
                fc="#020617",  # Restored original dark background
                ec="#f59e0b",  # Orange border matching the save button
                lw=1.2,
                alpha=0.95
            ),
            arrowprops=dict(arrowstyle="->", color="#f59e0b", lw=1.2),
            # Orange arrow
            color=TEXT,  # Restored original light text color
            fontsize=10
        )
        tooltip.set_visible(False)

        marker, = ax.plot([], [], marker="o", markersize=7, color=TEXT,
                          linestyle="None", zorder=10)

        def on_move(event):
            if event.inaxes != ax or event.xdata is None:
                tooltip.set_visible(False)
                marker.set_data([], [])
                self.chart_canvas.draw_idle()
                return

            closest_point = None
            min_distance = float("inf")

            # Find the nearest data point to the cursor
            for name, dates, values in series_list:
                x_values = mdates.date2num(pd.to_datetime(dates))
                index = np.searchsorted(x_values, event.xdata)

                possible_indexes = [i for i in (index - 1, index) if
                                    0 <= i < len(x_values)]

                for i in possible_indexes:
                    distance = abs(x_values[i] - event.xdata)
                    if distance < min_distance:
                        min_distance = distance
                        closest_point = (name, x_values[i], values[i])

            if closest_point is None:
                return

            name, x_point, y_point = closest_point
            date_text = mdates.num2date(x_point).strftime("%Y-%m-%d")

            tooltip.xy = (x_point, y_point)
            tooltip.set_text(
                f"{name}\nDate: {date_text}\nPrice: ${y_point:.2f}")

            # Keep tooltip inside canvas boundaries
            canvas_width = fig.canvas.get_width_height()[0]
            if event.x > canvas_width * 0.75:
                tooltip.set_position((-140, 15))
            else:
                tooltip.set_position((15, 15))

            marker.set_data([x_point], [y_point])
            tooltip.set_visible(True)
            self.chart_canvas.draw_idle()

        fig.canvas.mpl_connect("motion_notify_event", on_move)

    # ==========================================
    # LSTM PREDICTION LOGIC
    # ==========================================

    def start_forecast_thread(self):
        """Disables UI inputs and starts the prediction algorithm in a background thread."""
        self.forecast_button.config(state="disabled",
                                    text="AI is analyzing...")
        self.show_button.config(state="disabled")
        self.status_label.config(text="Status: AI process running...")

        # Change the mouse cursor to a loading spinner
        self.root.config(cursor="watch")
        self.root.update_idletasks()

        thread = threading.Thread(target=self.generate_lstm_forecast,
                                  daemon=True)
        thread.start()

    def generate_lstm_forecast(self):
        """
        Core forecasting method.
        Loads ML assets, scales data, runs the autoregressive loop, and updates the UI.
        """
        self.load_gold_data()
        if self.gold_data is None:
            self._reset_buttons()
            return

        try:
            forecast_days = int(self.forecast_entry.get())
            if not (2 <= forecast_days <= 30):
                raise ValueError("Out of bounds")
        except ValueError:
            messagebox.showerror("Validation Error",
                                 "Forecast days must be an integer between 2 and 30.")
            self._reset_buttons()
            return

        try:
            # 1. Load frozen model and scalers
            model_path = BASE_DIR / "data" / "best_gold_model.keras"
            feature_scaler = joblib.load(
                BASE_DIR / "data" / "feature_scaler.save")
            target_scaler = joblib.load(
                BASE_DIR / "data" / "target_scaler.save")
            model = load_model(model_path)

            # 2. Prepare sequences
            data = self.gold_data[FEATURE_COLUMNS].dropna()
            scaled_features = feature_scaler.transform(data[FEATURE_COLUMNS])
            last_sequence = scaled_features[-TIME_STEPS:].copy()
            future_predictions = []

            # 3. Autoregressive prediction loop
            for _ in range(forecast_days):
                input_data = last_sequence.reshape(1, TIME_STEPS,
                                                   len(FEATURE_COLUMNS))
                predicted_scaled_close = \
                model.predict(input_data, verbose=0)[0][0]

                # Inverse transform to get real USD value
                predicted_close = \
                target_scaler.inverse_transform([[predicted_scaled_close]])[0][
                    0]
                future_predictions.append(predicted_close)

                # Slide window forward and append new prediction
                new_sequence = np.roll(last_sequence, -1, axis=0)
                new_sequence[-1, 0] = predicted_scaled_close
                last_sequence = new_sequence

            # 4. Update UI with results
            last_real_price = self.gold_data["Close"].iloc[-1]
            last_forecast_price = future_predictions[-1]

            self.current_price_label.config(text=f"${last_real_price:.2f}")
            self.forecast_price_label.config(
                text=f"${last_forecast_price:.2f}")

            self.show_forecast_chart(future_predictions)
            self.status_label.config(
                text=f"Status: Forecast generated for {forecast_days} days")

        except Exception as e:
            messagebox.showerror("AI Critical Error",
                                 f"Model execution failed:\n{str(e)}")
            self.status_label.config(text="Status: Prediction error")

        finally:
            self._reset_buttons()

    def _reset_buttons(self):
        """Restores UI buttons to their interactive state after a background process finishes."""
        self.forecast_button.config(state="normal",
                                    text="Generate LSTM Forecast")
        self.show_button.config(state="normal")

        # Restore the default mouse pointer
        self.root.config(cursor="")

    # ==========================================
    # UTILITIES
    # ==========================================

    def save_chart(self):
        """Exports the current matplotlib figure to a PNG image file."""
        if self.current_figure is None:
            messagebox.showwarning("No Chart",
                                   "Please generate a historical chart or forecast first.")
            return

        file_path = filedialog.asksaveasfilename(
            defaultextension=".png",
            filetypes=[("PNG image", "*.png")],
            title="Save Chart As"
        )

        if file_path:
            self.current_figure.savefig(file_path, dpi=300,
                                        bbox_inches="tight")
            messagebox.showinfo("Success",
                                f"Chart saved successfully at:\n{file_path}")


# ==========================================
# APP ENTRY POINT
# ==========================================
if __name__ == "__main__":
    # Tell Windows to treat this script as a distinct application.
    # This forces the Windows taskbar to display our custom .ico file
    # instead of the default Python executable icon.
    try:
        # Create an arbitrary, unique application ID string
        app_id = 'analytics.goldforecast.ai.1.0'
        ctypes.windll.shell32.SetCurrentProcessExplicitAppUserModelID(app_id)
    except Exception:
        # Fails silently on non-Windows operating systems (macOS/Linux)
        pass

    root = tk.Tk()
    app = GoldForecastApp(root)
    root.mainloop()