import tkinter as tk
from tkinter import ttk, messagebox, filedialog
from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt

from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import numpy as np

from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from pandas.tseries.offsets import BDay
# =========================
# COLOURS
# =========================

BG = "#0f172a"
PANEL = "#111827"
CARD = "#1e293b"
BASE_DIR = Path(__file__).resolve().parent.parent
DATA_PATH = BASE_DIR / "data" / "Gold_features.csv"
gold_data = None
chart_canvas = None
current_figure = None
FEATURE_COLUMNS = ["Close", "Volume", "RSI", "MACD", "MACD_Signal"]
TARGET_COLUMN = "Close"
TIME_STEPS = 60
ACCENT = "#38bdf8"
ACCENT_GREEN = "#22c55e"
TEXT = "#f8fafc"
MUTED = "#94a3b8"
BUTTON = "#2563eb"
BUTTON_HOVER = "#1d4ed8"


# =========================
# WINDOW
# =========================

root = tk.Tk()
root.title("Gold Forecast AI - LSTM")
root.geometry("1450x850")
root.minsize(1200, 720)
root.configure(bg=BG)


# =========================
# STYLE
# =========================

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


# =========================
# HEADER
# =========================

header = tk.Frame(root, bg=BG)
header.pack(fill="x", padx=30, pady=(25, 10))

title = tk.Label(
    header,
    text="Gold Forecast AI",
    font=("Segoe UI", 30, "bold"),
    fg=TEXT,
    bg=BG
)
title.pack(anchor="w")

subtitle = tk.Label(
    header,
    text="Predykcja cen złota z wykorzystaniem sieci LSTM i wskaźników technicznych RSI/MACD",
    font=("Segoe UI", 12),
    fg=MUTED,
    bg=BG
)
subtitle.pack(anchor="w", pady=(5, 0))


# =========================
# MAIN LAYOUT
# =========================

main_frame = tk.Frame(root, bg=BG)
main_frame.pack(fill="both", expand=True, padx=30, pady=20)


# =========================
# LEFT PANEL
# =========================

left_panel = tk.Frame(main_frame, bg=PANEL, width=340)
left_panel.pack(side="left", fill="y", padx=(0, 20))
left_panel.pack_propagate(False)

panel_title = tk.Label(
    left_panel,
    text="Panel inwestora",
    font=("Segoe UI", 18, "bold"),
    fg=TEXT,
    bg=PANEL
)
panel_title.pack(anchor="w", padx=25, pady=(25, 20))


def label(text):
    return tk.Label(
        left_panel,
        text=text,
        font=("Segoe UI", 10, "bold"),
        fg=MUTED,
        bg=PANEL
    )


label("AKTYWO").pack(anchor="w", padx=25)

asset_combo = ttk.Combobox(
    left_panel,
    values=["Gold"],
    state="readonly",
    font=("Segoe UI", 11)
)

asset_combo.current(0)
asset_combo.pack(fill="x", padx=25, pady=(5, 20))


label("HISTORIA CEN — OSTATNIE DNI").pack(anchor="w", padx=25)

history_entry = tk.Entry(
    left_panel,
    font=("Segoe UI", 12),
    bg=CARD,
    fg=TEXT,
    insertbackground=TEXT,
    relief="flat"
)
history_entry.insert(0, "50")
history_entry.pack(fill="x", padx=25, pady=(5, 20), ipady=8)


label("PROGNOZA — KOLEJNE DNI").pack(anchor="w", padx=25)

forecast_entry = tk.Entry(
    left_panel,
    font=("Segoe UI", 12),
    bg=CARD,
    fg=TEXT,
    insertbackground=TEXT,
    relief="flat"
)
forecast_entry.insert(0, "20")
forecast_entry.pack(fill="x", padx=25, pady=(5, 25), ipady=8)

def show_historical_chart():
    global gold_data
    global chart_canvas
    global current_figure

    if gold_data is None:
        load_gold_data()

    if gold_data is None:
        return

    try:
        days = int(history_entry.get())
    except ValueError:
        messagebox.showerror(
            "Błąd",
            "Liczba dni historii musi być liczbą całkowitą."
        )
        return

    if days <= 0:
        messagebox.showerror(
            "Błąd",
            "Liczba dni musi być większa od zera."
        )
        return

    filtered_data = gold_data.tail(days)

    last_price = filtered_data["Close"].iloc[-1]

    current_price_label.config(
        text=f"{last_price:.2f} USD"
    )

    status_label.config(
        text=f"Status: pokazano ostatnie {len(filtered_data)} dni"
    )

    chart_placeholder.pack_forget()

    if chart_canvas is not None:
        chart_canvas.get_tk_widget().destroy()

    fig, ax = plt.subplots(figsize=(10, 5))

    fig.patch.set_facecolor(CARD)
    ax.set_facecolor(CARD)

    ax.plot(
        filtered_data["Date"],
        filtered_data["Close"],
        linewidth=2,
        label="Cena zamknięcia"
    )

    ax.set_title(
        f"Historyczny kurs złota — ostatnie {len(filtered_data)} dni",
        color=TEXT,
        fontsize=14,
        fontweight="bold"
    )

    ax.set_xlabel("Data", color=MUTED)
    ax.set_ylabel("Cena USD", color=MUTED)

    ax.tick_params(axis="x", colors=MUTED, rotation=30)
    ax.tick_params(axis="y", colors=MUTED)

    ax.grid(True, alpha=0.3)

    legend = ax.legend()
    legend.get_frame().set_facecolor(CARD)
    legend.get_frame().set_edgecolor(CARD)

    for text in legend.get_texts():
        text.set_color(TEXT)

    fig.tight_layout()
    current_figure = fig

    chart_canvas = FigureCanvasTkAgg(fig, master=chart_panel)
    chart_canvas.draw()
    chart_canvas.get_tk_widget().pack(fill="both", expand=True, padx=25, pady=20)


def load_gold_data():
    global gold_data

    if not DATA_PATH.exists():
        messagebox.showerror(
            "Błąd",
            "Nie znaleziono pliku data/Gold_features.csv"
        )
        status_label.config(text="Status: brak pliku z danymi")
        return

    gold_data = pd.read_csv(DATA_PATH)

    gold_data["Date"] = pd.to_datetime(gold_data["Date"])
    gold_data = gold_data.sort_values("Date")
    gold_data = gold_data.dropna()

    last_price = gold_data["Close"].iloc[-1]

    current_price_label.config(
        text=f"{last_price:.2f} USD"
    )

    status_label.config(
        text=f"Status: wczytano {len(gold_data)} rekordów"
    )

    messagebox.showinfo(
        "Sukces",
        "Dane złota zostały poprawnie wczytane."
    )


show_button = tk.Button(
    left_panel,
    text="Pokaż dane historyczne",
    font=("Segoe UI", 11, "bold"),
    bg=BUTTON,
    fg="white",
    activebackground=BUTTON_HOVER,
    activeforeground="white",
    relief="flat",
    cursor="hand2",
    height=2,
    command=show_historical_chart
)
show_button.pack(fill="x", padx=25, pady=(0, 12))

def create_sequences(features, target, time_steps):
    X = []
    y = []

    for i in range(time_steps, len(features)):
        X.append(features[i - time_steps:i])
        y.append(target[i])

    return np.array(X), np.array(y)

def generate_lstm_forecast():
    global gold_data
    global chart_canvas

    if gold_data is None:
        load_gold_data()

    if gold_data is None:
        return

    try:
        forecast_days = int(forecast_entry.get())
    except ValueError:
        messagebox.showerror(
            "Błąd",
            "Liczba dni prognozy musi być liczbą całkowitą."
        )
        return

    if forecast_days <= 0:
        messagebox.showerror(
            "Błąd",
            "Liczba dni prognozy musi być większa od zera."
        )
        return

    forecast_button.config(state="disabled", text="Trwa generowanie...")
    show_button.config(state="disabled")
    status_label.config(text="Status: trwa generowanie prognozy LSTM...")
    root.update_idletasks()

    data = gold_data[FEATURE_COLUMNS].dropna()

    feature_scaler = MinMaxScaler(feature_range=(0, 1))
    target_scaler = MinMaxScaler(feature_range=(0, 1))

    scaled_features = feature_scaler.fit_transform(data[FEATURE_COLUMNS])
    scaled_target = target_scaler.fit_transform(data[[TARGET_COLUMN]])

    X, y = create_sequences(
        scaled_features,
        scaled_target,
        TIME_STEPS
    )

    train_size = int(len(X) * 0.8)

    X_train = X[:train_size]
    y_train = y[:train_size]

    X_test = X[train_size:]
    y_test = y[train_size:]

    model = Sequential()

    model.add(
        LSTM(
            units=64,
            return_sequences=True,
            input_shape=(X_train.shape[1], X_train.shape[2])
        )
    )
    model.add(Dropout(0.2))

    model.add(
        LSTM(
            units=64,
            return_sequences=False
        )
    )
    model.add(Dropout(0.2))

    model.add(Dense(32, activation="relu"))
    model.add(Dense(1))

    model.compile(
        optimizer="adam",
        loss="mean_squared_error"
    )

    model.fit(
        X_train,
        y_train,
        epochs=10,
        batch_size=32,
        validation_data=(X_test, y_test),
        verbose=0
    )

    last_sequence = scaled_features[-TIME_STEPS:].copy()
    future_predictions = []

    for _ in range(forecast_days):
        input_data = last_sequence.reshape(1, TIME_STEPS, len(FEATURE_COLUMNS))

        predicted_scaled_close = model.predict(input_data, verbose=0)[0][0]

        predicted_close = target_scaler.inverse_transform(
            [[predicted_scaled_close]]
        )[0][0]

        future_predictions.append(predicted_close)

        next_row = last_sequence[-1].copy()
        next_row[0] = predicted_scaled_close

        last_sequence = np.vstack([
            last_sequence[1:],
            next_row
        ])

    last_real_price = gold_data["Close"].iloc[-1]
    last_forecast_price = future_predictions[-1]

    current_price_label.config(
        text=f"{last_real_price:.2f} USD"
    )

    forecast_price_label.config(
        text=f"{last_forecast_price:.2f} USD"
    )

    show_forecast_chart(future_predictions)

    status_label.config(
        text=f"Status: wygenerowano prognozę na {forecast_days} dni"
    )
    forecast_button.config(state="normal", text="Generuj prognozę LSTM")
    show_button.config(state="normal")


def show_forecast_chart(future_predictions):
    global chart_canvas
    global current_figure

    try:
        history_days = int(history_entry.get())
    except ValueError:
        history_days = 50

    historical_data = gold_data.tail(history_days)

    last_date = historical_data["Date"].iloc[-1]

    future_dates = pd.bdate_range(
        start=last_date + BDay(1),
        periods=len(future_predictions)
    )

    chart_placeholder.pack_forget()

    if chart_canvas is not None:
        chart_canvas.get_tk_widget().destroy()

    fig, ax = plt.subplots(figsize=(10, 5))

    fig.patch.set_facecolor(CARD)
    ax.set_facecolor(CARD)

    ax.plot(
        historical_data["Date"],
        historical_data["Close"],
        linewidth=2,
        label="Cena historyczna"
    )

    ax.plot(
        future_dates,
        future_predictions,
        linewidth=2,
        linestyle="--",
        label="Prognoza LSTM"
    )

    ax.set_title(
        "Prognoza ceny złota przy użyciu LSTM",
        color=TEXT,
        fontsize=14,
        fontweight="bold"
    )

    ax.set_xlabel("Data", color=MUTED)
    ax.set_ylabel("Cena USD", color=MUTED)

    ax.tick_params(axis="x", colors=MUTED, rotation=30)
    ax.tick_params(axis="y", colors=MUTED)

    ax.grid(True, alpha=0.3)

    legend = ax.legend()
    legend.get_frame().set_facecolor(CARD)
    legend.get_frame().set_edgecolor(CARD)

    for text in legend.get_texts():
        text.set_color(TEXT)

    fig.tight_layout()
    current_figure = fig

    chart_canvas = FigureCanvasTkAgg(fig, master=chart_panel)
    chart_canvas.draw()
    chart_canvas.get_tk_widget().pack(fill="both", expand=True, padx=25, pady=20)

def save_chart():
    if current_figure is None:
        messagebox.showwarning(
            "Brak wykresu",
            "Najpierw wygeneruj wykres historyczny albo prognozę."
        )
        return

    file_path = filedialog.asksaveasfilename(
        defaultextension=".png",
        filetypes=[("PNG image", "*.png")],
        title="Zapisz wykres jako"
    )

    if not file_path:
        return

    current_figure.savefig(file_path, dpi=300, bbox_inches="tight")

    messagebox.showinfo(
        "Sukces",
        f"Wykres został zapisany:\n{file_path}"
    )


forecast_button = tk.Button(
    left_panel,
    text="Generuj prognozę LSTM",
    font=("Segoe UI", 11, "bold"),
    bg=ACCENT_GREEN,
    fg="#052e16",
    activebackground="#16a34a",
    activeforeground="white",
    relief="flat",
    cursor="hand2",
    height=2,
    command=generate_lstm_forecast
)
forecast_button.pack(fill="x", padx=25)

save_chart_button = tk.Button(
    left_panel,
    text="Zapisz wykres PNG",
    font=("Segoe UI", 11, "bold"),
    bg="#f59e0b",
    fg="#111827",
    activebackground="#d97706",
    activeforeground="white",
    relief="flat",
    cursor="hand2",
    height=2,
    command=save_chart
)

save_chart_button.pack(fill="x", padx=25, pady=(12, 0))


# =========================
# KARTY WYNIKÓW
# =========================

cards_frame = tk.Frame(left_panel, bg=PANEL)
cards_frame.pack(fill="x", padx=25, pady=30)


def create_card(title_text, value_text, color):
    frame = tk.Frame(cards_frame, bg=CARD)
    frame.pack(fill="x", pady=8)

    title_label = tk.Label(
        frame,
        text=title_text,
        font=("Segoe UI", 9, "bold"),
        fg=MUTED,
        bg=CARD
    )
    title_label.pack(anchor="w", padx=15, pady=(12, 0))

    value_label = tk.Label(
        frame,
        text=value_text,
        font=("Segoe UI", 18, "bold"),
        fg=color,
        bg=CARD
    )
    value_label.pack(anchor="w", padx=15, pady=(3, 12))

    return value_label


current_price_label = create_card(
    "AKTUALNA CENA",
    "-",
    ACCENT
)

forecast_price_label = create_card(
    "PROGNOZOWANA CENA",
    "-",
    ACCENT_GREEN
)

status_label = tk.Label(
    left_panel,
    text="Status: oczekiwanie na dane",
    font=("Segoe UI", 10),
    fg=MUTED,
    bg=PANEL
)
status_label.pack(anchor="w", padx=25, pady=(15, 0))


# =========================
# RIGHT PANEL
# =========================

right_panel = tk.Frame(main_frame, bg=BG)
right_panel.pack(side="right", fill="both", expand=True)


top_cards = tk.Frame(right_panel, bg=BG)
top_cards.pack(fill="x", pady=(0, 20))


def small_metric(parent, title_text, value_text):
    frame = tk.Frame(parent, bg=CARD, height=90)
    frame.pack(side="left", fill="x", expand=True, padx=8)
    frame.pack_propagate(False)

    tk.Label(
        frame,
        text=title_text,
        font=("Segoe UI", 9, "bold"),
        fg=MUTED,
        bg=CARD
    ).pack(anchor="w", padx=18, pady=(15, 0))

    tk.Label(
        frame,
        text=value_text,
        font=("Segoe UI", 18, "bold"),
        fg=TEXT,
        bg=CARD
    ).pack(anchor="w", padx=18, pady=(4, 0))


small_metric(top_cards, "MODEL", "LSTM")
small_metric(top_cards, "WSKAŹNIKI", "RSI / MACD")
small_metric(top_cards, "AKTYWO", "GOLD")
small_metric(top_cards, "TRYB", "LOCAL APP")


chart_panel = tk.Frame(right_panel, bg=CARD)
chart_panel.pack(fill="both", expand=True)

chart_title = tk.Label(
    chart_panel,
    text="Wykres kursu złota i prognozy",
    font=("Segoe UI", 18, "bold"),
    fg=TEXT,
    bg=CARD
)
chart_title.pack(anchor="w", padx=25, pady=(25, 10))

chart_placeholder = tk.Label(
    chart_panel,
    text=(
        "Tutaj zostanie wyświetlony wykres:\n\n"
        "• historyczny kurs złota\n"
        "• prognozowany kurs na kolejne dni\n"
        "• porównanie ceny rzeczywistej i predykcji LSTM\n\n"
        "W kolejnym kroku podłączymy dane CSV oraz wykres Matplotlib."
    ),
    font=("Segoe UI", 15),
    fg=MUTED,
    bg=CARD,
    justify="center"
)
chart_placeholder.pack(expand=True)


# =========================
# START
# =========================





root.mainloop()
