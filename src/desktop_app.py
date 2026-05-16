import tkinter as tk
from tkinter import ttk


# =========================
# KOLORY
# =========================

BG = "#0f172a"
PANEL = "#111827"
CARD = "#1e293b"
ACCENT = "#38bdf8"
ACCENT_GREEN = "#22c55e"
TEXT = "#f8fafc"
MUTED = "#94a3b8"
BUTTON = "#2563eb"
BUTTON_HOVER = "#1d4ed8"


# =========================
# OKNO
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
    height=2
)
show_button.pack(fill="x", padx=25, pady=(0, 12))


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
    height=2
)
forecast_button.pack(fill="x", padx=25)


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