"""Esempio standalone con tre tabelle ancorate.

La tabella di sinistra è ancorata al bordo sinistro della finestra,
quella di destra al bordo destro e la tabella centrale si espande per
riempire lo spazio residuo. La larghezza minima della finestra è
calcolata come la somma delle larghezze minime dei titoli delle tre
Tabelle, più il padding orizzontale applicato al layout.
"""

import tkinter as tk
from tkinter import ttk
import tkinter.font as tkfont

OUTER_PAD_X = 16  # padding del contenitore principale
INNER_PAD_X = 8   # spazio orizzontale tra le tabelle


def _measure_heading(text: str) -> int:
    """Restituisce la larghezza minima per un'intestazione di colonna."""
    style = ttk.Style()
    font_name = style.lookup("Treeview.Heading", "font") or "TkDefaultFont"
    font = tkfont.nametofont(font_name)
    return font.measure(text if text else " ") + 24  # spazio per padding interno


def _spread_even(tree: ttk.Treeview, columns, minwidths, total_target: int) -> None:
    """Ripartisce lo spazio extra tra le colonne della Treeview."""
    if not columns:
        width = max(minwidths[0], total_target)
        tree.column("—", width=width, minwidth=minwidths[0], stretch=False, anchor="center")
        return

    base_sum = sum(minwidths)
    extra = max(0, total_target - base_sum)
    per_col = extra // len(columns)
    remainder = extra % len(columns)
    for index, (col, minimum) in enumerate(zip(columns, minwidths)):
        width = minimum + per_col + (1 if index < remainder else 0)
        tree.column(col, width=width, minwidth=minimum, stretch=False, anchor="center")


def _build_table(parent, title: str, columns, rows, mode: str):
    frame = tk.LabelFrame(parent, text=title)
    frame.grid_rowconfigure(0, weight=1)
    frame.grid_columnconfigure(0, weight=(1 if mode == "center" else 0))

    tree = ttk.Treeview(frame, columns=columns or ("—",), show="headings", height=10)
    tree.grid(row=0, column=0, sticky=("nsew" if mode == "center" else "ns"))

    # scrollbar verticale sempre visibile
    scroll = ttk.Scrollbar(frame, orient="vertical", command=tree.yview)
    scroll.grid(row=0, column=1, sticky="ns")
    tree.configure(yscrollcommand=scroll.set)

    minwidths = []
    configured_columns = columns or ["—"]

    for col in configured_columns:
        heading = col if columns else "—"
        tree.heading(col, text=heading)
        minimum = _measure_heading(heading)
        tree.column(col, width=minimum, minwidth=minimum, stretch=(mode == "center"))
        minwidths.append(minimum)

    for row in rows:
        tree.insert("", "end", values=row)

    return frame, tree, configured_columns, minwidths


def main():
    root = tk.Tk()
    root.title("Tre Tabelle Ancorate")

    container = tk.Frame(root)
    container.pack(fill="both", expand=True, padx=OUTER_PAD_X, pady=OUTER_PAD_X)

    container.grid_columnconfigure(0, weight=0)  # colonna sinistra fissa
    container.grid_columnconfigure(1, weight=1)  # colonna centrale elastica
    container.grid_columnconfigure(2, weight=0)  # colonna destra fissa
    container.grid_rowconfigure(0, weight=1)

    recorded_cols = ("Q [m3/h]", "Head [m]", "P2 [bar]")
    recorded_rows = [
        ("120", "34.1", "8.9"),
        ("140", "33.0", "9.1"),
        ("160", "31.8", "9.6"),
    ]

    calc_cols = ("Eff %", "Pwr [kW]", "NPSH [m]", "Slip %")
    calc_rows = [
        ("76.4", "9.1", "3.2", "4.1"),
        ("78.9", "9.6", "3.4", "3.8"),
        ("80.1", "10.1", "3.6", "3.5"),
    ]

    conv_cols = ("Q [gpm]", "Head [ft]", "P2 [psi]")
    conv_rows = [
        ("529", "111.8", "129.2"),
        ("617", "108.3", "132.0"),
        ("705", "104.3", "139.3"),
    ]

    left_frame, left_tree, left_cols, left_mins = _build_table(
        container, "Recorded", recorded_cols, recorded_rows, "left"
    )
    middle_frame, middle_tree, middle_cols, middle_mins = _build_table(
        container, "Calculated", calc_cols, calc_rows, "center"
    )
    right_frame, right_tree, right_cols, right_mins = _build_table(
        container, "Converted", conv_cols, conv_rows, "right"
    )

    left_frame.grid(row=0, column=0, sticky="nsw", padx=(0, INNER_PAD_X))
    middle_frame.grid(row=0, column=1, sticky="nsew", padx=(INNER_PAD_X, INNER_PAD_X))
    right_frame.grid(row=0, column=2, sticky="nse", padx=(INNER_PAD_X, 0))

    left_base = sum(left_mins)
    middle_base = sum(middle_mins)
    right_base = sum(right_mins)

    # assicurati che i frame laterali abbiano le rispettive larghezze minime
    container.grid_columnconfigure(0, minsize=left_base)
    container.grid_columnconfigure(2, minsize=right_base)

    def _resize_middle(_event=None):
        middle_frame.update_idletasks()
        target = max(middle_base, middle_frame.winfo_width() - 2 * INNER_PAD_X)
        _spread_even(middle_tree, middle_cols, middle_mins, target)

    middle_frame.bind("<Configure>", _resize_middle)
    root.after_idle(_resize_middle)

    min_window_width = left_base + middle_base + right_base + 2 * OUTER_PAD_X + INNER_PAD_X * 4
    root.update_idletasks()
    root.minsize(min_window_width, root.winfo_reqheight())

    root.mainloop()


if __name__ == "__main__":
    main()
