# certificate_view.py
import os
import sys
import re
import math
from collections import defaultdict
import tkinter as tk
from tkinter import ttk, messagebox, filedialog
import tkinter.font as tkfont  # <— per misurare i titoli

# nptdms
try:
    from nptdms import TdmsFile
    NPTDMS_OK = True
except Exception:
    TdmsFile = None
    NPTDMS_OK = False

# numpy (media su tutti i campioni)
try:
    import numpy as np
    NUMPY_OK = True
except Exception:
    NUMPY_OK = False

# Export PDF (opzionale)
try:
    from reportlab.lib.pagesizes import A4
    from reportlab.pdfgen import canvas as rl_canvas
    REPORTLAB_OK = True
except Exception:
    REPORTLAB_OK = False


# -------------------- UI helper --------------------
def _kv_row(parent, label, value="—"):
    row = tk.Frame(parent, bg="#ffffff")
    row.pack(fill="x", padx=8, pady=2)
    tk.Label(row, text=label, width=22, anchor="w", bg="#ffffff",
             font=("Segoe UI", 10, "bold")).pack(side="left")
    tk.Label(row, text=value if value else "—", anchor="w", bg="#ffffff",
             font=("Segoe UI", 10)).pack(side="left", fill="x", expand=True)
    return row


# -------------------- TDMS utils (scalar per Contract/Loop) --------------------
def _first_nonempty(seq):
    if not (hasattr(seq, "__iter__") and not isinstance(seq, (str, bytes, bytearray))):
        seq = [seq]
    for x in seq:
        if x is None:
            continue
        if isinstance(x, (bytes, bytearray)):
            try:
                x = x.decode("utf-8", errors="ignore")
            except Exception:
                x = str(x)
        s = str(x).strip()
        if s:
            return s
    return ""

def _get_group_ci(tdms, group_name: str):
    tgt = (group_name or "").lower()
    for g in tdms.groups():
        if (g.name or "").lower() == tgt:
            return g
    return None

def _get_channel_ci(group, channel_name: str):
    tgt = (channel_name or "").lower()
    for ch in group.channels():
        if (ch.name or "").lower() == tgt:
            return ch
    return None

def tdms_read_scalar_string(tdms, group_name: str, channel_name: str) -> str:
    try:
        grp = _get_group_ci(tdms, group_name)
        if not grp:
            return "—"
        ch = _get_channel_ci(grp, channel_name)
        if not ch:
            return "—"
        try:
            data = ch[:]
        except Exception:
            data = getattr(ch, "data", [])
        val = _first_nonempty(data)
        return val if val else "—"
    except Exception:
        return "—"


# -------------------- PERFORMANCE loader (dinamico + media a chunk) --------------------
GROUP_RE = re.compile(
    r"^(?P<test>\d+)_(?P<point>\d+)_"
    r"(?P<prefix>PERFORMANCE_PERFORM)"
    r"_(?:Test_)?(?P<kind>Recorded|Calc|Converted)$"
)

KIND_ORDER = ("Recorded", "Calc", "Converted")

KNOWN_UNITS = {"rpm", "m3/h", "m", "kw", "%", "bar", "pa", "°c"}
NOISE_TOKENS = {"-", "–", "—"}
INCLUDE_ALWAYS = {"rpm"}
QUAL_MAP = {
    "suct": "Suction", "suct.": "Suction", "suction": "Suction",
    "disch": "Discharge", "disch.": "Discharge", "discharge": "Discharge",
    "in": "In", "out": "Out",
}

def _tokens(raw: str):
    return [t for t in (raw or "").split() if t]

def _is_unit(tok: str) -> bool:
    return tok.lower().strip("[]()") in KNOWN_UNITS

def _is_noise(tok: str) -> bool:
    if tok in NOISE_TOKENS:
        return True
    return bool(re.fullmatch(r"[\W_]+", tok))

def _primary_token(raw: str):
    for t in _tokens(raw):
        if t.lower().strip("[]()") in INCLUDE_ALWAYS:
            return t
    for t in _tokens(raw):
        if _is_unit(t) or _is_noise(t):
            continue
        return t
    return None

def _find_qualifier(raw: str):
    toks = _tokens(raw)
    if not toks:
        return None
    base = _primary_token(raw)
    started = False
    for t in toks:
        if not started:
            if t == base:
                started = True
            continue
        if _is_unit(t) or _is_noise(t):
            continue
        low = t.lower().strip().strip("[]()")
        if low in QUAL_MAP:
            return QUAL_MAP[low]
        if re.search(r"[A-Za-z0-9]", t):
            return t.replace("/", "_")
    return None

def _find_unit(raw: str):
    for t in _tokens(raw):
        low = t.lower().strip("[]()")
        if low in KNOWN_UNITS:
            return low
    return None

def _mean_all_strict(data):
    if NUMPY_OK:
        try:
            a = np.asarray(data, dtype=float)
        except Exception:
            return 0.0
        if a.size == 0:
            return 0.0
        a = a.copy()
        mask = ~np.isfinite(a)
        if mask.any():
            a[mask] = 0.0
        return float(np.mean(a))
    seq = data if hasattr(data, "__iter__") and not isinstance(data, (str, bytes, bytearray)) else [data]
    total = 0.0
    n = 0
    for v in seq:
        try:
            f = float(v)
        except Exception:
            f = 0.0
        if math.isfinite(f):
            total += f
        n += 1
    return (total / n) if n else 0.0

def _fmt_num(x):
    try:
        return f"{float(x):g}"
    except Exception:
        return str(x)

def _collect_perf_points(tdms, test_index: int = 0):
    points = defaultdict(lambda: {"Recorded": [], "Calc": [], "Converted": []})
    for g in tdms.groups():
        m = GROUP_RE.match(g.name or "")
        if not m:
            continue
        if int(m.group("test")) != int(test_index):
            continue
        p = int(m.group("point"))
        k = m.group("kind")
        points[p][k].append(g)
    return dict(points)

def _nan_sum_and_count(a):
    if NUMPY_OK:
        try:
            finite = np.isfinite(a)
        except Exception:
            try:
                a = np.asarray(list(a), dtype=float)
                finite = np.isfinite(a)
            except Exception:
                return 0.0, 0
        if not np.any(finite):
            return 0.0, 0
        s = float(np.sum(a[finite]))
        c = int(np.count_nonzero(finite))
        return s, c
    s = 0.0; c = 0
    for v in a:
        try:
            f = float(v)
        except Exception:
            continue
        if math.isfinite(f):
            s += f; c += 1
    return s, c

def _mean_channel_fast(ch, chunk_size=2_000_000, use_float32=True):
    try:
        n = len(ch)
    except Exception:
        try:
            data = ch[:]
        except Exception:
            data = getattr(ch, "data", [])
        return _mean_all_strict(data)

    total = 0.0
    count = 0
    to_dtype = np.float32 if (NUMPY_OK and use_float32) else float

    start = 0
    while start < n:
        stop = min(start + chunk_size, n)
        try:
            part = ch[start:stop]
        except Exception:
            try:
                part = ch[:]
            except Exception:
                part = getattr(ch, "data", [])
            start = n
        else:
            start = stop

        if NUMPY_OK:
            try:
                arr = np.asarray(part, dtype=to_dtype, order="C")
            except Exception:
                arr = np.array([], dtype=to_dtype)
        else:
            arr = list(part)

        s, c = _nan_sum_and_count(arr)
        total += s
        count += c

    if count == 0:
        return 0.0
    return total / count

# ===== Builder: colonne SENZA unità e SENZA nome categoria (vale per ogni tabella) =====
def _build_kind_model_no_unit(groups_by_point: dict):
    first_seen_order = []
    preferred_unit = {}
    max_dups = defaultdict(int)

    def base_key_no_unit(ch_name: str):
        base = _primary_token(ch_name)
        if base is None:
            return None, None
        qual = _find_qualifier(ch_name)
        unit = _find_unit(ch_name) or ""
        key = f"{base}_{qual}" if qual else f"{base}"
        return key, unit

    for p in sorted(groups_by_point.keys()):
        counts_this_point = defaultdict(int)
        for grp in groups_by_point[p]:
            for ch in grp.channels():
                key, unit = base_key_no_unit(ch.name)
                if key is None:
                    continue
                counts_this_point[key] += 1
                if key not in preferred_unit:
                    preferred_unit[key] = unit
                if key not in first_seen_order:
                    first_seen_order.append(key)
        for key, cnt in counts_this_point.items():
            if cnt > max_dups[key]:
                max_dups[key] = cnt

    columns, units = [], []
    for key in first_seen_order:
        n = max(1, max_dups.get(key, 1))
        for i in range(1, n + 1):
            col = key if i == 1 else f"{key}__{i}"
            columns.append(col)
            units.append(preferred_unit.get(key, ""))

    rows = []
    for p in sorted(groups_by_point.keys()):
        seq = defaultdict(int)
        row_map = {}
        for grp in groups_by_point[p]:
            for ch in grp.channels():
                key, _u = base_key_no_unit(ch.name)
                if key is None:
                    continue
                seq[key] += 1
                col = key if seq[key] == 1 else f"{key}__{seq[key]}"
                try:
                    mean_val = _mean_channel_fast(ch)
                except Exception:
                    try:
                        data = ch[:]
                    except Exception:
                        data = getattr(ch, "data", [])
                    mean_val = _mean_all_strict(data)
                row_map[col] = _fmt_num(mean_val)
        rows.append(tuple(row_map.get(c, "") for c in columns))

    return columns, units, rows


def read_performance_tables_dynamic(tdms_path: str, test_index: int = 0):
    out = {k: {"columns": [], "units": [], "rows": []} for k in KIND_ORDER}
    if not (tdms_path and os.path.exists(tdms_path) and NPTDMS_OK):
        return out
    try:
        with TdmsFile.open(tdms_path) as tdms:
            points = _collect_perf_points(tdms, test_index=test_index)
            if not points:
                return out
            by_kind = {k: defaultdict(list) for k in KIND_ORDER}
            for p, kinds in points.items():
                for k in KIND_ORDER:
                    if kinds[k]:
                        by_kind[k][p].extend(kinds[k])

            for k in KIND_ORDER:
                if by_kind[k]:
                    cols, units, rows = _build_kind_model_no_unit(by_kind[k])
                    out[k]["columns"] = cols
                    out[k]["units"]   = units
                    out[k]["rows"]    = rows
            return out
    except Exception:
        return out


# -------------------- Export PDF placeholder --------------------
def _export_pdf_placeholder(pdf_path: str, header_text: str, meta_lines: list[str]):
    if not REPORTLAB_OK:
        raise RuntimeError("ReportLab non installato")
    c = rl_canvas.Canvas(pdf_path, pagesize=A4)
    w, h = A4
    c.setFont("Helvetica-Bold", 18); c.drawString(40, h - 60, header_text)
    y = h - 100; c.setFont("Helvetica", 10)
    for line in meta_lines:
        c.drawString(40, y, line); y -= 16
    y -= 12; c.setFont("Helvetica-Bold", 12); c.drawString(40, y, "Contractual Data")
    y -= 10; c.setFont("Helvetica", 10); c.drawString(40, y, "(contenuti in arrivo)")
    y -= 24; c.setFont("Helvetica-Bold", 12); c.drawString(40, y, "Loop Details — Test performed with :")
    y -= 10; c.setFont("Helvetica", 10); c.drawString(40, y, "(contenuti in arrivo)")
    y -= 24; c.setFont("Helvetica-Bold", 12); c.drawString(40, y, "Recorded / Calculated / Converted")
    y -= 10; c.setFont("Helvetica", 10); c.drawString(40, y, "(tabelle in arrivo)")
    c.showPage(); c.save()


# -------------------- Apertura file con OS --------------------
def _open_file_with_os(path: str):
    if not path or not os.path.exists(path):
        messagebox.showwarning("File", "File non trovato.")
        return
    if sys.platform.startswith("win"):
        os.startfile(path)  # type: ignore[attr-defined]
    elif sys.platform == "darwin":
        import subprocess
        subprocess.run(["open", path], check=False)
    else:
        import subprocess
        subprocess.run(["xdg-open", path], check=False)


# -------------------- util UI: misura titolo, ridistribuzione colonne  ----------------
def _measure_title(text: str) -> int:
    try:
        style = ttk.Style()  # più robusto
        font_name = style.lookup("Treeview.Heading", "font") or "TkDefaultFont"
    except Exception:
        font_name = "TkDefaultFont"
    fnt = tkfont.nametofont(font_name)
    return fnt.measure(text if text else " ") + 40

def _spread_even_in_tv(tv, cols, minwidths, total_target, *, stretch=False):
    """Distribuisce uniformemente lo spazio in più tra le colonne della treeview."""
    if not cols:
        mw = max(minwidths[0], total_target)
        tv.column("—", width=mw, minwidth=minwidths[0], stretch=stretch, anchor="center")
        return
    base_sum = sum(minwidths)
    extra = max(0, total_target - base_sum)
    n = len(cols)
    add_each = extra // n
    rem = extra % n
    for i, (c, wmin) in enumerate(zip(cols, minwidths)):
        w = wmin + add_each + (1 if i < rem else 0)
        tv.column(c, width=w, minwidth=wmin, stretch=stretch, anchor="center")


# -------------------- Finestra di dettaglio --------------------
def open_detail_window(root, columns, values, meta):
    win = tk.Toplevel(root)
    win.title("Test Certificate")
    win.minsize(400, 800)
    win.configure(bg="#f0f0f0")

    cert_num   = values[1] if len(values) > 1 else "—"
    test_date  = values[4] if len(values) > 4 else "—"
    job_dash   = values[0] if len(values) > 0 else "—"
    pump_dash  = values[3] if len(values) > 3 else "—"

    state = {"tdms_path": None}

    # -------------------- UI layout --------------------
    nb = ttk.Notebook(win); nb.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
    tab = tk.Frame(nb, bg="#ffffff"); nb.add(tab, text="Certificato")

    tab.columnconfigure(0, weight=1)
    tab.rowconfigure(1, weight=1)

    # Header
    header = tk.Frame(tab, bg="#ffffff")
    header.grid(row=0, column=0, sticky="ew", padx=16, pady=(16,8))
    header.columnconfigure(0, weight=1)
    header.columnconfigure(1, weight=1)
    tk.Label(header, text="TEST CERTIFICATE", bg="#ffffff", font=("Segoe UI", 20, "bold")).grid(row=0, column=0, sticky="w")
    right = tk.Frame(header, bg="#ffffff"); right.grid(row=0, column=1, sticky="e")
    tk.Button(right, text="Carica TDMS…", command=do_load_tdms, width=14).pack(side="right", padx=(8,0))
    lbl_cert = tk.Label(right, text=f"N° Cert.: {cert_num}", bg="#ffffff", font=("Segoe UI", 10)); lbl_cert.pack(anchor="e")
    tk.Label(right, text="U.M. System: SI (Metric)", bg="#ffffff", font=("Segoe UI", 10)).pack(anchor="e")
    lbl_date = tk.Label(right, text=f"Test Date: {test_date}", bg="#ffffff", font=("Segoe UI", 10)); lbl_date.pack(anchor="e")

    # Body scroll unico (verticale)
    body_wrap = tk.Frame(tab, bg="#ffffff"); body_wrap.grid(row=1, column=0, sticky="nsew")
    body_wrap.columnconfigure(0, weight=1); body_wrap.rowconfigure(0, weight=1)
    canvas = tk.Canvas(body_wrap, highlightthickness=0, bg="#ffffff")
    vscroll = ttk.Scrollbar(body_wrap, orient="vertical", command=canvas.yview)
    canvas.configure(yscrollcommand=vscroll.set)
    canvas.grid(row=0, column=0, sticky="nsew"); vscroll.grid(row=0, column=1, sticky="ns")
    body = tk.Frame(canvas, bg="#ffffff"); body_id = canvas.create_window((0,0), window=body, anchor="nw")

    def _on_cfg(_e=None):
        canvas.configure(scrollregion=canvas.bbox("all"))
        try: canvas.itemconfigure(body_id, width=canvas.winfo_width())
        except Exception: pass
    canvas.bind("<Configure>", _on_cfg)
    body.bind("<Configure>", lambda _e: canvas.configure(scrollregion=canvas.bbox("all")))

    # Containers
    blocks = tk.Frame(body, bg="#ffffff")
    blocks.pack(fill="x", expand=True, padx=16, pady=(8,8))
    blocks.columnconfigure(0, weight=1)
    blocks.columnconfigure(1, weight=1)

    tables_row = tk.Frame(body, bg="#ffffff")
    tables_row.pack(fill="both", expand=True, padx=16, pady=(8,16))

    # 3 colonne di griglia: 0 sinistra (ancorata), 1 centro (elastica), 2 destra (ancorata)
    tables_row.grid_columnconfigure(0, weight=0)  # sinistra fissa
    tables_row.grid_columnconfigure(1, weight=1)  # centro prende tutto lo spazio residuo
    tables_row.grid_columnconfigure(2, weight=0)  # destra fissa
    tables_row.grid_rowconfigure(0, weight=1)

    # --- renderer: Contractual + Loop -------------------------------------------------
    def render_contract_and_loop(tdms_path: str):
        for w in blocks.winfo_children():
            w.destroy()

        cap = tdh = eff = abs_pow = speed = sg = temp = visc = npsh = liquid = "—"
        cust = po = end_user = specs = "—"
        item = pump = sn = imp_draw = imp_mat = imp_dia = "—"
        suction = discharge = watt_const = atmpress = knpsh = watertemp = kventuri = "—"

        if NPTDMS_OK and tdms_path and os.path.exists(tdms_path):
            try:
                with TdmsFile.open(tdms_path) as tdms:
                    cap     = tdms_read_scalar_string(tdms, "Ref. Contract Data", "Capacity [m3/h]")
                    tdh     = tdms_read_scalar_string(tdms, "Ref. Contract Data", "TDH [m]")
                    eff     = tdms_read_scalar_string(tdms, "Ref. Contract Data", "Efficiency [%]")
                    abs_pow = tdms_read_scalar_string(tdms, "Ref. Contract Data", "ABS_Power [kW]")
                    speed   = tdms_read_scalar_string(tdms, "Ref. Contract Data", "Speed [rpm]")
                    sg      = tdms_read_scalar_string(tdms, "Ref. Contract Data", "SG Contract")
                    temp    = tdms_read_scalar_string(tdms, "Ref. Contract Data", "Temperature [C]")
                    visc    = tdms_read_scalar_string(tdms, "Ref. Contract Data", "Viscosity [cP]")
                    npsh    = tdms_read_scalar_string(tdms, "Ref. Contract Data", "NPSH [m]")
                    liquid  = tdms_read_scalar_string(tdms, "Ref. Contract Data", "Liquid")

                    cust    = tdms_read_scalar_string(tdms, "Ref. Test Param.", "Customer")
                    po      = tdms_read_scalar_string(tdms, "Ref. Test Param.", "Purchaser Order")
                    end_user= tdms_read_scalar_string(tdms, "Ref. Test Param.", "End User")
                    specs   = tdms_read_scalar_string(tdms, "Ref. Test Param.", "Applic. Specs.")

                    item    = tdms_read_scalar_string(tdms, "Ref. Pump Type", "Item")
                    pump    = tdms_read_scalar_string(tdms, "Ref. Pump Type", "Pump")
                    sn      = tdms_read_scalar_string(tdms, "Ref. Pump Type", "Serial Number_Elenco")
                    imp_draw= tdms_read_scalar_string(tdms, "Ref. Pump Type", "Impeller Drawing")
                    imp_mat = tdms_read_scalar_string(tdms, "Ref. Pump Type", "Impeller Material")
                    imp_dia = tdms_read_scalar_string(tdms, "Ref. Pump Type", "Diam Nominal")

                    suction     = tdms_read_scalar_string(tdms, "Ref. Test Detail", "Suction [Inch]")
                    discharge   = tdms_read_scalar_string(tdms, "Ref. Test Detail", "Discharge [Inch]")
                    watt_const  = tdms_read_scalar_string(tdms, "Ref. Test Detail", "Wattmeter Const.")
                    atmpress    = tdms_read_scalar_string(tdms, "Ref. Test Detail", "AtmPress [m]")
                    knpsh       = tdms_read_scalar_string(tdms, "Ref. Test Detail", "KNPSH [m]")
                    watertemp   = tdms_read_scalar_string(tdms, "Ref. Test Detail", "WaterTemp [C]")
                    kventuri    = tdms_read_scalar_string(tdms, "Ref. Test Detail", "KVenturi")

            except Exception:
                pass

        contractual = tk.LabelFrame(blocks, text="Contractual Data", bg="#ffffff")
        contractual.grid(row=0, column=0, sticky="nsew", padx=(0,8))
        contractual.columnconfigure(0, weight=1); contractual.columnconfigure(1, weight=1)

        col1 = tk.Frame(contractual, bg="#ffffff"); col1.grid(row=0, column=0, sticky="nsew", padx=(0,8))
        _kv_row(col1, "Capacity [m³/h]", cap); _kv_row(col1, "TDH [m]", tdh); _kv_row(col1, "Efficiency", eff)
        _kv_row(col1, "ABS_Power [kW]", abs_pow); _kv_row(col1, "Speed [rpm]", speed); _kv_row(col1, "SG", sg)
        _kv_row(col1, "Temperature [°C]", "—" if temp == "—" else temp)
        _kv_row(col1, "Viscosity [cP]", visc); _kv_row(col1, "NPSH [m]", npsh); _kv_row(col1, "Liquid", liquid)

        col2 = tk.Frame(contractual, bg="#ffffff"); col2.grid(row=0, column=1, sticky="nsew", padx=(8,0))
        _kv_row(col2, "FSG ORDER", job_dash if job_dash and job_dash != "—" else "—")
        _kv_row(col2, "CUSTOMER", cust); _kv_row(col2, "P.O.", po)
        _kv_row(col2, "End User", end_user); _kv_row(col2, "Item", item)
        pump_model = pump if pump and pump != "—" else (values[3] if len(values) > 3 else "—")
        _kv_row(col2, "Pump", pump_model)
        _kv_row(col2, "S. N.", sn); _kv_row(col2, "Imp. Draw.", imp_draw); _kv_row(col2, "Imp. Mat.", imp_mat)
        _kv_row(col2, "Imp Dia [mm]", imp_dia); _kv_row(col2, "Specs", specs)

        loop = tk.LabelFrame(blocks, text="Loop Details", bg="#ffffff")
        loop.grid(row=0, column=1, sticky="nsew", padx=(8,0))
        tk.Label(loop, text="Test performed with :", bg="#ffffff", font=("Segoe UI", 10, "bold")).pack(anchor="w", padx=8, pady=(6,2))
        for line in [
            f"Suction [Inch] {suction}",
            f"Discharge [Inch] {discharge}",
            f"Wattmeter Const. {watt_const}",
            f"AtmPress [m] {atmpress}",
            f"KNPSH [m] {knpsh}",
            f"WaterTemp [°C] {watertemp}",
            f"Kventuri {kventuri}",
        ]:
            tk.Label(loop, text=line, bg="#ffffff", font=("Segoe UI", 10)).pack(anchor="w", padx=8, pady=2)

    # --- renderer: Tre tabelle con ancoraggi sinistra/centro/destra ------------------
    def render_tables(tdms_path: str):
        for w in tables_row.winfo_children():
            w.destroy()

        perf = read_performance_tables_dynamic(tdms_path, test_index=0) if tdms_path else {
            "Recorded": {"columns": [], "units": [], "rows": []},
            "Calc": {"columns": [], "units": [], "rows": []},
            "Converted": {"columns": [], "units": [], "rows": []},
        }
        rec_cols, rec_units, rec_rows    = perf["Recorded"]["columns"],   perf["Recorded"]["units"],   perf["Recorded"]["rows"]
        calc_cols, calc_units, calc_rows = perf["Calc"]["columns"],       perf["Calc"]["units"],       perf["Calc"]["rows"]
        conv_cols, conv_units, conv_rows = perf["Converted"]["columns"],  perf["Converted"]["units"],  perf["Converted"]["rows"]

        def _make_table(parent, title, cols, units, mode):
            lf = tk.LabelFrame(parent, text=title, bg="#ffffff")
            lf.rowconfigure(0, weight=1)
            lf.columnconfigure(0, weight=(1 if mode == "center" else 0))

            tv = ttk.Treeview(lf, columns=cols or ("—",), show="headings", height=12, selectmode="browse")

            minwidths = []
            if not cols:
                tv.heading("—", text="—")
                mw = _measure_title("—")
                tv.column("—", minwidth=mw, width=mw, anchor="center", stretch=(mode=="center"))
                minwidths = [mw]
            else:
                for c in cols:
                    tv.heading(c, text=c)
                    mw = _measure_title(c)
                    tv.column(c, minwidth=mw, width=mw, anchor="center", stretch=(mode=="center"))
                    minwidths.append(mw)

            tv.grid(row=0, column=0, sticky=("nsew" if mode=="center" else "ns"))
            tv.tag_configure("units_row", background="#EFEFEF")
            if cols:
                tv.insert("", "end", iid="units", values=tuple(u or "" for u in units), tags=("units_row",))
            return lf, tv, (cols if cols else ["—"]), minwidths

        # crea le tre tabelle
        lf_left,  tv_left,  left_cols,  left_mins  = _make_table(tables_row, "Recorded Data",    rec_cols,  rec_units,  "left")
        lf_mid,   tv_mid,   mid_cols,   mid_mins   = _make_table(tables_row, "Calculated Values", calc_cols, calc_units, "center")
        lf_right, tv_right, right_cols, right_mins = _make_table(tables_row, "Converted Values",  conv_cols, conv_units, "right")

        # posizionamento: sinistra ancorata a sx, destra a dx, centro riempie
        lf_left.grid (row=0, column=0, sticky="w",   padx=(0,8))
        lf_mid.grid  (row=0, column=1, sticky="nsew", padx=8)
        lf_right.grid(row=0, column=2, sticky="e",   padx=(8,0))

        # --- clamp: impedisce a left/right di uscire dal proprio frame ---
        def _clamp_tv_to_frame(tv, cols, mins, frame, pad=16):
            """Riduce le colonne (senza scendere sotto i min) per stare dentro il frame."""
            try:
                frame.update_idletasks()
            except Exception:
                pass

            if not cols:
                # colonna placeholder "—": limita al frame
                avail = max(mins[0], frame.winfo_width() - pad)
                cur   = tv.column("—", option="width")
                tv.column("—", width=min(cur, avail))
                return

            avail = max(sum(mins), frame.winfo_width() - pad)
            widths = [tv.column(c, "width") for c in cols]
            total  = sum(widths)
            if total <= avail:
                return  # già dentro

            extra = total - avail  # quanto dobbiamo togliere
            while extra > 0 and any(w > m for w, m in zip(widths, mins)):
                idxs = [i for i, (w, m) in enumerate(zip(widths, mins)) if w > m]
                if not idxs:
                    break
                dec = max(1, extra // len(idxs))
                for i in idxs:
                    if extra <= 0:
                        break
                    room = widths[i] - mins[i]
                    d = min(dec, room)
                    widths[i] -= d
                    extra -= d

            for c, w in zip(cols, widths):
                tv.column(c, width=w)

        # clamp su resize del frame sinistro/destro
        lf_left.bind("<Configure>",  lambda e: _clamp_tv_to_frame(tv_left,  left_cols,  left_mins,  lf_left))
        lf_right.bind("<Configure>", lambda e: _clamp_tv_to_frame(tv_right, right_cols, right_mins, lf_right))

        # clamp dopo drag dei separatori di colonna (rilascio)
        def _maybe_clamp_after_drag(tv, cols, mins, frame):
            tv.update_idletasks()
            _clamp_tv_to_frame(tv, cols, mins, frame)

        tv_left.bind("<ButtonRelease-1>",  lambda e: _maybe_clamp_after_drag(tv_left,  left_cols,  left_mins,  lf_left))
        tv_right.bind("<ButtonRelease-1>", lambda e: _maybe_clamp_after_drag(tv_right, right_cols, right_mins, lf_right))
        # opzionale: anche durante il trascinamento
        # tv_left.bind("<B1-Motion>",  lambda e: _maybe_clamp_after_drag(tv_left,  left_cols,  left_mins,  lf_left))
        # tv_right.bind("<B1-Motion>", lambda e: _maybe_clamp_after_drag(tv_right, right_cols, right_mins, lf_right))

        # min larghezze da titoli (serve per i vincoli di layout)
        left_base  = sum(left_mins) if left_mins  else 0
        mid_base   = sum(mid_mins)   if mid_mins   else 0
        right_base = sum(right_mins) if right_mins else 0

        # colonna centrale elastica, laterali ancorate
        tables_row.grid_columnconfigure(0, weight=0, minsize=left_base)
        tables_row.grid_columnconfigure(1, weight=1, minsize=mid_base)
        tables_row.grid_columnconfigure(2, weight=0, minsize=right_base)
        
        # inserisci righe dati (p001, p002, ...)
        for idx, vals in enumerate(rec_rows, start=1):
            tv_left.insert("", "end", iid=f"p{idx:03d}", values=vals)
        for idx, vals in enumerate(calc_rows, start=1):
            tv_mid.insert("", "end", iid=f"p{idx:03d}", values=vals)
        for idx, vals in enumerate(conv_rows, start=1):
            tv_right.insert("", "end", iid=f"p{idx:03d}", values=vals)

        # ridistribuzione equa SOLO nella tabella centrale in base alla larghezza del suo frame
        def _resize_center(_e=None):
            try:
                lf_mid.update_idletasks()
            except Exception:
                return
            target = max(mid_base, lf_mid.winfo_width() - 16)  # 16 ~ padding interno
            _spread_even_in_tv(tv_mid, mid_cols, mid_mins, target, stretch=True)
        lf_mid.bind("<Configure>", _resize_center)
        win.after_idle(_resize_center)

        # --- minsize robusta: usa la larghezza richiesta del blocco tabelle ---
        def _apply_minsize_once():
            # 1) Assicura che _resize_center abbia finito
            win.update_idletasks()

            # 2) Larghezza richiesta del solo "riga tabelle" (tre LabelFrame + grid padx)
            tables_req = tables_row.winfo_reqwidth()

            # 3) Considera anche gli altri blocchi orizzontali sopra (header/contract/loop)
            header_req  = header.winfo_reqwidth()
            blocks_req  = blocks.winfo_reqwidth()
            content_req = max(tables_req, blocks_req, header_req)

            # 4) Padding esterni (pack) che NON sono inclusi in reqwidth
            def _pack_padx_total(widget):
                try:
                    px = widget.pack_info().get("padx", 0)
                except Exception:
                    return 0
                if isinstance(px, (tuple, list)):
                    return sum(int(p) for p in px)
                try:
                    return int(px) * 2  # singolo => sx+dx
                except Exception:
                    return 0

            outer_tables = _pack_padx_total(tables_row)  # nel tuo codice è 16 per lato => 32
            outer_nb     = _pack_padx_total(nb)          # Notebook: nel tuo codice è 10 per lato => 20

            # 5) Scrollbar verticale del canvas “ruba” spazio orizzontale
            try:
                vscroll_w = max(vscroll.winfo_reqwidth(), 15)
            except Exception:
                vscroll_w = 15

            safety = 12  # margine anti-sottostima (bordi tema ttk, arrotondamenti)
            min_window_width = content_req + outer_tables + outer_nb + vscroll_w + safety

            EXTRA_HEIGHT = 100   # 👈 decidi tu di quanto vuoi aumentare l’altezza
            BASE_HEIGHT  = 740   # valore di partenza usato nel codice originale

            min_window_height = BASE_HEIGHT + EXTRA_HEIGHT

            win.minsize(min_window_width, min_window_height)
            if win.winfo_width() < min_window_width or win.winfo_height() < min_window_height:
                win.geometry(f"{min_window_width}x{min_window_height}")


        # Esegui due volte a idle per essere sicuri che tutti i layout siano stabili
        win.after_idle(_apply_minsize_once)
        win.after_idle(lambda: win.after_idle(_apply_minsize_once))

        # sync selezione (ignora 'units')
        sync_state = {"syncing": False, "current_iid": None}
        def _apply_sync(iid, origin):
            if not iid or iid == "units": return
            for tv in (tv_left, tv_mid, tv_right):
                if tv is origin: continue
                try:
                    if not tv.exists(iid): continue
                    cur = tv.selection()
                    if cur and cur[0] == iid: continue
                    tv.selection_set(iid); tv.focus(iid); tv.see(iid)
                except Exception: pass

        def _on_select(src_tv, _e=None):
            if sync_state["syncing"]: return
            try:
                sel = src_tv.selection()
                iid = sel[0] if sel else None
            except Exception:
                iid = None
            if not iid or iid == "units" or sync_state["current_iid"] == iid: return
            sync_state["syncing"] = True
            sync_state["current_iid"] = iid
            win.after_idle(lambda: (_apply_sync(iid, src_tv), sync_state.update(syncing=False)))

        for tv in (tv_left, tv_mid, tv_right):
            tv.bind("<<TreeviewSelect>>", lambda e, src=tv: _on_select(src, e))

        # selezione iniziale se disponibile
        try:
            first_iid = next(i for i in tv_left.get_children() if i != "units")
            tv_left.selection_set(first_iid); tv_left.focus(first_iid); tv_left.see(first_iid)
        except StopIteration:
            pass

    # --- handler: Carica TDMS --------------------------------------------------------
    def do_load_tdms():
        path = filedialog.askopenfilename(
            title="Seleziona un file TDMS",
            filetypes=[("TDMS files", "*.tdms"), ("Tutti i file", "*.*")]
        )
        if not path:
            return
        state["tdms_path"] = path
        render_contract_and_loop(state["tdms_path"])
        render_tables(state["tdms_path"])

    # Render iniziale: NESSUN caricamento automatico
    render_contract_and_loop(state["tdms_path"])
    render_tables(state["tdms_path"])
    
    # Footer
    footer = tk.Frame(tab, bg="#ffffff")
    footer.grid(row=2, column=0, sticky="ew", padx=16, pady=(8,16))
    for i in range(4):
        footer.columnconfigure(i, weight=1, uniform="btns")

    def do_open_folder():
        path = state.get("tdms_path")
        if not path:
            messagebox.showwarning("Cartella", "Nessun file TDMS caricato."); return
        if sys.platform.startswith("win"):
            import subprocess
            try: subprocess.run(["explorer", "/select,", path], check=False)
            except Exception: subprocess.run(["explorer", os.path.dirname(path)], check=False)
        elif sys.platform == "darwin":
            import subprocess
            try: subprocess.run(["open", "-R", path], check=False)
            except Exception: subprocess.run(["open", os.path.dirname(path)], check=False)
        else:
            import subprocess
            subprocess.run(["xdg-open", os.path.dirname(path)], check=False)

    def do_open_tdms():
        path = state.get("tdms_path")
        if not path:
            messagebox.showwarning("File TDMS", "Nessun file TDMS caricato."); return
        _open_file_with_os(path)

    def _export_pdf():
        if not REPORTLAB_OK:
            messagebox.showinfo("Export PDF", "ReportLab non è installato.\nInstalla con: pip install reportlab"); return
        tdms_basename = os.path.basename(state.get("tdms_path") or "")
        default_name = f"TestCertificate_{cert_num or 'N'}_{test_date or 'date'}.pdf"
        pdf_path = filedialog.asksaveasfilename(title="Esporta PDF", defaultextension=".pdf",
                                               initialfile=default_name, filetypes=[("PDF","*.pdf")])
        if not pdf_path: return
        meta_lines = [
            f"Certificate No.: {cert_num or '—'}",
            f"Job: {job_dash or '—'}",
            f"Pump: {pump_dash or '—'}",
            f"Test Date: {test_date or '—'}",
            f"File: {tdms_basename}",
        ]
        try:
            _export_pdf_placeholder(pdf_path, "TEST CERTIFICATE", meta_lines)
            messagebox.showinfo("Export PDF", f"PDF esportato:\n{pdf_path}")
        except Exception as e:
            messagebox.showerror("Export PDF", f"Errore durante l'export:\n{e}")

    tk.Button(footer, text="Apri cartella", command=do_open_folder, width=14)\
        .grid(row=0, column=0, sticky="ew", padx=6)
    tk.Button(footer, text="Apri file TDMS", command=do_open_tdms, width=14)\
        .grid(row=0, column=1, sticky="ew", padx=6)
    tk.Button(footer, text="Esporta PDF…", command=_export_pdf, width=14)\
        .grid(row=0, column=2, sticky="ew", padx=6)
    tk.Button(footer, text="Chiudi", command=win.destroy, width=12)\
        .grid(row=0, column=3, sticky="ew", padx=(6,0))


