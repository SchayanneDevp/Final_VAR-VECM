# -*- coding: utf-8 -*-
"""
MODELO VECM (Johansen) AUTOMATICO (Econometria II)
=================================================
Usted SOLO escribe el nombre del archivo (celda 0). El codigo:
  1. Lee la base (.xlsx/.xls/.csv), elige la hoja, detecta la columna de tiempo, la frecuencia
     y el periodo de inicio, y decide que columnas son variables (descarta ids, constantes, texto, etc.).
  2. Muestra un SONDEO de la base (consola + explorador_datos.html con tabla consultable).
  3. Sugiere log (si procede), ordena las variables para Cholesky (exogeneidad por Granger).
  4. Decide estructura: VAR en niveles / VAR en diferencias / VECM(r) usando ADF+KPSS, estabilidad
     de las raices, criterios de informacion (AIC/BIC/HQ/FPE) + diagnosticos, y Johansen (traza y lmax).
  5. Estima el modelo final, hace diagnosticos, pronostico, IRF/OIRF con bandas bootstrap
     y escribe un informe HTML (con boton de PDF) que incluye la BITACORA de cada decision.
Todo se puede forzar con AJUSTES (opcional), pero normalmente queda vacio.

NOVEDADES de esta version (perfil "curso", activo por defecto; ponga AJUSTES["perfil"]="libre" para el comportamiento anterior):
  * ADF con regression="n" y maxlag=3 (como el taller); VECM con deterministic="ci" y Johansen det_order=0.
  * Johansen se corre SIEMPRE (tambien si las series salen I(0)) como confirmacion.
  * Si el modelo es VAR en diferencias, p_dif = p - 1 (el VECM(p-1) con r = 0).
  * Exporta resultados/auto_<base>.json, el mismo formato que usa generar_informe.py (poster multi-base).
  * VECM_ARCHIVO / VECM_AJUSTES (variables de entorno) permiten correrlo desde ejecutar_todo.py.
"""
# %% 0. LO UNICO QUE USTED EDITA ----------------------------------------------------
ARCHIVO = "datos_escenario2_I(1)_cointegradas.xlsx"   # <- nombre de su base (.xlsx / .xls / .csv, en la carpeta de este script)

AJUSTES = {"modelo": "VECM"}   # NO borre "modelo". Opcional: "columnas", "orden", "frecuencia", "inicio", "p", "alpha", ...

# %% 1. Paquetes y parametros por defecto -------------------------------------------
import os as _os
import re, io, json, html as _html, datetime as _dt, warnings, sys, os, subprocess
from pathlib import Path
ARCHIVO = os.environ.get("VECM_ARCHIVO", ARCHIVO)
if os.environ.get("VECM_AJUSTES"):
    AJUSTES = {**AJUSTES, **json.loads(os.environ["VECM_AJUSTES"])}
import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
from scipy import stats

warnings.filterwarnings("ignore")
try:
    from statsmodels.tsa.api import VAR
    from statsmodels.tsa.vector_ar.vecm import VECM, coint_johansen
    from statsmodels.tsa.stattools import adfuller, kpss
    from statsmodels.stats.diagnostic import het_arch
    SM_OK = True
except ImportError:
    SM_OK = False

DEFECTO = dict(
    hoja=None, columnas=None, orden=None, frecuencia=None, inicio=None,
    alpha=0.05, criterio_p="aic", p=None, maxlags=None, rango=None, johansen_det=None,
    usar_log=None,                  # None = decide el codigo; True/False = fuerza
    auto_orden=True, max_vars=5, max_nan=0.20, interp_max=3, cercano_a_1=0.95,
    adf_maxlag=None, lags_portm=10, lags_arch=12,
    pasos_pron=12, pasos_irf=18, boot_runs=100, semilla=202601, nivel_ic=0.95,
    generar_pdf=False, carpeta_salida="salidas_auto", filas_tabla=2000,
    perfil="curso",                 # "curso" = convenciones del taller; "libre" = decisiones 100% automaticas
    adf_regresion=None,             # None | "n" | "c" | "ct" (el perfil curso fija "n")
    vecm_det=None,                  # None | "n" | "co" | "ci" | "lo" | "colo" | "cili" (el perfil curso fija "ci")
    johansen_siempre=True, poster=False,
    modelo="auto",                  # "auto" | "VAR" | "VECM": VAR_auto.py fija "VAR" y VECM_auto.py fija "VECM"
    p_dif=None,                     # rezagos del VAR en diferencias (None: p-1 en perfil curso)
)
if AJUSTES.get("modelo", "auto") not in ("auto", "VAR", "VECM"):
    raise ValueError("AJUSTES['modelo'] debe ser 'auto', 'VAR' o 'VECM'")
_mal = sorted(set(AJUSTES) - set(DEFECTO))
if _mal:
    raise KeyError(f"AJUSTES contiene claves no validas: {_mal}. Validas: {sorted(DEFECTO)}")
C = {**DEFECTO, **AJUSTES}
if C["perfil"] == "curso":
    for _k, _v in dict(adf_regresion="n", adf_maxlag=3, vecm_det="ci", johansen_det=0).items():
        if _k not in AJUSTES:
            C[_k] = _v
ALFA = C["alpha"]
BASE = Path(__file__).resolve().parent if "__file__" in globals() else Path.cwd()
SAL = BASE / C["carpeta_salida"] / (re.sub(r"\W+", "_", Path(ARCHIVO).stem) + ("" if C["modelo"] == "auto" else "_" + C["modelo"]))
SAL.mkdir(parents=True, exist_ok=True)
LOG = []                                    # bitacora de decisiones


def reg(paso, decision, motivo):
    LOG.append((paso, decision, motivo))
    print(f"[{paso}] {decision}  <-  {motivo}")


# %% 2. LECTURA Y SONDEO DE LA BASE (no requiere statsmodels) --------------------------
RX_TIEMPO = re.compile(r"fecha|date|per[ií]odo|periodo|period|tiempo|time|a[nñ]o|year|trim|quarter|mes$|month|^obs", re.I)
RX_ID = re.compile(r"^(id|index|indice|[íi]ndice|n|no|num|obs|unnamed.*|codigo|c[óo]digo)$", re.I)
RX_Q = re.compile(r"^\s*(\d{4})\s*[-_/ ]?\s*[QqTt]\s*([1-4])\s*$")
RX_M = re.compile(r"^\s*(\d{4})\s*[-_/ ]\s*[Mm]?(\d{1,2})\s*$")


def a_numerico(s):
    """Serie numerica (float) o None si la columna no es numerica (acepta coma decimal)."""
    if pd.api.types.is_bool_dtype(s):
        return None
    if pd.api.types.is_numeric_dtype(s):
        return s.astype(float)
    t = s.astype(str).str.strip().str.replace(r"\s", "", regex=True)
    t = pd.to_numeric(t.str.replace(",", ".", regex=False), errors="coerce")
    ok = s.notna().sum()
    return t if ok > 0 and t.notna().sum() >= 0.9 * ok else None


def leer_base(nombre):
    ruta = Path(str(nombre).strip().strip('"'))
    if not ruta.is_absolute():
        # acepta: ruta relativa a la carpeta del script, relativa a donde se ejecuta, o solo el nombre del archivo
        for cand_ in (BASE / ruta, Path.cwd() / ruta, BASE / ruta.name):
            if cand_.exists():
                ruta = cand_
                break
        else:
            ruta = BASE / ruta.name
    if not ruta.exists():
        cand = sorted(p.name for p in BASE.iterdir() if p.suffix.lower() in (".xlsx", ".xls", ".csv", ".txt"))
        raise FileNotFoundError(f"No encuentro '{ruta}'. Archivos de datos en la carpeta: {cand}")
    if ruta.suffix.lower() in (".csv", ".txt"):
        return ruta, {"(csv)": pd.read_csv(ruta, sep=None, engine="python")}
    return ruta, pd.read_excel(ruta, sheet_name=None)


def elegir_hoja(hojas, C):
    if C["hoja"] is not None:
        return C["hoja"], hojas[C["hoja"]]
    puntaje = {}
    for k, d in hojas.items():
        nn = sum(a_numerico(d[c]) is not None for c in d.columns)
        puntaje[k] = nn * len(d)
    k = max(puntaje, key=puntaje.get)
    return k, hojas[k]


def freq_desde_dias(dias):
    if dias <= 1.5: return "D"
    if dias <= 9: return "W"
    if dias <= 35: return "M"
    if dias <= 100: return "Q"
    if 330 <= dias <= 400: return "Y"
    return None


def construir_indice(d, C):
    """Devuelve (indice, columnas_usadas_como_tiempo, frecuencia|None, nota)."""
    n = len(d)
    if C["inicio"] is not None and C["frecuencia"] is not None:
        cols = [c for c in d.columns if RX_TIEMPO.search(str(c))]
        idx = pd.period_range(start=C["inicio"], periods=n, freq=C["frecuencia"], name="tiempo")
        return idx, cols, C["frecuencia"], "indice fijado por el usuario (inicio/frecuencia)"
    cols = list(d.columns)
    for c in cols:                                        # 1) columna datetime
        if pd.api.types.is_datetime64_any_dtype(d[c]):
            dt = pd.DatetimeIndex(d[c])
            fr = C["frecuencia"] or freq_desde_dias(float(np.median((dt[1:] - dt[:-1]).total_seconds()) / 86400)) if n > 1 else None
            if fr:
                return pd.PeriodIndex(dt, freq=fr, name="tiempo"), [c], fr, f"columna de fechas '{c}'"
    for c in cols:                                        # 2) texto 1990Q1 / 1990-03
        if d[c].dtype == object or pd.api.types.is_string_dtype(d[c]):
            v = d[c].dropna().astype(str)
            if len(v) and v.map(lambda x: bool(RX_Q.match(x))).mean() > 0.9:
                p = d[c].astype(str).map(lambda x: (lambda m: f"{m.group(1)}Q{m.group(2)}")(RX_Q.match(x)))
                return pd.PeriodIndex(p, freq="Q", name="tiempo"), [c], "Q", f"texto trimestral en '{c}'"
            if len(v) and v.map(lambda x: bool(RX_M.match(x)) and 1 <= int(RX_M.match(x).group(2)) <= 12).mean() > 0.9:
                p = d[c].astype(str).map(lambda x: (lambda m: f"{m.group(1)}-{int(m.group(2)):02d}")(RX_M.match(x)))
                return pd.PeriodIndex(p, freq="M", name="tiempo"), [c], "M", f"texto mensual en '{c}'"
            try:                                          # 3) texto de fecha cualquiera
                dt = pd.to_datetime(d[c], errors="raise")
                fr = freq_desde_dias(float(np.median((pd.DatetimeIndex(dt)[1:] - pd.DatetimeIndex(dt)[:-1]).total_seconds()) / 86400))
                if fr and RX_TIEMPO.search(str(c)):
                    return pd.PeriodIndex(dt, freq=fr, name="tiempo"), [c], fr, f"fechas en texto '{c}'"
            except Exception:
                pass

    def es_anio(s):
        v = a_numerico(s)
        return v is not None and v.notna().all() and v.between(1800, 2200).all() and (v % 1 == 0).all()
    anio = [c for c in cols if re.search(r"a[nñ]o|year|^yr", str(c), re.I) and es_anio(d[c])]
    sub = [c for c in cols if re.search(r"trim|quarter|^q$|mes$|month", str(c), re.I)]
    if anio and sub:                                      # 4) a?o + trimestre/mes
        a = a_numerico(d[anio[0]]).astype(int)
        s = a_numerico(d[sub[0]]).astype(int)
        esq = re.search(r"trim|quarter|^q$", str(sub[0]), re.I) is not None
        idx = pd.PeriodIndex([f"{y}Q{q}" if esq else f"{y}-{q:02d}" for y, q in zip(a, s)], freq="Q" if esq else "M", name="tiempo")
        return idx, [anio[0], sub[0]], "Q" if esq else "M", f"'{anio[0]}' + '{sub[0]}'"
    if anio:                                              # 5) solo a?o
        a = a_numerico(d[anio[0]]).astype(int)
        if a.is_monotonic_increasing and (np.diff(a) == 1).all():
            return pd.PeriodIndex(a.astype(str), freq="Y", name="tiempo"), [anio[0]], "Y", f"anual en '{anio[0]}'"
    return pd.RangeIndex(n, name="tiempo"), [], None, "SIN columna de tiempo: se usa indice 0..T-1 (declare inicio/frecuencia en AJUSTES para fechas)"


def perfilar(d, cols_t, C):
    """Tabla de perfil por columna y lista final de variables."""
    filas = []
    for c in d.columns:
        s = d[c]
        v = a_numerico(s)
        rol, why = "usada", ""
        if c in cols_t:
            rol, why = "tiempo", "indice temporal"
        elif v is None:
            rol, why = "excluida", "no numerica"
        else:
            nn = v.dropna()
            if len(nn) == 0:
                rol, why = "excluida", "vacia"
            elif s.isna().mean() > C["max_nan"]:
                rol, why = "excluida", f"{s.isna().mean():.0%} faltantes (> {C['max_nan']:.0%})"
            elif nn.nunique() <= 1:
                rol, why = "excluida", "constante"
            elif RX_ID.match(str(c).strip()):
                rol, why = "excluida", "parece identificador"
            elif (nn % 1 == 0).all() and np.all(np.diff(nn.to_numpy()) == 1):
                rol, why = "excluida", "contador 1,2,3..."
            elif nn.nunique() < 6 and len(nn) > 30:
                rol, why = "excluida", "pocos valores distintos (categorica/dummy)"
        nn = v.dropna() if v is not None else pd.Series(dtype=float)
        filas.append(dict(columna=c, dtype=str(s.dtype), faltantes=s.isna().mean(), rol=rol, motivo=why,
                          min=nn.min() if len(nn) else np.nan, max=nn.max() if len(nn) else np.nan,
                          media=nn.mean() if len(nn) else np.nan, desv=nn.std() if len(nn) else np.nan))
    P = pd.DataFrame(filas).set_index("columna")
    if C["columnas"] is not None:
        bajo = {str(c).lower(): c for c in d.columns}
        pedidas = [bajo.get(str(x).lower(), x) for x in C["columnas"]]
        falta = [x for x in pedidas if x not in d.columns]
        if falta:
            raise KeyError(f"Columnas pedidas que no existen: {falta}. Disponibles: {list(d.columns)}")
        for c in P.index:
            if c in cols_t:
                continue
            P.loc[c, ["rol", "motivo"]] = ("usada", "elegida por el usuario") if c in pedidas else ("excluida", "no incluida en AJUSTES['columnas']")
        usadas = pedidas
    else:
        usadas = [c for c in P.index if P.loc[c, "rol"] == "usada"]
        if len(usadas) > C["max_vars"]:
            orden = sorted(usadas, key=lambda c: (P.loc[c, "faltantes"], list(P.index).index(c)))[:C["max_vars"]]
            for c in usadas:
                if c not in orden:
                    P.loc[c, ["rol", "motivo"]] = ("excluida", f"mas de {C['max_vars']} variables: se prioriza la mas completa (use AJUSTES['columnas'])")
            usadas = [c for c in usadas if c in orden]
    if len(usadas) < 2:
        raise ValueError("Se necesitan al menos 2 variables numericas utiles. Revise la tabla de perfil o use AJUSTES['columnas'].")
    return P, usadas


def _spark(s, w=130, h=30):
    v = s.dropna().to_numpy(float)
    if len(v) < 2:
        return ""
    if len(v) > 300:
        v = v[:: int(np.ceil(len(v) / 300))]
    lo, hi = v.min(), v.max()
    rg = (hi - lo) or 1.0
    xs = np.linspace(1, w - 1, len(v))
    ys = h - 3 - (v - lo) / rg * (h - 6)
    pts = " ".join(f"{x:.1f},{y:.1f}" for x, y in zip(xs, ys))
    return (f"<svg width='{w}' height='{h}' viewBox='0 0 {w} {h}'><polyline points='{pts}' fill='none' "
            f"stroke='#0f766e' stroke-width='1.3'/></svg>")


def _f(v, nd=4):
    if v is None or (isinstance(v, float) and np.isnan(v)):
        return "&ndash;"
    if isinstance(v, (int, np.integer)):
        return str(int(v))
    return f"{v:.{nd}f}" if (abs(v) >= 1e-4 or v == 0) else f"{v:.2e}"


def html_sondeo(d, P, usadas, idx, freq, hoja, ruta, nota_t, tabla_id="tbl-datos"):
    ini, fin = str(idx[0]), str(idx[-1])
    cards = [("Archivo", ruta.name), ("Hoja", str(hoja)), ("Observaciones", str(len(d))), ("Columnas", str(d.shape[1])),
             ("Periodo", f"{ini} &rarr; {fin}"), ("Frecuencia", freq or "sin fecha"), ("Variables usadas", str(len(usadas)))]
    kp = "".join(f"<div class='kpi'><b>{b}</b><span>{a}</span></div>" for a, b in cards)
    f = ["<table class='tbl'><thead><tr><th>columna</th><th>tipo</th><th>rol</th><th>motivo</th><th>% faltante</th>"
         "<th>min</th><th>media</th><th>max</th><th>desv.</th><th>forma</th></tr></thead><tbody>"]
    for c, r in P.iterrows():
        cls = {"usada": "ok", "tiempo": "tm"}.get(r["rol"], "no")
        v = a_numerico(d[c])
        f.append(f"<tr><th>{_html.escape(str(c))}</th><td>{r['dtype']}</td><td><span class='bd {cls}'>{r['rol']}</span></td>"
                 f"<td class='l'>{_html.escape(r['motivo'])}</td><td>{r['faltantes']:.1%}</td><td>{_f(r['min'], 3)}</td>"
                 f"<td>{_f(r['media'], 3)}</td><td>{_f(r['max'], 3)}</td><td>{_f(r['desv'], 3)}</td>"
                 f"<td>{_spark(v) if v is not None and r['rol'] != 'tiempo' else ''}</td></tr>")
    f.append("</tbody></table>")
    vista = d.head(C["filas_tabla"]).copy()
    vista.insert(0, "tiempo", [str(x) for x in idx[:len(vista)]])
    tabla = vista.to_html(index=False, border=0, classes="tbl", table_id=tabla_id, na_rep="", float_format=lambda v: f"{v:.6g}")
    js = ("<script>(function(){var q=document.getElementById('buscar'),t=document.getElementById('" + tabla_id + "');"
          "if(!q||!t)return;q.addEventListener('input',function(){var s=q.value.toLowerCase();"
          "Array.prototype.forEach.call(t.tBodies[0].rows,function(r){r.style.display=r.innerText.toLowerCase().indexOf(s)<0?'none':''});});"
          "Array.prototype.forEach.call(t.tHead.rows[0].cells,function(h,i){h.style.cursor='pointer';h.addEventListener('click',function(){"
          "var rows=Array.prototype.slice.call(t.tBodies[0].rows),a=h._a=!h._a;rows.sort(function(x,y){var u=x.cells[i].innerText,v=y.cells[i].innerText,"
          "n=parseFloat(u),m=parseFloat(v);var r=(!isNaN(n)&&!isNaN(m))?n-m:u.localeCompare(v);return a?r:-r});"
          "rows.forEach(function(r){t.tBodies[0].appendChild(r)})})})})();</script>")
    nota = f"<div class='ans'>Tiempo: {_html.escape(nota_t)}. Variables que entran al modelo: <b>{', '.join(map(str, usadas))}</b>.</div>"
    mas = f" (se muestran las primeras {len(vista)} de {len(d)} filas)" if len(d) > len(vista) else ""
    return (f"<div class='kpis'>{kp}</div>{nota}<h3>Perfil de columnas</h3>{''.join(f)}"
            f"<h3>Consulta de la base{mas}</h3><input id='buscar' class='srch' placeholder='Filtrar filas (texto o numero)...'>"
            f"<div class='scroll'>{tabla}</div>{js}")


CSS = """
:root{--ink:#0f172a;--mut:#475569;--ac:#0f766e;--bg:#f8fafc;--card:#fff;--line:#e2e8f0;--warn:#b45309}
*{box-sizing:border-box}body{margin:0;background:var(--bg);color:var(--ink);font:15px/1.6 'Segoe UI',system-ui,-apple-system,Helvetica,Arial,sans-serif}
header{background:linear-gradient(135deg,#0f172a,#134e4a);color:#fff;padding:42px 6vw 34px}
header h1{margin:0 0 6px;font-size:28px}header p{margin:2px 0;color:#cbd5e1}
main{max-width:1060px;margin:0 auto;padding:22px 20px 60px}
nav.toc{background:var(--card);border:1px solid var(--line);border-radius:10px;padding:14px 20px;margin:18px 0}
nav.toc a{color:var(--ac);text-decoration:none;margin-right:14px;white-space:nowrap}
.kpis{display:grid;grid-template-columns:repeat(auto-fit,minmax(140px,1fr));gap:12px;margin:18px 0}
.kpi{background:var(--card);border:1px solid var(--line);border-radius:10px;padding:12px 14px}
.kpi b{display:block;font-size:19px;color:var(--ac);overflow-wrap:anywhere}.kpi span{color:var(--mut);font-size:12.5px}
h2{margin:34px 0 8px;padding-bottom:6px;border-bottom:2px solid var(--ac);font-size:20px}h3{font-size:15.5px;margin:14px 0 6px}
.item{background:var(--card);border:1px solid var(--line);border-radius:10px;padding:14px 18px;margin:12px 0;break-inside:avoid}
.ans{border-left:4px solid var(--ac);background:#f0fdfa;padding:8px 12px;margin:8px 0;border-radius:0 6px 6px 0}
.warn{border-left:4px solid var(--warn);background:#fffbeb;padding:8px 12px;margin:8px 0;border-radius:0 6px 6px 0}
table.tbl{border-collapse:collapse;margin:8px 0;font-size:12.5px;font-variant-numeric:tabular-nums}
.tbl th,.tbl td{border-bottom:1px solid var(--line);padding:4px 9px;text-align:right}.tbl td.l{text-align:left}
.tbl thead th{background:#f1f5f9;border-bottom:2px solid #cbd5e1;position:sticky;top:0}.tbl tbody th{text-align:left;background:#fafcfd}
td.best{background:#ccfbf1!important;font-weight:700;color:#0f766e}
.bd{padding:1px 8px;border-radius:10px;font-size:11.5px;font-weight:600}.bd.ok{background:#ccfbf1;color:#0f766e}
.bd.no{background:#fee2e2;color:#991b1b}.bd.tm{background:#dbeafe;color:#1e40af}
.dec{border:2px solid var(--ac);background:#f0fdfa;border-radius:10px;padding:12px 16px;margin:14px 0}.dec.bad{border-color:var(--warn);background:#fffbeb}
.scroll{max-height:420px;overflow:auto;border:1px solid var(--line);border-radius:8px;background:#fff}.srch{width:100%;padding:8px 10px;border:1px solid var(--line);border-radius:8px;margin:6px 0}
pre{background:#0f172a;color:#e2e8f0;padding:12px;border-radius:8px;overflow:auto;font-size:11.5px;line-height:1.4}
figure{margin:10px 0;text-align:center}figure svg{max-width:100%;height:auto}figcaption{color:var(--mut);font-size:12.5px}
.btn{position:fixed;right:18px;bottom:18px;background:var(--ac);color:#fff;border:0;border-radius:24px;padding:11px 20px;font-size:14px;cursor:pointer;box-shadow:0 4px 14px rgba(0,0,0,.25)}
footer{color:var(--mut);font-size:12.5px;border-top:1px solid var(--line);margin-top:30px;padding-top:12px}
@page{size:A4;margin:14mm 12mm}
@media print{body{background:#fff;font-size:10.5pt}.btn,nav.toc,.srch{display:none}.scroll{max-height:none;overflow:visible}
header{padding:18px 0;-webkit-print-color-adjust:exact;print-color-adjust:exact}h2{break-after:avoid}pre{white-space:pre-wrap;overflow:visible}
.ans,.warn,.kpi,.bd,.dec{-webkit-print-color-adjust:exact;print-color-adjust:exact}}
"""

RUTA, HOJAS = leer_base(ARCHIVO)
HOJA, RAW = elegir_hoja(HOJAS, C)
RAW = RAW.dropna(how="all").reset_index(drop=True)
RAW.columns = [str(c).strip() for c in RAW.columns]
print(f"\n=== SONDEO DE LA BASE: {RUTA.name} | hoja '{HOJA}' (hojas: {list(HOJAS)}) ===")
RAW.info()
IDX, COLS_T, FREQ, NOTA_T = construir_indice(RAW, C)
reg("Tiempo", f"frecuencia={FREQ or 'ninguna'}, {IDX[0]} .. {IDX[-1]}", NOTA_T)
PERFIL, USADAS = perfilar(RAW, COLS_T, C)
print(PERFIL.drop(columns=["dtype"]).round(4).to_string())
reg("Variables", ", ".join(USADAS), "columnas numericas utiles (ver perfil)")
(SAL / "explorador_datos.html").write_text(
    f"<!doctype html><html lang='es'><head><meta charset='utf-8'><meta name='viewport' content='width=device-width,initial-scale=1'>"
    f"<title>Explorador de datos</title><style>{CSS}</style></head><body><header><h1>Explorador de la base</h1>"
    f"<p>{_html.escape(RUTA.name)}</p></header><main>{html_sondeo(RAW, PERFIL, USADAS, IDX, FREQ, HOJA, RUTA, NOTA_T)}</main></body></html>",
    encoding="utf-8")
print("Explorador:", SAL / "explorador_datos.html")
if not SM_OK:
    raise SystemExit("Sondeo listo. Para estimar instale statsmodels:  pip install statsmodels")

# %% 3. Preparacion: faltantes, log, orden de Cholesky ------------------------------
DATOS_ORIG = pd.DataFrame({c: a_numerico(RAW[c]).to_numpy() for c in USADAS}, index=IDX)
if not DATOS_ORIG.index.is_monotonic_increasing:
    DATOS_ORIG = DATOS_ORIG.sort_index()
    reg("Orden temporal", "se ordeno por fecha", "el indice no era creciente")
if isinstance(DATOS_ORIG.index, pd.PeriodIndex):
    completo = pd.period_range(DATOS_ORIG.index[0], DATOS_ORIG.index[-1], freq=DATOS_ORIG.index.freq)
    if len(completo) != len(DATOS_ORIG):
        reg("Periodos faltantes", f"se insertan {len(completo) - len(DATOS_ORIG)} periodos", "huecos en la serie temporal")
        DATOS_ORIG = DATOS_ORIG.reindex(completo)
n_na = int(DATOS_ORIG.isna().sum().sum())
if n_na:
    DATOS_ORIG = DATOS_ORIG.interpolate(limit=C["interp_max"], limit_area="inside").dropna()
    reg("Faltantes", f"{n_na} valores interpolados linealmente (max {C['interp_max']} seguidos); filas incompletas eliminadas",
        "VAR/VECM requieren panel completo")
T0, K = DATOS_ORIG.shape
nombres = list(DATOS_ORIG.columns)
if T0 < 30 + 5 * K:
    reg("Aviso", f"muestra pequena (T={T0}, K={K})", "resultados con baja potencia; interprete con cautela")


def sugerir_log(Y):
    out = {}
    for c in Y:
        s = Y[c]
        if (s > 0).all() and s.max() / s.min() >= 3:
            dy = s.diff().abs().dropna()
            r = np.corrcoef(dy, s.loc[dy.index])[0, 1]
            out[c] = bool(r > 0.3)
        else:
            out[c] = False
    return out


LOGS = sugerir_log(DATOS_ORIG)
if isinstance(C["usar_log"], (list, tuple)):
    malos = [c for c in C["usar_log"] if c not in nombres]
    if malos:
        raise KeyError(f"usar_log: columnas inexistentes {malos}. Variables: {nombres}")
    LOGS = {c: (c in C["usar_log"] and bool((DATOS_ORIG[c] > 0).all())) for c in nombres}
elif C["usar_log"] is True:
    LOGS = {c: bool((DATOS_ORIG[c] > 0).all()) for c in nombres}
elif C["usar_log"] is False:
    LOGS = {c: False for c in nombres}
Y = DATOS_ORIG.copy()
for c in nombres:
    if LOGS[c]:
        Y[c] = np.log(Y[c])
if any(LOGS.values()):
    reg("Transformacion", "log en: " + ", ".join(c for c in nombres if LOGS[c]),
        "serie positiva, escala >= 3x y la volatilidad crece con el nivel (corr(|dy|, y) > 0.3)")
else:
    reg("Transformacion", "sin log", "no hay series positivas con volatilidad proporcional al nivel")


def espec_det(x):
    """Especificacion determinista de la ADF: n / c / ct (fijada por el perfil del curso o por AJUSTES)."""
    if C["adf_regresion"]:
        return C["adf_regresion"]
    x = x.dropna()
    dx = x.diff().dropna()
    if abs(x.mean()) < 0.1 * x.std():
        return "n"
    t = dx.mean() / (dx.std(ddof=1) / np.sqrt(len(dx)))
    return "ct" if abs(t) > 2 else "c"


def maxlags_auto(T, K):
    if C["maxlags"]:
        return int(C["maxlags"])
    base = {"M": 12, "Q": 6, "Y": 4, "W": 8, "D": 8}.get(FREQ, 6)
    return int(max(1, min(base, int((T - 1) / (K + 1)) - 2)))


def adf_tabla(X, nivel=True):
    ml = C["adf_maxlag"] or int(min(12 * (len(X) / 100) ** 0.25, 8))
    filas = []
    for c in X.columns:
        x = X[c].dropna()
        e = espec_det(x)
        r = adfuller(x, maxlag=ml, autolag="AIC", regression=e)
        kp = np.nan
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                kp = kpss(x, regression="ct" if e == "ct" else "c", nlags="auto")[1]
        except Exception:
            pass
        rechaza, kpr = r[1] < ALFA, (not np.isnan(kp)) and kp < ALFA
        clase = ("estacionaria I(0)" if rechaza and not kpr else "raiz unitaria" if (not rechaza) and kpr
                 else "ambigua (ADF y KPSS rechazan)" if rechaza and kpr else "inconclusa (baja potencia)")
        filas.append(dict(variable=c, espec=e, t=r[0], p_value=r[1], rezagos=r[2], crit_5=r[4]["5%"],
                          adf="rechaza raiz unitaria" if rechaza else "no rechaza", kpss_p=kp, lectura=clase))
    return pd.DataFrame(filas).set_index("variable")


ADF_N = adf_tabla(Y)
print("\nADF + KPSS (niveles):\n", ADF_N.round(4).to_string())


def tendencia_var(adf):
    return "n" if (adf["espec"] == "n").all() else "c"


def orden_exogeneidad(Z, p, trend):
    """Orden de Cholesky: variables mas exogenas primero (Granger)."""
    r = VAR(Z.reset_index(drop=True)).fit(p, trend=trend)
    k = Z.shape[1]
    sc = {}
    for j, vj in enumerate(Z.columns):
        a, b = [], []
        for i, vi in enumerate(Z.columns):
            if i == j:
                continue
            a.append(r.test_causality(vj, [vi], kind="f").pvalue)       # vi causa a vj ?  (p alto = vj exogena)
            b.append(1 - r.test_causality(vi, [vj], kind="f").pvalue)   # vj causa a vi ?  (cerca de 1 = vj influye)
        sc[vj] = 0.5 * (np.mean(a) + np.mean(b))
    return sorted(sc, key=sc.get, reverse=True), sc


if C["orden"] is not None:
    falta = [v for v in C["orden"] if v not in nombres]
    if falta or len(C["orden"]) != len(nombres):
        raise KeyError(f"AJUSTES['orden'] debe contener exactamente {nombres}")
    Y = Y[C["orden"]]
    reg("Orden Cholesky", " > ".join(C["orden"]), "fijado por el usuario")
elif C["auto_orden"]:
    todas_I0 = (ADF_N["p_value"] < ALFA).all()
    Z = Y if todas_I0 else Y.diff().dropna()
    pz = 1
    try:
        pz = max(1, int(VAR(Z.reset_index(drop=True)).select_order(maxlags=min(4, maxlags_auto(len(Z), K)), trend="c").selected_orders["aic"]))
    except Exception:
        pass
    try:
        ordn, SCORE = orden_exogeneidad(Z, pz, "c")
        Y = Y[ordn]
        reg("Orden Cholesky", " > ".join(ordn), f"exogeneidad por Granger en VAR({pz}) sobre {'niveles' if todas_I0 else 'diferencias'} "
            "(1o la variable que menos es causada y mas causa a las demas); heuristica: verifique con la teoria economica")
    except Exception as e:
        reg("Orden Cholesky", "orden del archivo", f"no se pudo calcular Granger ({e})")
nombres = list(Y.columns)
T, K = Y.shape
ML = maxlags_auto(T, K)

# %% 4. Funciones de analisis (nucleo de su codigo, generalizado) ---------------------


def tabla_criterios(Yn, trend, ml):
    sel = VAR(Yn).select_order(maxlags=ml, trend=trend)
    t = pd.DataFrame(sel.ics)
    n = len(t)
    t.index = range(1, n + 1) if n == ml else range(0, n)
    t.index.name = "p"
    t = t.loc[t.index >= 1, [c for c in ("aic", "bic", "hqic", "fpe") if c in t]]
    return t, {c: int(t[c].idxmin()) for c in t}


def autovalores(coefs):
    p, k, _ = coefs.shape
    top = np.hstack(list(coefs))
    if p == 1:
        return np.linalg.eigvals(top)
    bot = np.hstack([np.eye(k * (p - 1)), np.zeros((k * (p - 1), k))])
    return np.linalg.eigvals(np.vstack([top, bot]))


def estabilidad(coefs):
    ev = autovalores(np.asarray(coefs))
    mx = float(np.max(np.abs(ev)))
    if mx >= 1:
        msg = "NO estable: raices unitarias/explosivas"
    elif mx > C["cercano_a_1"]:
        msg = "estable pero muy persistente (cerca de raiz unitaria)"
    else:
        msg = "estable (estacionario)"
    return dict(eig=ev, max_eig=mx, estable=mx < 1, mensaje=msg)


def comparar_modelos(Yn, ic, trend):
    filas = []
    for p in list(ic.index):
        r = VAR(Yn).fit(int(p), trend=trend)
        e = np.asarray(r.resid)
        nl = max(C["lags_portm"], int(p) + 4)
        w = r.test_whiteness(nlags=nl, adjusted=False)
        fila = {c.upper(): ic.loc[p, c] for c in ic}
        fila.update(max_eig=float(np.max(np.abs(autovalores(np.asarray(r.coefs))))), portm_p=float(w.pvalue),
                    arch_min_p=float(min(het_arch(e[:, k], nlags=C["lags_arch"])[1] for k in range(e.shape[1]))),
                    jb_p=float(r.test_normality().pvalue), p=int(p))
        filas.append(fila)
    t = pd.DataFrame(filas).set_index("p")
    t["estable"] = t["max_eig"] < C["cercano_a_1"]
    t["blanco"] = t["portm_p"] >= ALFA
    return t


def elegir_p(ic, mejor, comp):
    crit = C["criterio_p"]
    if C["p"]:
        return int(C["p"]), "p fijado por el usuario"
    ok = comp[comp["estable"] & comp["blanco"]]
    if len(ok):
        return int(ok[crit.upper()].idxmin()), f"minimo {crit.upper()} entre los p con raices estables y sin autocorrelacion (portmanteau p >= {ALFA})"
    cb = comp[comp["blanco"]]
    if len(cb):
        return int(cb[crit.upper()].idxmin()), f"minimo {crit.upper()} entre los p sin autocorrelacion residual (ninguno es estable: serie no estacionaria)"
    return int(mejor[crit]), f"ningun p pasa el blanqueo; se usa el minimo {crit.upper()} (revise diagnosticos)"


def analizar(X, trend, adf, p_fijo=None):
    Xn = X.reset_index(drop=True)
    ml = maxlags_auto(len(X), X.shape[1])
    ic, mejor = tabla_criterios(Xn, trend, ml)
    comp = comparar_modelos(Xn, ic, trend)
    p, why = elegir_p(ic, mejor, comp)
    if p_fijo:
        p, why = int(p_fijo), ("fijado por el usuario (p_dif)" if C["p_dif"] else "p - 1 del VAR en niveles (perfil del curso: VECM(p-1) con r = 0)")
    res = VAR(Xn).fit(p, trend=trend)
    est = estabilidad(res.coefs)
    razones = []
    if not est["estable"]:
        razones.append("hay raices sobre/dentro del circulo unitario")
    elif est["max_eig"] >= C["cercano_a_1"]:
        razones.append(f"max|autovalor| = {est['max_eig']:.3f} >= {C['cercano_a_1']} (practicamente raiz unitaria)")
    nr = [v for v in adf.index if adf.loc[v, "p_value"] >= ALFA]
    if nr:
        razones.append("ADF no rechaza raiz unitaria en " + ", ".join(nr))
    return dict(X=X, adf=adf, ic=ic, mejor=mejor, p=p, why_p=why, comp=comp, res=res, est=est, trend=trend,
                ok=not razones, razones=razones)


def johansen_tabla(niveles, p, det):
    j = coint_johansen(niveles.reset_index(drop=True), det_order=det, k_ar_diff=max(p - 1, 0))
    k = niveles.shape[1]
    t = pd.DataFrame({"H0: rango <= r": range(k), "traza": j.lr1, "crit5_traza": j.cvt[:, 1],
                      "lmax": j.lr2, "crit5_lmax": j.cvm[:, 1]})
    t["traza_decide"] = np.where(t["traza"] > t["crit5_traza"], "rechaza", "no rechaza")
    r = 0
    while r < k and t.loc[r, "traza"] > t.loc[r, "crit5_traza"]:
        r += 1
    rm = 0
    while rm < k and t.loc[rm, "lmax"] > t.loc[rm, "crit5_lmax"]:
        rm += 1
    return t.set_index("H0: rango <= r"), r, rm


def sensibilidad_johansen(niveles, ps):
    filas = []
    for det in (-1, 0, 1):
        fila = {"det_order": det}
        for p in ps:
            try:
                fila[f"p={p}"] = johansen_tabla(niveles, p, det)[1]
            except Exception:
                fila[f"p={p}"] = np.nan
        filas.append(fila)
    return pd.DataFrame(filas).set_index("det_order")


def diagnosticos(res, nombres_):
    e = np.asarray(res.resid)
    nl = int(max(3, min(C["lags_portm"], len(e) // 4)))
    w = res.test_whiteness(nlags=nl, adjusted=False)
    arch = pd.DataFrame({v: dict(zip(["LM", "p_LM", "F", "p_F"], het_arch(e[:, k], nlags=C["lags_arch"])[:4]))
                         for k, v in enumerate(nombres_)}).T
    nm = res.test_normality()
    nu = pd.DataFrame({v: dict(JB=stats.jarque_bera(e[:, k]).statistic, p_value=stats.jarque_bera(e[:, k]).pvalue)
                       for k, v in enumerate(nombres_)}).T
    return dict(white=dict(stat=float(w.test_statistic), p=float(w.pvalue), df=int(w.df), nlags=nl), arch=arch,
                norm=dict(stat=float(nm.test_statistic), p=float(nm.pvalue), df=int(nm.df)), norm_uni=nu)


def indice_futuro(n):
    if isinstance(DATOS_ORIG.index, pd.PeriodIndex):
        return pd.period_range(DATOS_ORIG.index[-1] + 1, periods=n, freq=DATOS_ORIG.index.freq, name="tiempo")
    return pd.RangeIndex(len(DATOS_ORIG), len(DATOS_ORIG) + n, name="tiempo")


def bandas_irf_var(res, trend, H):
    rng = np.random.default_rng(C["semilla"])
    Yv = np.asarray(res.endog)
    Tn, k = Yv.shape
    p = res.k_ar
    U = np.asarray(res.resid)
    U = U - U.mean(0)
    coefs = np.asarray(res.coefs)
    cte = np.asarray(res.intercept) if trend != "n" else np.zeros(k)
    oi, oo = [], []
    for _ in range(C["boot_runs"]):
        e = U[rng.integers(0, len(U), Tn - p)]
        Yb = np.empty_like(Yv)
        Yb[:p] = Yv[:p]
        for t in range(p, Tn):
            Yb[t] = cte + sum(coefs[j] @ Yb[t - 1 - j] for j in range(p)) + e[t - p]
        try:
            ib = VAR(Yb).fit(p, trend=trend).irf(H)
            oi.append(ib.irfs)
            oo.append(ib.orth_irfs)
        except Exception:
            continue
    q = [(1 - C["nivel_ic"]) / 2 * 100, (1 + C["nivel_ic"]) / 2 * 100]
    return np.percentile(oi, q, axis=0), np.percentile(oo, q, axis=0)


def bandas_irf_vecm(res, Ylev, kd, r, det, H):
    """Bootstrap de residuales para un VECM; el termino determinista se recupera de los propios residuos."""
    rng = np.random.default_rng(C["semilla"])
    Yv = np.asarray(Ylev, float)
    Tn, k = Yv.shape
    p = kd + 1
    al, be, ga = np.asarray(res.alpha), np.asarray(res.beta), np.asarray(res.gamma)
    U = np.asarray(res.resid)
    dY = np.diff(Yv, axis=0)
    d = np.zeros((Tn - p, k))
    for t in range(p, Tn):
        ect = al @ (be.T @ Yv[t - 1])
        gam = sum(ga[:, i * k:(i + 1) * k] @ dY[t - 1 - (i + 1)] for i in range(kd)) if kd > 0 else 0
        d[t - p] = dY[t - 1] - U[t - p] - ect - gam
    if det in ("n", "co") and np.max(np.std(d, axis=0)) > 1e-6 * max(1.0, np.max(np.abs(Yv))):
        return None
    Uc = U - U.mean(0)
    oi, oo = [], []
    for _ in range(C["boot_runs"]):
        e = Uc[rng.integers(0, len(Uc), Tn - p)]
        Yb = np.empty_like(Yv)
        Yb[:p] = Yv[:p]
        for t in range(p, Tn):
            ect = al @ (be.T @ Yb[t - 1])
            gam = sum(ga[:, i * k:(i + 1) * k] @ (Yb[t - 1 - i] - Yb[t - 2 - i]) for i in range(kd)) if kd > 0 else 0
            Yb[t] = Yb[t - 1] + d[t - p] + ect + gam + e[t - p]
        try:
            ib = VECM(Yb, k_ar_diff=kd, coint_rank=r, deterministic=det).fit().irf(H)
            oi.append(ib.irfs)
            oo.append(ib.orth_irfs)
        except Exception:
            continue
    if len(oi) < 20:
        return None
    q = [(1 - C["nivel_ic"]) / 2 * 100, (1 + C["nivel_ic"]) / 2 * 100]
    return np.percentile(oi, q, axis=0), np.percentile(oo, q, axis=0)


# %% 5. EJECUCION: decision de estructura y modelo final --------------------------------
TREND_N = tendencia_var(ADF_N)
N = analizar(Y, TREND_N, ADF_N)
print("\nCriterios (niveles):", N["mejor"], "\n", N["ic"].round(4))
print("Comparacion de modelos (niveles):\n", N["comp"].round(4))
reg("p (niveles)", f"VAR({N['p']})", N["why_p"])
reg("Estabilidad (niveles)", f"max|autovalor| = {N['est']['max_eig']:.4f}", N["est"]["mensaje"])

D = JOH = RANGO = RANGO_LMAX = JOH_SENS = RANGO_JOH = None
DET = None
if N["ok"]:
    ESTRUCTURA, TIPO = "VAR en niveles", "VAR"
    reg("Estructura", ESTRUCTURA, "ADF rechaza raiz unitaria en todas y el VAR es claramente estable")
    if C["modelo"] == "VECM":
        reg("MODO VECM", "no aplica un VECM", "las series son estacionarias (ADF + estabilidad): se estima el VAR en niveles")
    if C["johansen_siempre"]:
        try:
            DET = int(C["johansen_det"]) if C["johansen_det"] is not None else 0
            JOH, RANGO, RANGO_LMAX = johansen_tabla(Y, N["p"], DET)
            JOH_SENS = sensibilidad_johansen(Y, sorted({int(N["p"]), 2}))
            reg("Johansen (confirmacion)", f"det_order={DET}, rango por traza r = {RANGO}",
                "r = K confirma que las series son estacionarias" if RANGO >= K else
                f"AVISO: ADF dice I(0) pero Johansen halla r = {RANGO} < K = {K}; revise la especificacion")
        except Exception as ex_:
            JOH = JOH_SENS = RANGO = None
            reg("Johansen (confirmacion)", "no se pudo calcular", str(ex_))
else:
    reg("Niveles descartados", "; ".join(N["razones"]), "regla del curso: raiz unitaria => diferenciar y probar cointegracion")
    Yd = Y.diff().dropna()
    ADF_D = adf_tabla(Yd, nivel=False)
    t_dif = Yd.mean() / (Yd.std(ddof=1) / np.sqrt(len(Yd)))
    TREND_D = "c" if (t_dif.abs() > 2).any() else "n"
    D = analizar(Yd, TREND_D, ADF_D, p_fijo=C["p_dif"] or (max(N["p"] - 1, 1) if C["perfil"] == "curso" else None))
    reg("Diferencias", f"VAR({D['p']}) tendencia='{TREND_D}'", D["why_p"] + "; ADF diferencias: " + ", ".join(
        f"{v}: {ADF_D.loc[v, 'lectura']}" for v in ADF_D.index))
    if C["johansen_det"] is not None:
        DET = int(C["johansen_det"])
        why_det = "fijado por el usuario" if "johansen_det" in AJUSTES else "perfil del curso (det_order=0, como el taller)"
    else:
        frac = float((ADF_N["espec"] == "ct").mean())
        DET = 1 if frac >= 0.5 else (-1 if (ADF_N["espec"] == "n").all() else 0)
        why_det = (f"{frac:.0%} de las series tienen deriva significativa => tendencia lineal en niveles" if DET == 1
                   else "series sin media cero ni deriva clara => constante sin restringir" if DET == 0 else "series centradas en cero")
    JOH, RANGO, RANGO_LMAX = johansen_tabla(Y, N["p"], DET)
    reg("Johansen", f"det_order={DET}, k_ar_diff={max(N['p'] - 1, 0)}", why_det)
    PS_SENS = sorted({int(N["p"]), *[int(v) for v in N["mejor"].values()], 2})
    JOH_SENS = sensibilidad_johansen(Y, PS_SENS)
    print("\nRobustez del rango de Johansen (traza) segun determinista y rezagos:\n", JOH_SENS)
    if C["rango"] is None:
        robusto = JOH_SENS.stack().nunique() == 1
        reg("Robustez Johansen", "rango estable en todas las especificaciones" if robusto else
            "AVISO: el rango CAMBIA segun determinista/rezagos",
            "ver tabla en la seccion 2; si cambia, justifique la especificacion o fije AJUSTES['rango'], ['johansen_det'], ['p']")
    if C["rango"] is not None:
        RANGO = int(C["rango"])
        reg("Rango", f"r = {RANGO}", "fijado por el usuario")
    else:
        reg("Rango", f"r = {RANGO} (traza)", f"secuencial al {ALFA:.0%}; lmax sugiere r = {RANGO_LMAX}" +
            ("" if RANGO == RANGO_LMAX else " (DIFIEREN: se usa traza, revise)"))
    if RANGO == 0:
        ESTRUCTURA, TIPO = "VAR en primeras diferencias", "VAR"
    elif RANGO < K:
        ESTRUCTURA, TIPO = f"VECM con r = {RANGO}", "VECM"
    else:
        ESTRUCTURA, TIPO = "VAR en niveles (Johansen: rango completo)", "VAR"
    reg("Estructura", ESTRUCTURA, "resultado de Johansen" if TIPO == "VECM" or RANGO == 0 else
        "r = K implica estacionariedad: ADF y Johansen discrepan, revise")
    if C["modelo"] == "VAR" and TIPO == "VECM":
        reg("MODO VAR", "se estima un VAR en diferencias aunque Johansen halla r = %d" % RANGO,
            "AVISO: un VAR en diferencias omite la relacion de largo plazo; para modelarla use VECM_auto.py")
        RANGO_JOH = RANGO          # rango REAL de Johansen (para el informe)
        RANGO, ESTRUCTURA, TIPO = 0, "VAR en primeras diferencias (modo VAR)", "VAR"

DET_VECM = C["vecm_det"] or {-1: "n", 0: "co", 1: "colo"}.get(DET)
if TIPO == "VECM":
    kd = max(N["p"] - 1, 0)
    RES = VECM(Y.reset_index(drop=True), k_ar_diff=kd, coint_rank=RANGO, deterministic=DET_VECM).fit()
    MODELO_P, X_MOD, TREND_MOD = N["p"], Y, None
    reg("Modelo final", f"VECM(k_ar_diff={kd}, r={RANGO}, det='{DET_VECM}')",
        "perfil curso: det_order=0 con deterministic='ci' (constante en la relacion de cointegracion); cambie con AJUSTES['vecm_det']")
elif D is not None and RANGO == 0:
    RES, MODELO_P, X_MOD, TREND_MOD = D["res"], D["p"], D["X"], D["trend"]
    reg("Modelo final", f"VAR({MODELO_P}) en diferencias", "sin cointegracion" if RANGO_JOH is None else "modo VAR: se omite la cointegracion que halla Johansen")
else:
    RES, MODELO_P, X_MOD, TREND_MOD = N["res"], N["p"], Y, N["trend"]
    reg("Modelo final", f"VAR({MODELO_P}) en niveles", "p y estructura segun las reglas anteriores")
VAR_REP = np.asarray(RES.var_rep) if TIPO == "VECM" else np.asarray(RES.coefs)
EST_MOD = estabilidad(VAR_REP)
if TIPO == "VECM":
    EST_MOD["mensaje"] = f"{K - RANGO} raices unitarias esperadas (K - r); el resto de raices debe quedar dentro del circulo"
DIAG = diagnosticos(RES, nombres)
reg("Diagnosticos", f"Portmanteau p = {DIAG['white']['p']:.4f}; JB p = {DIAG['norm']['p']:.4f}; ARCH min p = {DIAG['arch']['p_LM'].min():.4f}",
    ("sin" if DIAG["white"]["p"] >= ALFA else "CON") + f" autocorrelacion residual (portmanteau al {ALFA:.0%}); detalle en la seccion de supuestos")

_av = []
if DIAG["norm"]["p"] < ALFA:
    _av.append("residuos no normales (JB)")
if DIAG["arch"]["p_LM"].min() < ALFA:
    _av.append("heterocedasticidad condicional (ARCH)")
if _av:
    reg("AVISO supuestos", "; ".join(_av), "los IC de pronostico/IRF y los p-valores son aproximados; considere dummies de atipicos, cambios de regimen o modelos GARCH")

H = C["pasos_pron"]
if TIPO == "VECM":
    pt, lo, hi = RES.predict(steps=H, alpha=1 - C["nivel_ic"])
else:
    pt, lo, hi = RES.forecast_interval(np.asarray(RES.endog)[-RES.k_ar:], steps=H, alpha=1 - C["nivel_ic"])
FC = {}
for k_, v in enumerate(nombres):
    tb = pd.DataFrame({"fcst": pt[:, k_], "inf": lo[:, k_], "sup": hi[:, k_]}, index=indice_futuro(H))
    if TIPO == "VAR" and D is not None and RANGO == 0:
        tb["nivel"] = Y[v].iloc[-1] + tb["fcst"].cumsum()
        if LOGS[v]:
            tb["nivel_orig"] = np.exp(tb["nivel"])
    elif LOGS[v]:
        tb["fcst_orig"] = np.exp(tb["fcst"])
    FC[v] = tb

HI = C["pasos_irf"] - 1
IRA = RES.irf(HI)
IRF, OIRF = IRA.irfs, IRA.orth_irfs
if TIPO == "VECM":
    B = bandas_irf_vecm(RES, Y, max(N["p"] - 1, 0), RANGO, DET_VECM, HI)
else:
    B = bandas_irf_var(RES, TREND_MOD, HI)
if B is None:
    BI = BO = None
    reg("Bandas IRF", "no calculadas", "el bootstrap del VECM no se pudo validar para este caso")
else:
    BI, BO = B
    reg("Bandas IRF", f"bootstrap de residuales, {C['boot_runs']} replicas, {int(C['nivel_ic'] * 100)}%", "percentiles")
reg("Pronostico", f"{H} pasos", "ver seccion 6")
print("\n>>> MODELO FINAL:", ESTRUCTURA)

# %% 6. Informe HTML (+ PDF) ---------------------------------------------------------------


def dec_txt(p):
    return "rechaza H0" if p < ALFA else "no rechaza H0"


def tbl(df, nd=4, idx=True):
    return df.to_html(classes="tbl", border=0, index=idx, float_format=lambda v: _f(v, nd), na_rep="")


def tbl_ic(t, mejor):
    h = "<table class='tbl'><thead><tr><th>p</th>" + "".join(f"<th>{c.upper()}</th>" for c in t) + "</tr></thead><tbody>"
    for p, row in t.iterrows():
        h += f"<tr><th>{p}</th>" + "".join(f"<td class='{'best' if mejor[c] == p else ''}'>{_f(v, 5)}</td>" for c, v in row.items()) + "</tr>"
    return h + "</tbody></table>"


def svg(fig):
    b = io.StringIO()
    fig.savefig(b, format="svg", bbox_inches="tight")
    plt.close(fig)
    s = b.getvalue()
    return s[s.find("<svg"):]


def grid_irf(irf, bands, titulo):
    k, h = len(nombres), irf.shape[0]
    fig, ax = plt.subplots(k, k, figsize=(3.3 * k + .6, 2.5 * k + .5), sharex=True, squeeze=False)
    x = np.arange(h)
    for i in range(k):
        for j in range(k):
            a = ax[i, j]
            if bands is not None:
                a.fill_between(x, bands[0][:h, i, j], bands[1][:h, i, j], color="#0f766e", alpha=.18, lw=0)
            a.plot(x, irf[:, i, j], color="#0f766e", lw=1.8)
            a.axhline(0, color="#64748b", lw=.7)
            a.set_title(f"{nombres[i]} <- choque en {nombres[j]}", fontsize=9)
            a.grid(alpha=.25)
    fig.suptitle(titulo, fontsize=10)
    fig.tight_layout()
    return fig


def texto_irf(irf):
    h, k = irf.shape[0], len(nombres)
    thr = 0.01 * float(np.max(np.abs(irf)))
    out = []
    for j in range(k):
        for i in range(k):
            r = irf[:, i, j]
            pk = int(np.argmax(np.abs(r)))
            hl = next((s for s in range(pk, h) if abs(r[s]) <= abs(r[pk]) / 2), None)
            ds = next((s for s in range(h) if np.all(np.abs(r[s:]) < thr)), None)
            if pk > 0 and abs(r[-1]) >= 0.9 * abs(r[pk]):
                out.append(f"<li><b>{nombres[j]} &rarr; {nombres[i]}</b>: efecto en t=0 {_f(r[0])}; la respuesta no se disipa: "
                           f"llega a {_f(r[-1])} al final del horizonte (efecto permanente, propio de variables I(1)).</li>")
                continue
            out.append(f"<li><b>{nombres[j]} &rarr; {nombres[i]}</b>: efecto en t=0 {_f(r[0])}; pico {'positivo' if r[pk] > 0 else 'negativo'} "
                       f"{_f(r[pk])} en el paso {pk}; vida media {('paso ' + str(hl)) if hl is not None else 'mas alla del horizonte'}; "
                       f"|respuesta| &lt; 1% del maximo desde el paso {ds if ds is not None else 'mas alla del horizonte'}.</li>")
    return "<ul>" + "".join(out) + "</ul>"


def figuras():
    F = {}
    f, ax = plt.subplots(1, K, figsize=(3.6 * K, 2.8), squeeze=False)
    for k_, v in enumerate(nombres):
        x = DATOS_ORIG.index.to_timestamp() if isinstance(DATOS_ORIG.index, pd.PeriodIndex) else DATOS_ORIG.index
        ax[0, k_].plot(x, DATOS_ORIG[v], lw=.9, color="#0f766e")
        ax[0, k_].set_title(v, fontsize=9)
        ax[0, k_].grid(alpha=.25)
        ax[0, k_].tick_params(labelsize=7)
    f.tight_layout()
    F["series"] = svg(f)
    f, a = plt.subplots(figsize=(3.8, 3.8))
    t = np.linspace(0, 2 * np.pi, 200)
    a.plot(np.cos(t), np.sin(t), color="#64748b", lw=1)
    a.scatter(N["est"]["eig"].real, N["est"]["eig"].imag, color="#b45309", label="niveles (VAR)", zorder=3)
    if D is not None:
        a.scatter(D["est"]["eig"].real, D["est"]["eig"].imag, color="#1d4ed8", label="diferencias", zorder=3)
    if TIPO == "VECM":
        a.scatter(EST_MOD["eig"].real, EST_MOD["eig"].imag, marker="x", color="#0f766e", label="VECM (forma VAR)", zorder=4)
    a.set_aspect("equal")
    a.grid(alpha=.25)
    a.legend(fontsize=8, loc="lower left")
    F["roots"] = svg(f)
    f, ax = plt.subplots(1, K, figsize=(3.8 * K, 3), squeeze=False)
    for k_, v in enumerate(nombres):
        tb = FC[v]
        ix = tb.index.to_timestamp() if isinstance(tb.index, pd.PeriodIndex) else tb.index
        ax[0, k_].plot(ix, tb["fcst"], color="#0f766e")
        ax[0, k_].fill_between(ix, tb["inf"], tb["sup"], color="#0f766e", alpha=.18, lw=0)
        ax[0, k_].set_title(f"Pronostico {v}", fontsize=9)
        ax[0, k_].grid(alpha=.25)
        ax[0, k_].tick_params(axis="x", labelrotation=30, labelsize=7)
    f.tight_layout()
    F["fc"] = svg(f)
    F["irf"] = svg(grid_irf(IRF, BI, "IRF (no ortogonalizada)"))
    F["oirf"] = svg(grid_irf(OIRF, BO, "OIRF (Cholesky: " + " > ".join(nombres) + ")"))
    return F


def guardar():
    ADF_N.to_csv(SAL / "adf_niveles.csv")
    N["ic"].to_csv(SAL / "criterios_niveles.csv")
    N["comp"].to_csv(SAL / "comparacion_modelos_niveles.csv")
    if D is not None:
        D["adf"].to_csv(SAL / "adf_diferencias.csv")
        D["ic"].to_csv(SAL / "criterios_diferencias.csv")
        D["comp"].to_csv(SAL / "comparacion_modelos_diferencias.csv")
        JOH.to_csv(SAL / "johansen.csv")
    for v, t in FC.items():
        t.to_csv(SAL / f"pronostico_{v}.csv")
    DIAG["arch"].to_csv(SAL / "arch.csv")
    DIAG["norm_uni"].to_csv(SAL / "normalidad.csv")
    np.save(SAL / "irf.npy", IRF)
    np.save(SAL / "oirf.npy", OIRF)
    pd.DataFrame(LOG, columns=["paso", "decision", "motivo"]).to_csv(SAL / "bitacora_decisiones.csv", index=False)
    resumen = dict(archivo=RUTA.name, hoja=HOJA, variables=nombres, orden_cholesky=nombres, estructura=ESTRUCTURA, p=MODELO_P,
                   rango=RANGO, johansen_det=DET, logs=LOGS, max_eig_modelo=EST_MOD["max_eig"],
                   diagnosticos={k: (v if not isinstance(v, pd.DataFrame) else v.to_dict()) for k, v in DIAG.items()},
                   config={k: v for k, v in C.items()})
    (SAL / "resultados.json").write_text(json.dumps(resumen, indent=1, default=str), encoding="utf-8")


def construir_html(F, ruta):
    secs, it = [], []

    def item(t, body): it.append(f'<div class="item"><h3>{t}</h3>{body}</div>')
    def ans(t): return f'<div class="ans">{t}</div>'
    def fig(s, cap): return f"<figure>{s}<figcaption>{cap}</figcaption></figure>"
    def sec(t, i): secs.append((t, i, "".join(it))); it.clear()

    # 0 sondeo
    item("Base de datos", html_sondeo(RAW, PERFIL, USADAS, IDX, FREQ, HOJA, RUTA, NOTA_T, tabla_id="tbl-datos-rep"))
    sec("0. Sondeo de la base", "s0")
    # 1 decision
    bad = "" if (N["ok"]) else "bad"
    dec = (f"<div class='dec {bad}'><b>Estructura recomendada: {ESTRUCTURA}</b><br>Modelo final: "
           f"{('VECM (k_ar_diff = ' + str(MODELO_P - 1) + ', r = ' + str(RANGO) + ')') if TIPO == 'VECM' else 'VAR(' + str(MODELO_P) + ')'}; orden de Cholesky: {' &gt; '.join(nombres)}.</div>")
    bit = ("<table class='tbl'><thead><tr><th>paso</th><th>decision</th><th>motivo</th></tr></thead><tbody>" +
           "".join(f"<tr><th>{_html.escape(a)}</th><td class='l'>{_html.escape(b)}</td><td class='l'>{_html.escape(c)}</td></tr>" for a, b, c in LOG) +
           "</tbody></table>")
    item("Decision", dec + fig(F["roots"], "Autovalores de la matriz companera; dentro del circulo => estacionario.") )
    item("Bitacora de decisiones automaticas (auditable)", bit)
    sec("1. Decision automatica del modelo", "s1")
    # 2 datos / raices
    item("Series originales", fig(F["series"], "Series en sus unidades originales."))
    item("ADF + KPSS (niveles)", tbl(ADF_N) + "<p>H0 ADF: raiz unitaria. H0 KPSS: estacionariedad. espec: n sin determinista, c constante, ct constante y tendencia (elegida segun media y deriva de la serie). "
         "Rezagos por AIC.</p>")
    if D is not None:
        item("ADF + KPSS (primeras diferencias)", tbl(D["adf"]))
    if JOH is not None:
        item("Robustez del rango (traza al 5%)", tbl(JOH_SENS) + "<p>Filas: determinista de Johansen (-1 ninguno, 0 constante, 1 tendencia). Columnas: rezagos del VAR en niveles. "
             "Si el rango cambia entre celdas, la conclusion de cointegracion depende de la especificacion.</p>")
        item("Johansen (niveles)", tbl(JOH) + ans(f"det_order = {DET}; rango por traza r = {RANGO}; por lmax r = {RANGO_LMAX}. Rango usado: <b>{RANGO}</b>."))
    sec("2. Raices unitarias y cointegracion", "s2")
    # 3 rezagos
    def bloque(A, nm):
        mj = ", ".join(f"{c.upper()}: p = {v}" for c, v in A["mejor"].items())
        t = A["comp"].copy()
        t["estable"] = np.where(t["estable"], "si", "no")
        t["blanco"] = np.where(t["blanco"], "si", "no")
        t.index = [f"{p} &#9664; elegido" if p == A["p"] else str(p) for p in t.index]
        return (ans(f"<b>{nm}</b>: minimos por criterio: {mj}. Elegido <b>VAR({A['p']})</b>: {A['why_p']}.") + tbl_ic(A["ic"], A["mejor"]) + tbl(t) +
                "<p>estable: max|autovalor| bajo el umbral; blanco: el portmanteau no rechaza. ARCH y JB son informativos.</p>")
    item("Criterios de informacion y comparacion (niveles)", bloque(N, "Niveles"))
    if D is not None:
        item("Criterios de informacion y comparacion (diferencias)", bloque(D, "Primeras diferencias"))
    sec("3. Seleccion de rezagos", "s3")
    # 4 estimacion
    if TIPO == "VECM":
        al = pd.DataFrame(np.asarray(RES.alpha), index=nombres, columns=[f"alfa{i + 1}" for i in range(RANGO)])
        be = pd.DataFrame(np.asarray(RES.beta), index=nombres, columns=[f"beta{i + 1}" for i in range(RANGO)])
        pi = pd.DataFrame(np.asarray(RES.alpha) @ np.asarray(RES.beta).T, index=nombres, columns=nombres)
        item(f"VECM estimado (r = {RANGO}, k_ar_diff = {MODELO_P - 1}, det = '{DET_VECM}')",
             "<p><b>&alpha;</b> (velocidades de ajuste)</p>" + tbl(al) + "<p><b>&beta;</b> (vectores de cointegracion, normalizados)</p>" + tbl(be) +
             "<p><b>&Pi; = &alpha;&beta;'</b></p>" + tbl(pi) +
             ans("Relacion de largo plazo: &beta;'Y<sub>t</sub> es estacionaria. Una variable corrige el desequilibrio cuando el signo de &alpha;<sub>i</sub> es opuesto al de su &beta;<sub>i</sub> (&alpha;<sub>i</sub>&middot;&beta;<sub>i</sub> &lt; 0) y &alpha;<sub>i</sub> es significativo."
                 + ((" Aqui corrigen: <b>" + ", ".join(v for i_, v in enumerate(nombres) if al.iloc[i_, 0] * be.iloc[i_, 0] < 0 and abs(be.iloc[i_, 0]) > 1e-3) + "</b>.") if RANGO == 1 else "")))
        if MODELO_P > 1:
            item("Dinamica de corto plazo &Gamma;", tbl(pd.DataFrame(np.asarray(RES.gamma), index=nombres,
                 columns=[f"{v}(-{l + 1})" for l in range(MODELO_P - 1) for v in nombres])))
        try:
            item("Resumen statsmodels", f"<pre>{_html.escape(str(RES.summary()))}</pre>")
        except Exception:
            pass
    else:
        ca = "".join(f"<p><b>A<sub>{k_ + 1}</sub></b></p>" + tbl(pd.DataFrame(np.asarray(RES.coefs)[k_], index=nombres, columns=nombres)) for k_ in range(RES.k_ar))
        item(f"VAR({RES.k_ar}) estimado" + (" en primeras diferencias" if D is not None else " en niveles"), ca + "<p>Filas = ecuaciones; columnas = regresores rezagados.</p>")
        eq = "".join(f"<p><b>Ecuacion {v}</b></p>" + tbl(pd.DataFrame({"coef": RES.params[v], "ee": RES.stderr[v], "t": RES.tvalues[v], "P>|t|": RES.pvalues[v]})) for v in nombres)
        item("Coeficientes por ecuacion", eq + f"<pre>{_html.escape(str(RES.summary()))}</pre>")
    item("Estabilidad del modelo final", ans(f"max |autovalor| = {_f(EST_MOD['max_eig'], 4)} &rarr; {EST_MOD['mensaje']}."))
    sec("4. Estimacion", "s4")
    # 5 supuestos
    w, nm_ = DIAG["white"], DIAG["norm"]
    item("Autocorrelacion (portmanteau)", ans(f"Q = {_f(w['stat'], 3)}, gl = {w['df']}, rezagos = {w['nlags']}, p = {_f(w['p'])} &rarr; {dec_txt(w['p'])} "
         f"({'no hay' if w['p'] >= ALFA else 'hay'} autocorrelacion residual)."))
    item("Heterocedasticidad condicional (ARCH-LM)", tbl(DIAG["arch"]) + ans("; ".join(f"{v}: p = {_f(DIAG['arch'].loc[v, 'p_LM'])} &rarr; {dec_txt(DIAG['arch'].loc[v, 'p_LM'])}" for v in nombres)))
    item("Normalidad (Jarque-Bera)", ans(f"JB multivariado = {_f(nm_['stat'], 3)} (gl = {nm_['df']}), p = {_f(nm_['p'])} &rarr; {dec_txt(nm_['p'])}.") + tbl(DIAG["norm_uni"]))
    sec("5. Verificacion de supuestos", "s5")
    # 6 pronostico
    extra = ""
    if TIPO == "VAR" and D is not None and RANGO == 0:
        extra = "<div class='warn'>El pronostico de 'fcst' es de las <b>diferencias</b>; 'nivel' lo reconstruye acumulando desde el ultimo dato (sin intervalo).</div>"
    if any(LOGS.values()):
        extra += "<div class='warn'>Las variables con log se pronostican en logaritmos; 'nivel_orig'/'fcst_orig' devuelve la unidad original.</div>"
    item(f"Pronostico a {C['pasos_pron']} pasos (IC {int(C['nivel_ic'] * 100)}%)", "".join(f"<p><b>{v}</b></p>" + tbl(t) for v, t in FC.items()) +
         fig(F["fc"], "Pronostico puntual e intervalo.") + extra)
    sec("6. Pronostico", "s6")
    # 7 IRF
    cols = [f"{v}<-{u}" for u in nombres for v in nombres]
    def tab(m):
        d_ = pd.DataFrame([m[h_].T.reshape(-1) for h_ in range(min(10, m.shape[0]))], columns=cols)
        d_.index.name = "paso"
        return tbl(d_)
    P_ = np.linalg.cholesky(np.asarray(RES.sigma_u))
    item("IRF no ortogonalizada (10 primeros pasos)", tab(IRF))
    item("IRF ortogonalizada (Cholesky)", tab(OIRF) + tbl(pd.DataFrame(P_, index=nombres, columns=nombres)) +
         ans("Choques de una desviacion estandar no correlacionados: la variable puesta primero afecta a todas en t; la ultima solo a si misma en t."))
    item("Graficos IRF / OIRF", fig(F["irf"], "IRF; filas = respuestas, columnas = impulsos.") + fig(F["oirf"], "OIRF (Cholesky).") +
         (f"<p>Bandas: bootstrap de residuales, {C['boot_runs']} replicas, semilla {C['semilla']}, percentil {int(C['nivel_ic'] * 100)}%.</p>" if BI is not None else "<p>Sin bandas.</p>"))
    item("Lectura de la IRF", texto_irf(IRF) + (ans("En un VECM los choques pueden tener efecto permanente sobre las variables I(1).") if TIPO == "VECM" else ""))
    item("Lectura de la OIRF", texto_irf(OIRF))
    sec("7. Funciones impulso-respuesta", "s7")

    toc = "".join(f'<a href="#{i}">{t}</a>' for t, i, _ in secs)
    body = "".join(f'<h2 id="{i}">{t}</h2>{c}' for t, i, c in secs)
    kp = [("Estructura", ESTRUCTURA.split(" (")[0]), ("Modelo", f"VECM r={RANGO}" if TIPO == "VECM" else f"VAR({MODELO_P})"), ("Rango r", "n/a" if RANGO is None else str(RANGO)),
          ("max|autovalor|", _f(EST_MOD["max_eig"], 3)), ("Portmanteau p", _f(w["p"], 3)), ("JB p", _f(nm_["p"], 3))]
    kpis = "".join(f'<div class="kpi"><b>{b}</b><span>{a}</span></div>' for a, b in kp)
    refs = ("<h2>Referencias</h2><ul><li>Dickey, D. A., &amp; Fuller, W. A. (1979). <i>JASA, 74</i>(366), 427&ndash;431.</li>"
            "<li>Enders, W. (2014). <i>Applied econometric time series</i> (4th ed.). Wiley.</li>"
            "<li>Johansen, S. (1991). <i>Econometrica, 59</i>(6), 1551&ndash;1580.</li>"
            "<li>L&uuml;tkepohl, H. (2005). <i>New introduction to multiple time series analysis</i>. Springer.</li>"
            "<li>Seabold, S., &amp; Perktold, J. (2010). Statsmodels. <i>Proc. 9th Python in Science Conf.</i>, 92&ndash;96.</li></ul>")
    doc = (f"<!doctype html><html lang='es'><head><meta charset='utf-8'><meta name='viewport' content='width=device-width,initial-scale=1'>"
           f"<title>Informe VAR/VECM</title><style>{CSS}</style></head><body><header><h1>Informe automatico VAR / VECM</h1>"
           f"<p>Econometr&iacute;a II &middot; {_html.escape(RUTA.name)} &middot; &alpha; = {ALFA} &middot; generado {_dt.date.today():%Y-%m-%d}</p>"
           f"<p>Variables [{', '.join(nombres)}]</p></header><main><div class='kpis'>{kpis}</div><nav class='toc'><b>Contenido:</b> {toc}</nav>{body}{refs}"
           f"<footer>Todas las pruebas con &alpha; = {ALFA}. Figuras vectoriales SVG.</footer></main>"
           f"<button class='btn' onclick='window.print()'>&#128438; Descargar PDF</button></body></html>")
    ruta.write_text(doc, encoding="utf-8")
    return ruta


def generar_pdf(ruta_html, ruta_pdf):
    try:
        from playwright.sync_api import sync_playwright
        with sync_playwright() as pw:
            b = pw.chromium.launch()
            pg = b.new_page()
            pg.goto(ruta_html.resolve().as_uri())
            pg.pdf(path=str(ruta_pdf), format="A4", print_background=True, prefer_css_page_size=True)
            b.close()
        return "playwright"
    except Exception:
        pass
    try:
        from weasyprint import HTML
        HTML(filename=str(ruta_html)).write_pdf(str(ruta_pdf))
        return "weasyprint"
    except Exception:
        print("PDF automatico no disponible: abra el HTML y pulse 'Descargar PDF' (Imprimir > Guardar como PDF).")
        return None



def _L(a):
    return np.asarray(a, dtype=float).tolist()


def exportar_json():
    """Escribe resultados/auto_<base>.json con el esquema que consume generar_informe.py (cada bloque aislado)."""
    import statsmodels
    _m = re.search(r"escenario[_ ]?(\d)", RUTA.stem, re.I)
    R = {"esc": int(_m.group(1)) if _m else RUTA.stem, "etiqueta": f"E{_m.group(1)}" if _m else re.sub(r"\W+", "_", RUTA.stem)[:12], "titulo": f"{RUTA.name} - {ESTRUCTURA}",
         "motor": f"statsmodels {statsmodels.__version__} (pipeline_auto)", "errores": [], "vars": nombres,
         "periodos": [str(i) for i in Y.index], "series": {v: _L(Y[v]) for v in nombres},
         "notas": ("Series en logaritmos: " + ", ".join(v for v in nombres if LOGS[v]) + ".") if any(LOGS.values()) else ""}
    sobre = "diferencias" if (D is not None and RANGO == 0) else "niveles"

    def seg(nombre, fn):
        try:
            R[nombre] = fn()
        except Exception as ex_:
            R[nombre] = None
            R["errores"].append(f"{nombre}: {type(ex_).__name__}: {ex_}")

    def diag(d):
        return {"portmanteau": {k: d["white"][k] for k in ("stat", "p", "df", "nlags")},
                "arch": {v: {"lm": float(d["arch"].loc[v, "LM"]), "p": float(d["arch"].loc[v, "p_LM"])} for v in nombres},
                "normalidad": dict(d["norm"]),
                "jb": {v: {"stat": float(d["norm_uni"].loc[v, "JB"]), "p": float(d["norm_uni"].loc[v, "p_value"])} for v in nombres}}

    def modelo():
        base = {"sobre": sobre, "A": [_L(a) for a in VAR_REP], "sigma_u": _L(RES.sigma_u), "nobs": int(RES.nobs),
                "estable": bool(EST_MOD["estable"]), "raiz_max_modulo": float(EST_MOD["max_eig"])}
        if TIPO == "VECM":
            al, be, kd = np.asarray(RES.alpha), np.asarray(RES.beta), MODELO_P - 1
            cc = (np.asarray(RES.const_coint).ravel().tolist() + [None] * RANGO)[:RANGO]
            rho = 1 + float(be[:, 0] @ al[:, 0])
            base.update(tipo=f"VECM(k_ar_diff={kd}, r={RANGO})", rango=int(RANGO), beta={v: _L(be[i]) for i, v in enumerate(nombres)},
                        alpha={v: _L(al[i]) for i, v in enumerate(nombres)}, const_coint=cc,
                        gamma=[_L(np.asarray(RES.gamma)[:, i * K:(i + 1) * K]) for i in range(kd)], rho_desequilibrio=rho,
                        vida_media=float(np.log(.5) / np.log(rho)) if (RANGO == 1 and 0 < rho < 1) else None)
        else:
            base.update(tipo=f"VAR({MODELO_P}) " + ("en diferencias" if sobre == "diferencias" else "en niveles"),
                        rango=0 if sobre == "diferencias" else K)
        return base

    def pron():
        tb = [FC[v] for v in nombres]
        col = lambda c: np.column_stack([t[c] for t in tb]).tolist()
        return {"h": len(tb[0]), "ci": C["nivel_ic"], "sobre": sobre, "fecha": [str(i) for i in tb[0].index],
                "pron": col("fcst"), "inf": col("inf"), "sup": col("sup"), "ult": _L(X_MOD.iloc[-1]),
                "hist": _L(X_MOD.iloc[-40:].to_numpy()), "hist_fecha": [str(i) for i in X_MOD.index[-40:]]}

    def irf():
        h = IRF.shape[0]

        def pack(M, Bd):
            return {f"{imp}>{resp}": {"irf": _L(M[:, j, i]), "lo": _L(Bd[0][:h, j, i] if Bd is not None else M[:, j, i]),
                                      "hi": _L(Bd[1][:h, j, i] if Bd is not None else M[:, j, i])}
                    for i, imp in enumerate(nombres) for j, resp in enumerate(nombres)}
        return {"pasos": list(range(h)), "no_ortog": pack(IRF, BI), "ortog": pack(OIRF, BO),
                "phi": [_L(m) for m in IRF[:11]], "psi": [_L(m) for m in OIRF[:11]], "reps": C["boot_runs"]}

    seg("adf", lambda: {v: {"stat": float(ADF_N.loc[v, "t"]), "p": float(ADF_N.loc[v, "p_value"]), "lags": int(ADF_N.loc[v, "rezagos"]),
                            "nobs": None, "crit": {"5%": float(ADF_N.loc[v, "crit_5"])}} for v in nombres})
    seg("rezagos", lambda: {"filas": [{"p": int(p), **{c: float(N["ic"].loc[p, c]) for c in N["ic"].columns}} for p in N["ic"].index],
                            "seleccionados": {k: int(v) for k, v in N["mejor"].items()}, "elegido": int(N["p"])})
    seg("var_niveles", lambda: {"p": int(N["p"]), "coefs": [_L(a) for a in N["res"].coefs], "sigma_u": _L(N["res"].sigma_u),
                                "nobs": int(N["res"].nobs), "estable": bool(N["est"]["estable"]), "raiz_max_modulo": float(N["est"]["max_eig"])})
    seg("diag_var", lambda: diag(diagnosticos(N["res"], nombres)))
    if JOH is not None:
        seg("johansen", lambda: {"traza": {"stat": _L(JOH["traza"]), "cv": [[None, float(c), None] for c in JOH["crit5_traza"]]},
                                 "eigen": {"stat": _L(JOH["lmax"]), "cv": [[None, float(c), None] for c in JOH["crit5_lmax"]]},
                                 "autovalores": None, "rango": int(RANGO if RANGO_JOH is None else RANGO_JOH), "nobs": None, "det_order": DET})
    seg("modelo", modelo)
    seg("diag_modelo", lambda: diag(DIAG))
    seg("pronostico", pron)
    seg("irf", irf)
    ruta = BASE / "resultados"
    ruta.mkdir(exist_ok=True)
    sufijo = "" if C["modelo"] == "auto" else "_" + C["modelo"]
    if C["modelo"] != "auto":
        R["esc"], R["etiqueta"] = RUTA.stem + sufijo, C["modelo"]
    destino = ruta / f"auto_{re.sub(chr(92) + 'W+', '_', RUTA.stem)}{sufijo}.json"
    destino.write_text(json.dumps(R, ensure_ascii=False, default=lambda o: o.tolist() if hasattr(o, "tolist") else str(o)), encoding="utf-8")
    print("JSON para el informe poster:", destino, f"(secciones con error: {len(R['errores'])})" if R["errores"] else "")
    for e_ in R["errores"]:
        print("   -", e_)
    return destino


FIGS = figuras()
guardar()
exportar_json()
HTML_OUT = construir_html(FIGS, SAL / "informe_auto.html")
print("\nInforme HTML:", HTML_OUT)
if C["generar_pdf"]:
    print("PDF generado con:", generar_pdf(HTML_OUT, SAL / "informe_auto.pdf"))
if C["poster"] and (BASE / "generar_informe.py").exists():
    subprocess.run([sys.executable, str(BASE / "generar_informe.py"), "--solo", "auto"], cwd=str(BASE))
# %%