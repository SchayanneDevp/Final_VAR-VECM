#!/usr/bin/env python3
"""Genera el informe HTML (poster de 3 columnas / informe vertical) y, opcionalmente, el PDF y un Excel con
todos los calculos del taller de VECM.

Uso (desde la carpeta python/ del taller):
    python generar_informe.py                 # usa resultados/escenario{1,2,3}.json si existen
    python generar_informe.py --pdf           # ademas crea informe_taller_VECM.pdf (requiere playwright)
    python generar_informe.py --motor numpy   # recalcula todo con el motor de respaldo (sin statsmodels)

Flujo normal: correr los 3 scripts del taller (cada uno guarda su JSON al final) y luego este archivo.
El informe ES autocontenido: se abre con doble clic y su boton "Descargar PDF" imprime en vectorial (nitido).
"""
from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path

import numpy as np

AQUI = Path(__file__).resolve().parent
ESPERADO = {1: 0, 2: 1, 3: 2}                      # rango de Johansen esperado en cada escenario del taller
EXCEL = {1: "datos_escenario1_I(1)_no_cointegradas.xlsx", 2: "datos_escenario2_I(1)_cointegradas.xlsx",
         3: "datos_escenario3_I(0).xlsx"}
TITULOS = {1: "Escenario 1 · Series I(1) no cointegradas", 2: "Escenario 2 · Series I(1) cointegradas",
           3: "Escenario 3 · Series I(0)"}


# ----------------------------------------------------------------------------- carga de resultados
def normalizar(R: dict) -> dict:
    """Acepta el esquema v1 (solo x, y) y el v2 (K variables en R['vars'])."""
    if "vars" not in R: R["vars"] = ["x", "y"]
    if "series" not in R: R["series"] = {"x": R["x"], "y": R["y"]}
    M = R.get("modelo")
    if M and "beta" in M:
        for k in ("beta", "alpha"):
            M[k] = {v: (a if isinstance(a, list) else [a]) for v, a in M[k].items()}
        if not isinstance(M.get("const_coint"), list): M["const_coint"] = [M.get("const_coint")]
    R.setdefault("etiqueta", f"E{R['esc']}" if isinstance(R.get("esc"), int) else str(R.get("esc", ""))[:10])
    return R


def cargar(motor: str, solo: str) -> dict:
    res = {}
    carpeta = AQUI / "resultados"
    if motor != "numpy":
        pats = {"escenarios": ["escenario*.json"], "auto": ["auto_*.json"], "todo": ["escenario*.json", "auto_*.json"]}[solo]
        for pat in pats:
            for ruta in sorted(carpeta.glob(pat)):
                res[ruta.stem] = normalizar(json.loads(ruta.read_text(encoding="utf-8")))
    if not res:
        import pandas as pd
        import motor_numpy
        for n in (1, 2, 3):
            print(f"[aviso] escenario {n}: sin JSON; se calcula con el motor de respaldo (numpy).")
            res[f"escenario{n}"] = normalizar(motor_numpy.analizar(pd.read_excel(AQUI / EXCEL[n]), n, TITULOS[n]))
    vistos = {}
    for R in res.values():                           # etiquetas unicas (chips del poster y hojas del Excel)
        e0 = R["etiqueta"]; vistos[e0] = vistos.get(e0, 0) + 1
        if vistos[e0] > 1: R["etiqueta"] = f"{e0[:9]}_{vistos[e0]}"
    return res


# ----------------------------------------------------------------------------- interpretacion automatica
def f(v, d=3):
    return "n/d" if v is None else f"{v:.{d}f}".replace("-", "−")


def pv(p):
    return "&lt;0.001" if p < 0.001 else f(p, 3)


def ok(c, si="cumple", no="revisar"):
    return f'<span class="b {"ok" if c else "al"}">{"✓" if c else "!"} {si if c else no}</span>'


def disipa(v, tol=0.05):
    v = np.abs(np.asarray(v)); m = v.max()
    if m < 1e-9: return "nula"
    fuera = np.where(v > tol * m)[0]
    h = int(fuera.max()) + 1 if len(fuera) else 0
    return f"se disipa hacia h≈{h}" if h < len(v) - 1 else "persiste"


def resumen_irf(src, vars_):
    """Lista completa si K<=2; si no, un resumen (cuantas se disipan y cuales persisten)."""
    est = {k: disipa(v["irf"]) for k, v in src.items()}
    if len(vars_) <= 2:
        return " ".join(f"{k.replace('>', '→')}: {t};" for k, t in est.items())
    pers = [k.replace(">", "→") for k, t in est.items() if t == "persiste"]
    return (f"{len(est) - len(pers)} de {len(est)} respuestas se disipan en el horizonte; persisten: "
            f"{', '.join(pers[:8]) + ('…' if len(pers) > 8 else '') if pers else 'ninguna'}.")


def interpretar(R: dict) -> dict:
    """Lecturas en lenguaje natural, ligadas a las cifras. Una cadena HTML por paso del taller."""
    T, e, V = {}, R["esc"], R["vars"]
    A, M, J = R.get("adf"), R.get("modelo"), R.get("johansen")
    S = R["series"]; n = len(S[V[0]])
    T[1] = (f"{n} observaciones ({R['periodos'][0]}–{R['periodos'][-1]}). Medias: " + ", ".join(f"{v} = <b>{f(np.mean(S[v]), 2)}</b>" for v in V)
            + ". " + ("Correlación en niveles <b>" + f(float(np.corrcoef(S[V[0]], S[V[1]])[0, 1]), 2) + "</b>." if len(V) >= 2 else ""))
    if R.get("notas"): T[1] += " " + R["notas"]
    if A:
        rej = [v for v in V if A[v]["p"] < .05]
        T[2] = ("Dickey-Fuller aumentada, H₀: raíz unitaria. "
                + ("<b>Se rechaza H₀ en todas las series</b> (p&lt;0.05): estacionarias, I(0)." if len(rej) == len(V) else
                   "<b>No se rechaza H₀ en ninguna serie</b> (p&gt;0.05): raíz unitaria en niveles (candidatas a I(1))." if not rej
                   else f"Resultado mixto: se rechaza H₀ solo en {', '.join(rej)}."))
    s = R.get("rezagos")
    if s:
        sel = s["seleccionados"]; el = s.get("elegido", 2)
        T[3] = (f"Criterios: AIC→<b>{sel['aic']}</b>, BIC→<b>{sel['bic']}</b>, HQ→<b>{sel['hqic']}</b>, FPE→<b>{sel['fpe']}</b>. "
                f"Se usa <b>p = {el}</b>; se justifica solo si los residuales resultan ruido blanco (siguiente paso).")
    d = R.get("diag_var")
    T[4] = lectura_diag(d) if d else ""
    if J:
        r, tr, ei, K = J["rango"], J["traza"], J["eigen"], len(V)
        filas = " ".join(f"<b>H₀: r{'=' if i == 0 else '≤'}{i}</b> traza {f(tr['stat'][i], 2)} vs {f(tr['cv'][i][1], 2)} "
                         f"({'rechaza' if tr['stat'][i] > tr['cv'][i][1] else 'no rechaza'});" for i in range(len(tr["stat"])))
        reg = ("Π tiene rango 0: no hay cointegración → <b>VAR en diferencias</b>." if r == 0 else
               f"Rango completo ({K}): series estacionarias → <b>VAR en niveles</b>." if r >= K else
               f"Rango {r}: hay <b>{r} relación(es) de cointegración</b> → <b>VECM</b> con coint_rank={r}.")
        con = "ambos criterios coinciden" if all((tr["stat"][i] > tr["cv"][i][1]) == (ei["stat"][i] > ei["cv"][i][1]) for i in range(len(tr["stat"]))) else "traza y valor propio máximo discrepan: revisar"
        esp = (D_ESPERADO.get(e) if isinstance(e, int) else None)
        T[5] = f"Al 5 %: {filas} {con}. {reg} " + (ok(r == esp, "coincide con el escenario", "no coincide con el escenario esperado") if esp is not None else "")
    if M: T[6] = lectura_modelo(R)
    d2 = R.get("diag_modelo")
    T[7] = lectura_diag(d2) if d2 else ""
    p = R.get("pronostico")
    if p:
        pr, inf, sup = (np.array(p[k]) for k in ("pron", "inf", "sup")); w = (sup - inf) / 2; crece = w[-1] / w[0]
        v0 = V[0]
        T[8] = (f"{p['h']} pasos, IC {int(p['ci'] * 100)} % sobre {p['sobre']}. Paso 1: " + "; ".join(f"{v} = <b>{f(pr[0, i])}</b> ± {f(w[0, i])}" for i, v in enumerate(V[:3]))
                + f". Paso {p['h']} ({v0}): <b>{f(pr[-1, 0])}</b> ± {f(w[-1, 0])}. "
                + (f"El ancho medio crece ×{f(crece[0], 1)}: la incertidumbre se acumula (series I(1))." if crece[0] > 1.5 and p['sobre'] == "niveles"
                   else f"El ancho medio apenas cambia (×{f(crece[0], 2)}): sistema estacionario." if p['sobre'] == "niveles"
                   else f"Se pronostican diferencias: la media tiende a ~0 y el ancho se estabiliza (×{f(crece[0], 2)})."))
    I = R.get("irf")
    if I:
        phi0, psi0 = np.array(I["phi"][0]), np.array(I["psi"][0])
        T[9] = "En t = 0 la IRF no ortogonalizada es la identidad Φ₀ = I (choque unitario en una variable, 0 en las demás). " + resumen_irf(I["no_ortog"], V)
        c1 = (f"un choque ortogonal en {V[0]} mueve a {V[1]} en t=0 por <b>{f(psi0[1, 0])}</b> y el choque en {V[1]} no afecta a {V[0]} contemporáneamente. " if len(V) == 2
              else f"Ψ₀ es triangular inferior: el choque en {V[0]} afecta a todas en t=0; el de {V[-1]} solo a sí misma. ")
        T[10] = f"Ψ₀ = Φ₀·P (P = Cholesky de Σᵤ, orden {' → '.join(V)}). Cambio frente a Φ₀: {c1}" + resumen_irf(I["ortog"], V)
    T[11] = conclusion(R)
    return T


def lectura_diag(d):
    pm, nm, ar = d["portmanteau"], d["normalidad"], d["arch"]
    L = [f"<b>Correlación serial</b> (Portmanteau, {pm['nlags']} rez.): p = {pv(pm['p'])} → " + ("sin evidencia de autocorrelación (supuesto clave)." if pm["p"] > .05 else "<b>hay autocorrelación</b>: aumentar rezagos.")]
    mal = [f"{v} (p={pv(a['p'])})" for v, a in ar.items() if a["p"] < .05]
    L.append("<b>Heterocedasticidad</b> (ARCH-LM): " + ("sin evidencia en ninguna ecuación." if not mal else f"se rechaza homocedasticidad en {', '.join(mal)} (con varias pruebas, un p cercano a 0.05 puede ser falsa alarma)."))
    L.append(f"<b>Normalidad</b> (Jarque-Bera multivariado): p = {pv(nm['p'])} → " + ("no se rechaza normalidad." if nm["p"] > .05 else "se rechaza normalidad."))
    return " ".join(L)


def lectura_modelo(R):
    M, V = R["modelo"], R["vars"]; t = M["tipo"]; A = np.array(M["A"])
    if t.startswith("VECM"):
        r = M.get("rango", 1); v0 = V[0]; be, al = M["beta"], M["alpha"]; c = M["const_coint"]
        s = ""
        for j in range(min(r, 2)):
            t_ = " ".join(f"{'+' if be[v][j] >= 0 else '−'} {f(abs(be[v][j]))}·{v}" for v in V if v != v0)
            s += f"Relación {j + 1} (β normalizado en {v0}): <b>{v0} {t_} {'+' if (c[j] or 0) >= 0 else '−'} {f(abs(c[j] or 0))} = 0</b>. "
        s += "Velocidades de ajuste α: " + ", ".join(f"α<sub>{v}</sub> = <b>{f(al[v][0])}</b>" for v in V) + ". "
        if r == 1 and len(V) == 2:
            ax, ay = al[V[0]][0], al[V[1]][0]
            s += (f"Con ECT = {v0} + β·{V[1]} + c, un ECT &gt; 0 hace que {v0} {'baje' if ax < 0 else 'suba'} y {V[1]} {'baje' if ay < 0 else 'suba'}: "
                  + ("ambas ajustan en sentido que cierra la brecha. " if ax < 0 < ay else "revisar signos (no ambas cierran la brecha). "))
        else:
            cierran = [v for v in V if al[v][0] * be[v][0] < 0 and abs(be[v][0]) > 1e-3]
            s += "Corrigen el desequilibrio (α·β < 0): <b>" + (", ".join(cierran) or "ninguna") + "</b>. "
        if M.get("vida_media"): s += f"ρ = 1 + β′α = {f(M['rho_desequilibrio'])} ⇒ vida media ≈ <b>{f(M['vida_media'], 2)}</b> periodos."
        return s
    est = "estable (raíces dentro del círculo unitario)" if M.get("estable") else "<b>inestable</b>"
    a1 = A[0]; K = len(V)
    s = f"Sistema {est}; " + ", ".join(f"efecto propio de {v}: {f(a1[i, i])}" for i, v in enumerate(V[:3])) + ". "
    if "diferencias" in t: s += "Sin relación de largo plazo: dinámica de corto plazo entre las diferencias; los coeficientes cruzados miden la transmisión."
    else: s += "Series estacionarias: los choques son transitorios y las variables revierten a su media."
    return s


def conclusion(R):
    M, J = R.get("modelo"), R.get("johansen")
    if not (M and J): return "Resultados incompletos: revise la lista de errores del registro."
    d = R.get("diag_modelo", {}); bien = d and d["portmanteau"]["p"] > .05
    if "diferencias" in M["tipo"] and (J.get("rango") or 0) > 0:
        base = (f"Johansen halla rango {J['rango']} (cointegración), pero el modo VAR estima un <b>{M['tipo']}</b>: "
                "omite la relación de largo plazo; para modelarla use el VECM. ")
    else:
        base = f"Johansen ⇒ rango {J['rango']} ⇒ <b>{M['tipo']}</b>. "
    base += "Residuales sin autocorrelación: el modelo es utilizable para pronóstico e IRF. " if bien else "Hay autocorrelación residual: ajustar rezagos antes de interpretar. "
    if M["tipo"].startswith("VECM"): base += "Las series comparten tendencia estocástica; lea pronóstico e IRF en niveles."
    elif "diferencias" in M["tipo"]:
        if (J.get("rango") or 0) == 0: base += "Estimar en niveles sería regresión espuria; por eso se modela en diferencias."
    else: base += "Se modela en niveles porque las series son estacionarias."
    return base


D_ESPERADO = ESPERADO


# ----------------------------------------------------------------------------- Excel con todos los calculos
def guardar_excel(res: dict, ruta: Path):
    import pandas as pd
    with pd.ExcelWriter(ruta) as xw:
        for n, R in res.items():
            pre = f"{R['etiqueta']}_"
            pd.DataFrame(R["adf"]).T.drop(columns="crit").to_excel(xw, sheet_name=(pre + "ADF")[:31])
            if R.get("rezagos"): pd.DataFrame(R["rezagos"]["filas"]).to_excel(xw, sheet_name=(pre + "Rezagos")[:31], index=False)
            if R.get("johansen"):
                J = R["johansen"]
                pd.DataFrame({"traza": J["traza"]["stat"], "cv95_traza": [c[1] for c in J["traza"]["cv"]],
                              "eigen": J["eigen"]["stat"], "cv95_eigen": [c[1] for c in J["eigen"]["cv"]]}).to_excel(xw, sheet_name=(pre + "Johansen")[:31])
            for clave in ("diag_var", "diag_modelo"):
                d = R.get(clave)
                if d:
                    filas = [("Portmanteau", d["portmanteau"]["stat"], d["portmanteau"]["p"]), ("Normalidad multivariada", d["normalidad"]["stat"], d["normalidad"]["p"])]
                    filas += [(f"ARCH {v}", a["lm"], a["p"]) for v, a in d["arch"].items()] + [(f"JB {v}", a["stat"], a["p"]) for v, a in d["jb"].items()]
                    pd.DataFrame(filas, columns=["prueba", "estadistico", "p_valor"]).to_excel(xw, sheet_name=(pre + clave)[:31], index=False)
            if R.get("pronostico"):
                p = R["pronostico"]
                cols = {"fecha": p["fecha"]}
                for i, v in enumerate(R["vars"]):
                    cols.update({v: np.array(p["pron"])[:, i], v + "_inf": np.array(p["inf"])[:, i], v + "_sup": np.array(p["sup"])[:, i]})
                pd.DataFrame(cols).to_excel(xw, sheet_name=(pre + "Pronostico")[:31], index=False)
            if R.get("irf"):
                I = R["irf"]; filas = []
                for tipo in ("no_ortog", "ortog"):
                    for k, v in I[tipo].items():
                        filas += [(tipo, k, h, a, lo, hi) for h, a, lo, hi in zip(I["pasos"], v["irf"], v["lo"], v["hi"])]
                pd.DataFrame(filas, columns=["tipo", "impulso>respuesta", "paso", "irf", "inf", "sup"]).to_excel(xw, sheet_name=(pre + "IRF")[:31], index=False)


# ----------------------------------------------------------------------------- principal
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--motor", choices=["auto", "numpy"], default="auto")
    ap.add_argument("--solo", choices=["todo", "escenarios", "auto"], default="todo", help="que JSON de resultados/ incluir")
    ap.add_argument("--pdf", action="store_true", help="tambien genera el PDF (playwright + chromium)")
    ap.add_argument("--salida", default="informe_taller_VECM.html")
    ap.add_argument("--titulo", default="Taller aplicado · Modelos VECM")
    ap.add_argument("--autor", default="Luis Guevara")
    a = ap.parse_args()
    res = cargar(a.motor, a.solo)
    datos = {"generado": datetime.now().strftime("%Y-%m-%d %H:%M"), "esc": {n: R for n, R in res.items()},
             "orden": list(res), "txt": {n: interpretar(R) for n, R in res.items()},
             "titulos": {n: R.get("titulo") or n for n, R in res.items()}, "esperado": {str(k): v for k, v in ESPERADO.items()}, "encabezado": a.titulo, "autor": a.autor}
    plantilla = (AQUI / "plantilla_informe.html").read_text(encoding="utf-8")
    html = plantilla.replace("/*__DATOS__*/null", json.dumps(datos, ensure_ascii=False))
    salida = AQUI / a.salida; salida.write_text(html, encoding="utf-8")
    print(f"HTML  → {salida}")
    (AQUI / "resultados").mkdir(exist_ok=True)
    try:
        guardar_excel(res, AQUI / "resultados" / "calculos_taller.xlsx"); print("Excel → resultados/calculos_taller.xlsx")
    except Exception as ex:                                  # noqa: BLE001
        print("[aviso] no se pudo crear el Excel:", ex)
    if a.pdf:
        try:
            from playwright.sync_api import sync_playwright
        except ImportError:
            print("[aviso] PDF automatico no disponible: falta playwright.\n"
                  "        Instale con:  pip install playwright   y luego:  playwright install chromium\n"
                  "        O abra el HTML y use el boton 'Descargar PDF' (imprime en vectorial).")
            return
        with sync_playwright() as pw:
            nav = pw.chromium.launch(); pg = nav.new_page(viewport={"width": 1800, "height": 1200})
            pg.goto(salida.as_uri()); pg.wait_for_timeout(300)
            pg.evaluate("window.prepararImpresion && window.prepararImpresion('poster')")
            alto = pg.evaluate("document.documentElement.scrollHeight")
            pg.pdf(path=str(AQUI / "informe_taller_VECM.pdf"), width="1800px", height=f"{alto + 2}px", print_background=True,
                   margin={"top": "0", "right": "0", "bottom": "0", "left": "0"})
            nav.close()
        print("PDF   → informe_taller_VECM.pdf")


if __name__ == "__main__":
    sys.exit(main())