#!/usr/bin/env python3
"""
informe_general.py  -  Informe general (Econometria II)

LO UNICO QUE USTED EDITA: el nombre de la base (ARCHIVO).  Luego:   python informe_general.py

Que hace
  1. Corre VAR_auto.py y VECM_auto.py sobre la misma base (cada uno decide solo columnas, tiempo, log,
     orden de Cholesky, rezagos y estructura, y deja su JSON en resultados/).
  2. Arma informe_general.html: CAPITULO 1 = resultados del VAR (columna 1) y del VECM (columna 2),
     seccion por seccion, con la bitacora de decisiones de cada uno, boton "Descargar PDF" y cajas editables.
  Si uno de los dos falla, se informa y el otro se muestra igual.
"""
import os
import subprocess
import sys
from pathlib import Path

# ----------------------------------------------------------------------------------------------
ARCHIVO = "datos_escenario2_I(1)_cointegradas.xlsx"   # <- nombre de su base (.xlsx / .xls / .csv) en esta carpeta
AUTOR = "Luis Guevara"
TITULO = "Informe general · Capítulo 1: resultados del VAR y del VECM"
PDF = False                                           # True: ademas crea el PDF (requiere playwright)
AJUSTES_COMUNES = '{}'   # vacio = todo automatico. Opcional (JSON, para AMBOS modelos): '{"inicio": "2000M1", "frecuencia": "M", "p": 2}'
# ----------------------------------------------------------------------------------------------

AQUI = Path(__file__).resolve().parent


def correr(script):
    env = {**os.environ, "VECM_ARCHIVO": ARCHIVO, "VECM_AJUSTES": AJUSTES_COMUNES}
    print(f"\n######## {script}  |  {ARCHIVO} ########")
    return subprocess.run([sys.executable, str(AQUI / script)], cwd=str(AQUI), env=env).returncode


def main():
    (AQUI / "resultados").mkdir(exist_ok=True)
    for f in (AQUI / "resultados").glob("auto_*.json"):
        f.unlink()                                   # el informe refleja solo esta corrida
    fallos = [s for s in ("VAR_auto.py", "VECM_auto.py") if correr(s) != 0]
    cmd = [sys.executable, str(AQUI / "generar_informe.py"), "--solo", "auto", "--salida", "informe_general.html",
           "--titulo", TITULO, "--autor", AUTOR] + (["--pdf"] if PDF else [])
    subprocess.run(cmd, cwd=str(AQUI))
    print("\nInforme general:", AQUI / "informe_general.html")
    if fallos:
        print("Con error (revise su salida arriba):", fallos)
        sys.exit(1)


if __name__ == "__main__":
    main()
