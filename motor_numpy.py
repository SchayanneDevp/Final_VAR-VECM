"""Motor de RESPALDO en numpy/scipy (sin statsmodels) para el taller de VECM.

Para que sirve
--------------
* Produce el MISMO esquema JSON que `registro_informe.py` (que captura los objetos de statsmodels),
  de modo que `generar_informe.py` funciona aunque statsmodels no este instalado.
* Sirve de verificacion cruzada: si lo corres junto a los scripts del taller, las cifras principales
  (beta, alpha, trazas de Johansen, IRF) deben coincidir; pequenas diferencias en p-valores de ADF
  (aqui simulados), Portmanteau y bandas bootstrap son normales.

Convenciones (las mismas del taller)
------------------------------------
ADF: regression="n", maxlag=3, autolag AIC | VAR: trend="n" | Johansen: det_order=0, k_ar_diff=p-1,
valores criticos del caso "constante sin restringir" (k=2) | VECM: deterministic="ci" (constante en la
relacion de cointegracion), beta normalizado en x | IRF ortogonalizadas: Cholesky de sigma_u (orden x, y).
"""
from __future__ import annotations

import numpy as np
from scipy import stats

NOMBRES = ["x", "y"]
# Valores criticos de Johansen (2 variables, constante) al 90/95/99 %: filas r=0 y r<=1.
CV_TRAZA = [[13.4294, 15.4943, 19.9349], [2.7055, 3.8415, 6.6349]]
CV_EIGEN = [[12.2971, 14.2639, 18.5200], [2.7055, 3.8415, 6.6349]]


# ----------------------------------------------------------------------------- utilidades
def _lags(Y, p, offset=0):
    """Matriz [Y_{t-1},...,Y_{t-p}] y respuesta Y_t, descartando las primeras p+offset filas."""
    T = len(Y)
    X = np.hstack([Y[p - j - 1 + offset: T - j - 1] for j in range(p)]) if p else np.zeros((T - p - offset, 0))
    return X, Y[p + offset:]


def _ols(X, Y):
    B = np.linalg.lstsq(X, Y, rcond=None)[0] if X.shape[1] else np.zeros((0, Y.shape[1]))
    return B, Y - X @ B


def _coefs(B, p, k):
    """B (k*p x k) -> lista de A_i (k x k) con la convencion y_t = sum A_i y_{t-i}."""
    return [B[i * k:(i + 1) * k].T for i in range(p)]


def _ma(A, h):
    k = A[0].shape[0]
    Phi = [np.eye(k)]
    for s in range(1, h + 1):
        Phi.append(sum(A[j - 1] @ Phi[s - j] for j in range(1, min(s, len(A)) + 1)))
    return np.array(Phi)


def _lista(a):
    return np.asarray(a, dtype=float).tolist()


# ----------------------------------------------------------------------------- ADF (regression="n")
_ADF_CACHE: dict = {}


def _adf_nulo(n, reps=20000, seed=7):
    """Distribucion simulada del estadistico tau (sin constante) con n observaciones efectivas."""
    if n not in _ADF_CACHE:
        rng = np.random.default_rng(seed)
        e = rng.standard_normal((reps, n))
        y = np.cumsum(e, axis=1)
        ylag, dy = y[:, :-1], e[:, 1:]
        g = (ylag * dy).sum(1) / (ylag ** 2).sum(1)
        s2 = ((dy - g[:, None] * ylag) ** 2).sum(1) / (n - 2)
        _ADF_CACHE[n] = np.sort(g / np.sqrt(s2 / (ylag ** 2).sum(1)))
    return _ADF_CACHE[n]


def adf(y, maxlag=3):
    y = np.asarray(y, float); dy = np.diff(y)

    def ajuste(L, desde):
        yl = y[desde:-1]; d = dy[desde:]
        Z = [yl] + [dy[desde - j - 1: len(dy) - j - 1] for j in range(L)]
        X = np.column_stack(Z); b, u = _ols(X, d[:, None]); u = u[:, 0]
        n = len(d); s2 = u @ u / (n - X.shape[1])
        t = b[0, 0] / np.sqrt(s2 * np.linalg.inv(X.T @ X)[0, 0])
        aic = n * np.log(u @ u / n) + 2 * X.shape[1]
        return t, aic, n
    mejor = min(range(maxlag + 1), key=lambda L: ajuste(L, maxlag)[1])   # misma muestra para comparar
    t, _, n = ajuste(mejor, mejor)
    nulo = _adf_nulo(n)
    return {"stat": float(t), "p": float(np.searchsorted(nulo, t) / len(nulo)), "lags": int(mejor), "nobs": int(n),
            "crit": {k: float(np.quantile(nulo, q)) for k, q in (("1%", .01), ("5%", .05), ("10%", .10))}}


# ----------------------------------------------------------------------------- VAR y diagnosticos
def seleccion_rezagos(Y, maxlags=6):
    k = Y.shape[1]; filas = []
    for p in range(1, maxlags + 1):
        X, Z = _lags(Y, p, maxlags - p); B, U = _ols(X, Z)
        n = len(U); ld = np.linalg.slogdet(U.T @ U / n)[1]; m = k * p
        filas.append({"p": p, "aic": ld + 2 * p * k * k / n, "bic": ld + np.log(n) * p * k * k / n,
                      "hqic": ld + 2 * np.log(np.log(n)) * p * k * k / n, "fpe": ((n + m) / (n - m)) ** k * np.exp(ld)})
    sel = {c: min(filas, key=lambda f: f[c])["p"] for c in ("aic", "bic", "hqic", "fpe")}
    return {"filas": filas, "seleccionados": sel}


def ajustar_var(Y, p):
    k = Y.shape[1]; X, Z = _lags(Y, p); B, U = _ols(X, Z); n = len(U)
    A = _coefs(B, p, k)
    comp = np.zeros((k * p, k * p)); comp[:k] = np.hstack(A)
    if p > 1: comp[k:, :-k] = np.eye(k * (p - 1))
    return {"A": A, "U": U, "n": n, "p": p, "k": k, "sigma_u": U.T @ U / (n - k * p),
            "raices": np.abs(np.linalg.eigvals(comp))}


def portmanteau(U, nlags, gl_p, adj=False):
    n = len(U); Uc = U - U.mean(0)
    C = [Uc[h:].T @ Uc[:len(Uc) - h] / n for h in range(nlags + 1)]
    ci = np.linalg.inv(C[0])
    q = sum((1 / (n - h) if adj else 1) * np.trace(C[h].T @ ci @ C[h] @ ci) for h in range(1, nlags + 1))
    q *= n * n if adj else n
    return {"stat": float(q), "p": float(stats.chi2.sf(q, gl_p)), "df": int(gl_p), "nlags": nlags}


def arch_lm(e, lags=12):
    e2 = e ** 2; n = len(e2) - lags
    X = np.column_stack([np.ones(n)] + [e2[lags - j - 1: len(e2) - j - 1] for j in range(lags)])
    y = e2[lags:]; b, u = _ols(X, y[:, None]); r2 = 1 - (u[:, 0] @ u[:, 0]) / ((y - y.mean()) @ (y - y.mean()))
    lm = n * r2
    return {"lm": float(lm), "p": float(stats.chi2.sf(lm, lags))}


def normalidad(U):
    n, k = U.shape; Uc = U - U.mean(0)
    w = np.linalg.inv(np.linalg.cholesky(np.cov(Uc, rowvar=False))) @ Uc.T
    b1 = (w ** 3).sum(1) / n; b2 = (w ** 4).sum(1) / n - 3
    jb = n * (b1 @ b1) / 6 + n * (b2 @ b2) / 24
    uni = {NOMBRES[i]: {"stat": float(stats.jarque_bera(U[:, i]).statistic), "p": float(stats.jarque_bera(U[:, i]).pvalue)}
           for i in range(k)}
    return {"stat": float(jb), "p": float(stats.chi2.sf(jb, 2 * k)), "df": 2 * k}, uni


def diagnosticos(U, gl_p, nlags=10):
    mult, uni = normalidad(U)
    return {"portmanteau": portmanteau(U, nlags, gl_p), "arch": {NOMBRES[i]: arch_lm(U[:, i]) for i in range(U.shape[1])},
            "normalidad": mult, "jb": uni}


# ----------------------------------------------------------------------------- Johansen y VECM
def _momentos(Y, p, ci):
    """Regresiones auxiliares de Johansen. ci=True: constante restringida a la relacion de cointegracion."""
    dY = np.diff(Y, axis=0)
    Z0 = dY[p - 1:]
    Z1 = Y[p - 1:-1]
    W = np.hstack([dY[p - 1 - j: len(dY) - j] for j in range(1, p)]) if p > 1 else np.zeros((len(Z0), 0))
    if ci: Z1 = np.column_stack([Z1, np.ones(len(Z1))])
    else:  W = np.column_stack([np.ones(len(Z0)), W])
    R0 = Z0 - W @ np.linalg.lstsq(W, Z0, rcond=None)[0] if W.shape[1] else Z0
    R1 = Z1 - W @ np.linalg.lstsq(W, Z1, rcond=None)[0] if W.shape[1] else Z1
    n = len(Z0)
    return R0.T @ R0 / n, R1.T @ R1 / n, R0.T @ R1 / n, n, Z0, Z1, W


def _autov(S00, S11, S01):
    M = np.linalg.solve(S11, S01.T @ np.linalg.solve(S00, S01))
    w, V = np.linalg.eig(M); o = np.argsort(-w.real)
    return w.real[o], V.real[:, o]


def johansen(Y, p):
    S00, S11, S01, n, *_ = _momentos(Y, p, ci=False)
    lam, _ = _autov(S00, S11, S01); lam = lam[:Y.shape[1]]
    tr = [-n * np.log(1 - lam[r:]).sum() for r in range(len(lam))]
    ei = [-n * np.log(1 - lam[r]) for r in range(len(lam))]
    rango = 0
    for r in range(len(lam)):
        if tr[r] > CV_TRAZA[r][1]: rango = r + 1
        else: break
    return {"traza": {"stat": _lista(tr), "cv": CV_TRAZA}, "eigen": {"stat": _lista(ei), "cv": CV_EIGEN},
            "autovalores": _lista(lam), "rango": int(rango), "nobs": int(n)}


def ajustar_vecm(Y, p, r=1):
    k = Y.shape[1]
    S00, S11, S01, n, Z0, Z1, W = _momentos(Y, p, ci=True)
    lam, V = _autov(S00, S11, S01)
    b = V[:, :r] / V[0, :r]                                   # normaliza en x (primer elemento = 1)
    al = S01 @ b @ np.linalg.inv(b.T @ S11 @ b)
    ect = Z1 @ b
    G, U = _ols(W, Z0 - ect @ al.T) if W.shape[1] else (np.zeros((0, k)), Z0 - ect @ al.T)
    Gam = [G[i * k:(i + 1) * k].T for i in range(p - 1)]
    Pi = al @ b[:k].T
    A = [np.eye(k) + Pi + (Gam[0] if Gam else 0)]
    for i in range(1, p):
        A.append((Gam[i] if i < p - 1 else 0) - Gam[i - 1])
    nu = (al @ b[k:].T).ravel()                               # intercepto del VAR en niveles
    return {"A": A, "alpha": al, "beta": b[:k], "const": b[k:], "gamma": Gam, "nu": nu, "U": U, "n": n, "p": p, "k": k,
            "sigma_u": U.T @ U / (n - (k * (p - 1) + 1)), "rango": r}


# ----------------------------------------------------------------------------- pronostico e IRF
def pronosticar(A, nu, sigma, hist, h=12, ci=0.95):
    p, k = len(A), len(sigma); y = [*hist[-p:]]; med = []
    for _ in range(h):
        y.append(nu + sum(A[j] @ y[-1 - j] for j in range(p))); med.append(y[-1])
    Phi = _ma(A, h); z = stats.norm.ppf(1 - (1 - ci) / 2)
    cov = np.cumsum([Phi[i] @ sigma @ Phi[i].T for i in range(h)], axis=0)
    se = np.sqrt(np.array([np.diag(c) for c in cov]))
    med = np.array(med)
    return med, med - z * se, med + z * se


def _irf(A, sigma, h):
    Phi = _ma(A, h); P = np.linalg.cholesky(sigma)
    return Phi, np.array([f @ P for f in Phi])


def _simular(A, nu, hist0, U, rng, T):
    p, k = len(A), len(nu); y = np.zeros((T, k)); y[:p] = hist0
    e = U[rng.integers(0, len(U), T - p)]
    for t in range(p, T):
        y[t] = nu + sum(A[j] @ y[t - j - 1] for j in range(p)) + e[t - p]
    return y


def bandas_irf(refit, A, nu, U, Y, h, reps=100, semilla=202601, nivel=0.95):
    rng = np.random.default_rng(semilla); P, O = [], []
    for _ in range(reps):
        try:
            Ab, sb = refit(_simular(A, nu, Y[:len(A)], U, rng, len(Y)))
            ph, ps = _irf(Ab, sb, h); P.append(ph); O.append(ps)
        except Exception:
            continue
    q = [(1 - nivel) / 2 * 100, (1 + nivel) / 2 * 100]
    return [np.percentile(np.array(x), q, axis=0) for x in (P, O)]


# ----------------------------------------------------------------------------- orquestador
def analizar(df, escenario, titulo="", pasos_irf=18, h_pron=12, reps=100, rank_forzado=None):
    Y = df[NOMBRES].to_numpy(float); k = 2; T = len(Y)
    periodos = [f"{1990 + i // 4}Q{i % 4 + 1}" for i in range(T)]
    fut = [f"{1990 + (T + i) // 4}Q{(T + i) % 4 + 1}" for i in range(h_pron)]
    R = {"esc": escenario, "titulo": titulo, "motor": "numpy (respaldo, sin statsmodels)", "errores": [],
         "periodos": periodos, "x": _lista(Y[:, 0]), "y": _lista(Y[:, 1]),
         "adf": {n: adf(Y[:, i]) for i, n in enumerate(NOMBRES)}}
    sel = seleccion_rezagos(Y); sel["elegido"] = 2; R["rezagos"] = sel
    v2 = ajustar_var(Y, 2)
    R["var_niveles"] = {"p": 2, "coefs": [_lista(a) for a in v2["A"]], "sigma_u": _lista(v2["sigma_u"]), "nobs": v2["n"],
                        "estable": bool(v2["raices"].max() < 1), "raiz_max_modulo": float(v2["raices"].max())}
    R["diag_var"] = diagnosticos(v2["U"], k * k * (10 - 2))
    jo = johansen(Y, 2); R["johansen"] = jo
    r = jo["rango"] if rank_forzado is None else rank_forzado

    if r == 0:                                               # VAR(1) en diferencias
        D = np.diff(Y, axis=0); m = ajustar_var(D, 1); A, nu, sig, base = m["A"], np.zeros(k), m["sigma_u"], D
        refit = lambda S: (lambda q: (q["A"], q["sigma_u"]))(ajustar_var(S, 1))
        R["modelo"] = {"tipo": "VAR(1) en diferencias", "rango": 0, "sobre": "diferencias", "A": [_lista(a) for a in A],
                       "sigma_u": _lista(sig), "nobs": m["n"], "estable": bool(m["raices"].max() < 1)}
        gl = k * k * (10 - 1); U = m["U"]
    elif r >= k:                                             # VAR(2) en niveles
        m = v2; A, nu, sig, base = m["A"], np.zeros(k), m["sigma_u"], Y
        refit = lambda S: (lambda q: (q["A"], q["sigma_u"]))(ajustar_var(S, 2))
        R["modelo"] = {"tipo": "VAR(2) en niveles", "rango": k, "sobre": "niveles", "A": [_lista(a) for a in A],
                       "sigma_u": _lista(sig), "nobs": m["n"], "estable": bool(m["raices"].max() < 1),
                       "raiz_max_modulo": float(m["raices"].max())}
        gl = k * k * (10 - 2); U = m["U"]
    else:                                                    # VECM(1), rango r, constante en la cointegracion
        m = ajustar_vecm(Y, 2, r); A, nu, sig, base = m["A"], m["nu"], m["sigma_u"], Y
        refit = lambda S: (lambda q: (q["A"], q["sigma_u"]))(ajustar_vecm(S, 2, r))
        rho = 1 + float((m["beta"][:, 0] @ m["alpha"][:, 0]))
        R["modelo"] = {"tipo": "VECM(1)", "rango": r, "sobre": "niveles", "A": [_lista(a) for a in A],
                       "beta": dict(zip(NOMBRES, _lista(m["beta"][:, 0]))), "const_coint": float(m["const"][0, 0]),
                       "alpha": dict(zip(NOMBRES, _lista(m["alpha"][:, 0]))), "gamma": [_lista(g) for g in m["gamma"]],
                       "sigma_u": _lista(sig), "nobs": m["n"], "rho_desequilibrio": rho,
                       "vida_media": float(np.log(.5) / np.log(rho)) if 0 < rho < 1 else None}
        gl = k * k * (10 - 2 + 1) - k * r; U = m["U"]
    R["diag_modelo"] = diagnosticos(U, gl)

    # pronostico
    sobre = R["modelo"]["sobre"]; hist = base
    med, lo, hi = pronosticar(A, nu, sig, hist, h_pron)
    fut_lbl = fut if sobre == "niveles" else fut
    R["pronostico"] = {"h": h_pron, "ci": 0.95, "sobre": sobre, "fecha": fut_lbl, "pron": _lista(med), "inf": _lista(lo),
                       "sup": _lista(hi), "ult": _lista(hist[-1]),
                       "hist": _lista(hist[-40:]), "hist_fecha": (periodos if sobre == "niveles" else periodos[1:])[-40:]}
    # IRF
    Phi, Psi = _irf(A, sig, pasos_irf)
    (bp, bo) = bandas_irf(refit, A, nu, U, base, pasos_irf, reps)
    def pack(M, B):
        return {f"{NOMBRES[i]}>{NOMBRES[j]}": {"irf": _lista(M[:, j, i]), "lo": _lista(B[0][:, j, i]), "hi": _lista(B[1][:, j, i])}
                for i in range(k) for j in range(k)}                    # impulso>respuesta
    R["irf"] = {"pasos": list(range(pasos_irf + 1)), "no_ortog": pack(Phi, bp), "ortog": pack(Psi, bo),
                "phi": [_lista(f) for f in Phi[:11]], "psi": [_lista(f) for f in Psi[:11]], "reps": reps}
    return R
