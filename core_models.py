"""
core_models.py — Motores matemáticos locales: Dixon-Coles, XGBoost Poisson, MCMC bayesiano.
Toda la computación pesada ocurre aquí; el LLM nunca recibe datos crudos.
"""

from __future__ import annotations

import warnings
import logging
from typing import Optional

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.stats import poisson

warnings.filterwarnings("ignore")
logger = logging.getLogger(__name__)

# ─── Constantes ────────────────────────────────────────────────────────────────
HALF_LIFE_DEFAULT = 365          # días; controla decaimiento temporal
MAX_GOALS = 8                    # rango de simulación por equipo
N_SIMULATIONS = 10_000           # simulaciones Monte Carlo
DATA_URL = (
    "https://raw.githubusercontent.com/martj42/international_results/"
    "master/results.csv"
)


# ═══════════════════════════════════════════════════════════════════════════════
# 1. CARGA Y PREPROCESAMIENTO DE DATOS
# ═══════════════════════════════════════════════════════════════════════════════

def load_data(url: str = DATA_URL) -> pd.DataFrame:
    """Descarga y limpia el dataset de resultados internacionales."""
    try:
        df = pd.read_csv(url, parse_dates=["date"])
    except Exception as exc:
        raise RuntimeError(f"No se pudo cargar el dataset: {exc}") from exc

    required = {"date", "home_team", "away_team", "home_score", "away_score"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Columnas faltantes en el dataset: {missing}")

    df = df.dropna(subset=list(required))
    df["home_score"] = df["home_score"].astype(int)
    df["away_score"] = df["away_score"].astype(int)
    df = df.sort_values("date").reset_index(drop=True)
    return df


def filter_teams(df: pd.DataFrame, team_a: str, team_b: str,
                 min_matches: int = 5, years: int = 12) -> pd.DataFrame:
    """
    Devuelve partidos que involucran a team_a O team_b, limitados a los
    últimos `years` años para mantener el conjunto de equipos manejable.
    """
    cutoff = df["date"].max() - pd.DateOffset(years=years)
    recent = df[df["date"] >= cutoff]
    mask = (recent["home_team"].isin([team_a, team_b])) | (recent["away_team"].isin([team_a, team_b]))
    sub = recent[mask].copy()
    for team in (team_a, team_b):
        count = ((sub["home_team"] == team) | (sub["away_team"] == team)).sum()
        if count < min_matches:
            raise ValueError(
                f"'{team}' tiene solo {count} partidos en los últimos {years} años "
                f"(mínimo requerido: {min_matches})."
            )
    return sub


def temporal_weights(dates: pd.Series, half_life: int = HALF_LIFE_DEFAULT,
                     ref_date: Optional[pd.Timestamp] = None) -> np.ndarray:
    """Pesos exponenciales: partidos recientes pesan más."""
    if ref_date is None:
        ref_date = dates.max()
    days_ago = (ref_date - dates).dt.days.clip(lower=0).values
    return np.exp(-np.log(2) * days_ago / half_life)


# ═══════════════════════════════════════════════════════════════════════════════
# 2. MODELO DIXON-COLES
# ═══════════════════════════════════════════════════════════════════════════════

def _dc_rho_correction(goals_h: int, goals_a: int,
                       mu_h: float, mu_a: float, rho: float) -> float:
    """Factor de corrección Dixon-Coles para marcadores bajos."""
    if goals_h == 0 and goals_a == 0:
        return 1 - mu_h * mu_a * rho
    if goals_h == 1 and goals_a == 0:
        return 1 + mu_a * rho
    if goals_h == 0 and goals_a == 1:
        return 1 + mu_h * rho
    if goals_h == 1 and goals_a == 1:
        return 1 - rho
    return 1.0


def _dc_log_likelihood(params: np.ndarray, teams: list[str],
                       home_idx: np.ndarray, away_idx: np.ndarray,
                       home_goals: np.ndarray, away_goals: np.ndarray,
                       weights: np.ndarray) -> float:
    """Log-verosimilitud negativa vectorizada del modelo Dixon-Coles."""
    n = len(teams)
    attack   = params[:n]
    defence  = params[n:2*n]
    home_adv = params[2 * n]
    rho      = params[2 * n + 1]

    mu_h = np.exp(attack[home_idx] + defence[away_idx] + home_adv)
    mu_a = np.exp(attack[away_idx] + defence[home_idx])

    gh = home_goals.astype(int)
    ga = away_goals.astype(int)

    # Corrección Dixon-Coles vectorizada
    tau = np.ones(len(gh))
    m00 = (gh == 0) & (ga == 0)
    m10 = (gh == 1) & (ga == 0)
    m01 = (gh == 0) & (ga == 1)
    m11 = (gh == 1) & (ga == 1)
    tau[m00] = np.maximum(1 - mu_h[m00] * mu_a[m00] * rho, 1e-10)
    tau[m10] = np.maximum(1 + mu_a[m10] * rho, 1e-10)
    tau[m01] = np.maximum(1 + mu_h[m01] * rho, 1e-10)
    tau[m11] = np.maximum(1 - rho, 1e-10)

    log_p = (
        poisson.logpmf(gh, mu_h)
        + poisson.logpmf(ga, mu_a)
        + np.log(tau)
    )
    return -np.dot(weights, log_p)


def fit_dixon_coles(df: pd.DataFrame,
                    half_life: int = HALF_LIFE_DEFAULT) -> dict:
    """
    Ajusta el modelo Dixon-Coles usando scipy.optimize.minimize.
    Retorna parámetros serializables (dict) listos para predicción.
    """
    teams = sorted(set(df["home_team"]) | set(df["away_team"]))
    n = len(teams)
    team_idx = {t: i for i, t in enumerate(teams)}

    weights = temporal_weights(df["date"], half_life)
    home_idx = np.array([team_idx[t] for t in df["home_team"]], dtype=np.int32)
    away_idx = np.array([team_idx[t] for t in df["away_team"]], dtype=np.int32)
    home_g   = df["home_score"].values.astype(float)
    away_g   = df["away_score"].values.astype(float)

    x0 = np.zeros(2 * n + 2)
    x0[2 * n] = 0.3
    x0[2 * n + 1] = -0.1

    bounds = (
        [(-3, 3)] * (2 * n)
        + [(0, 1)]
        + [(-1, 1)]
    )

    result = minimize(
        _dc_log_likelihood,
        x0,
        args=(teams, home_idx, away_idx, home_g, away_g, weights),
        method="L-BFGS-B",
        bounds=bounds,
        options={"maxiter": 500, "ftol": 1e-5},
    )

    if not result.success:
        logger.warning("Dixon-Coles no convergió: %s", result.message)

    params = result.x
    return {
        "teams": teams,
        "attack":  {t: float(params[i])   for i, t in enumerate(teams)},
        "defence": {t: float(params[n+i]) for i, t in enumerate(teams)},
        "home_adv": float(params[2 * n]),
        "rho":      float(params[2 * n + 1]),
        "half_life": half_life,
    }


def predict_score_matrix_dc(model: dict, home: str, away: str,
                             neutral: bool = False) -> np.ndarray:
    """
    Retorna la matriz de probabilidades [home_goals x away_goals]
    usando parámetros Dixon-Coles ya ajustados.
    """
    attack  = model["attack"]
    defence = model["defence"]
    home_adv = model["home_adv"] if not neutral else 0.0
    rho      = model["rho"]

    for team in (home, away):
        if team not in attack:
            raise ValueError(f"Equipo '{team}' no encontrado en el modelo DC.")

    mu_h = np.exp(attack[home] + defence[away] + home_adv)
    mu_a = np.exp(attack[away] + defence[home])

    matrix = np.zeros((MAX_GOALS + 1, MAX_GOALS + 1))
    for gh in range(MAX_GOALS + 1):
        for ga in range(MAX_GOALS + 1):
            tau = _dc_rho_correction(gh, ga, mu_h, mu_a, rho)
            p = poisson.pmf(gh, mu_h) * poisson.pmf(ga, mu_a) * tau
            matrix[gh, ga] = max(p, 0.0)

    total = matrix.sum()
    if total > 0:
        matrix /= total
    return matrix


# ═══════════════════════════════════════════════════════════════════════════════
# 3. MODELO XGBOOST POISSON
# ═══════════════════════════════════════════════════════════════════════════════

def _encode_features(df: pd.DataFrame, teams: list[str]) -> pd.DataFrame:
    """Codifica equipos como índices enteros para XGBoost."""
    team_idx = {t: i for i, t in enumerate(teams)}
    feat = pd.DataFrame({
        "home_idx": df["home_team"].map(team_idx).fillna(-1).astype(int),
        "away_idx": df["away_team"].map(team_idx).fillna(-1).astype(int),
    })
    return feat


def fit_xgb_poisson(df: pd.DataFrame,
                    half_life: int = HALF_LIFE_DEFAULT) -> dict:
    """
    Ajusta dos modelos XGBoost con objetivo Poisson:
    uno para goles del local, otro para goles del visitante.
    """
    try:
        import xgboost as xgb
    except ImportError as exc:
        raise ImportError("xgboost no instalado.") from exc

    teams = sorted(set(df["home_team"]) | set(df["away_team"]))
    weights = temporal_weights(df["date"], half_life)
    X = _encode_features(df, teams)

    params = {
        "objective": "count:poisson",
        "n_estimators": 100,
        "max_depth": 3,
        "learning_rate": 0.1,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "verbosity": 0,
        "random_state": 42,
    }

    model_h = xgb.XGBRegressor(**params)
    model_a = xgb.XGBRegressor(**params)

    model_h.fit(X, df["home_score"].values, sample_weight=weights)
    model_a.fit(X, df["away_score"].values, sample_weight=weights)

    return {"model_h": model_h, "model_a": model_a, "teams": teams}


def predict_xgb(xgb_bundle: dict, home: str, away: str) -> tuple[float, float]:
    """Retorna (mu_home, mu_away) predichos por XGBoost."""
    teams = xgb_bundle["teams"]
    team_idx = {t: i for i, t in enumerate(teams)}

    for team in (home, away):
        if team not in team_idx:
            raise ValueError(f"Equipo '{team}' no encontrado en el modelo XGB.")

    X = pd.DataFrame({
        "home_idx": [team_idx[home]],
        "away_idx": [team_idx[away]],
    })
    mu_h = float(xgb_bundle["model_h"].predict(X)[0])
    mu_a = float(xgb_bundle["model_a"].predict(X)[0])
    return max(mu_h, 0.01), max(mu_a, 0.01)


def predict_score_matrix_xgb(xgb_bundle: dict, home: str, away: str,
                              neutral: bool = False) -> np.ndarray:
    """Construye la matriz de probabilidades a partir de lambdas XGBoost."""
    mu_h, mu_a = predict_xgb(xgb_bundle, home, away)
    matrix = np.zeros((MAX_GOALS + 1, MAX_GOALS + 1))
    for gh in range(MAX_GOALS + 1):
        for ga in range(MAX_GOALS + 1):
            matrix[gh, ga] = poisson.pmf(gh, mu_h) * poisson.pmf(ga, mu_a)
    total = matrix.sum()
    if total > 0:
        matrix /= total
    return matrix


# ═══════════════════════════════════════════════════════════════════════════════
# 4. MCMC LOCAL (NumPy puro — sin PyMC para velocidad)
# ═══════════════════════════════════════════════════════════════════════════════

def mcmc_poisson_inference(home_goals: np.ndarray,
                           away_goals: np.ndarray,
                           n_samples: int = N_SIMULATIONS) -> dict:
    """
    Inferencia bayesiana simple via Metropolis-Hastings para estimar
    las tasas de gol (lambda_h, lambda_a) dados datos históricos H2H.
    Retorna muestras posteriores y estadísticos resumidos.
    """
    if len(home_goals) == 0:
        # Sin H2H: priors difusos
        return {"lambda_h": 1.3, "lambda_a": 1.0,
                "samples_h": np.random.gamma(1.3, 1, n_samples),
                "samples_a": np.random.gamma(1.0, 1, n_samples)}

    # Log-posterior: Poisson likelihood + Gamma(2,1) prior
    def log_posterior(lh: float, la: float) -> float:
        if lh <= 0 or la <= 0:
            return -np.inf
        ll = (
            np.sum(poisson.logpmf(home_goals, lh))
            + np.sum(poisson.logpmf(away_goals, la))
        )
        lp = (
            (2 - 1) * np.log(lh) - lh        # Gamma(2,1) prior
            + (2 - 1) * np.log(la) - la
        )
        return ll + lp

    # Metropolis-Hastings
    lh_curr = max(home_goals.mean(), 0.5)
    la_curr = max(away_goals.mean(), 0.5)
    lp_curr = log_posterior(lh_curr, la_curr)

    samples_h = np.empty(n_samples)
    samples_a = np.empty(n_samples)
    step = 0.3
    accepted = 0

    rng = np.random.default_rng(seed=42)
    for i in range(n_samples):
        lh_prop = lh_curr + rng.normal(0, step)
        la_prop = la_curr + rng.normal(0, step)
        lp_prop = log_posterior(lh_prop, la_prop)
        if np.log(rng.uniform()) < lp_prop - lp_curr:
            lh_curr, la_curr = lh_prop, la_prop
            lp_curr = lp_prop
            accepted += 1
        samples_h[i] = lh_curr
        samples_a[i] = la_curr

    acceptance_rate = accepted / n_samples
    logger.debug("MCMC acceptance rate: %.2f", acceptance_rate)

    return {
        "lambda_h": float(samples_h.mean()),
        "lambda_a": float(samples_a.mean()),
        "samples_h": samples_h,
        "samples_a": samples_a,
        "acceptance_rate": acceptance_rate,
    }


def predict_score_matrix_mcmc(mcmc_result: dict) -> np.ndarray:
    """Construye la matriz de probabilidades a partir de muestras MCMC (vectorizado)."""
    samples_h = mcmc_result["samples_h"]
    samples_a = mcmc_result["samples_a"]

    # Submuestreo a 500 muestras para velocidad
    step = max(1, len(samples_h) // 500)
    sub_h = samples_h[::step]
    sub_a = samples_a[::step]

    goals = np.arange(MAX_GOALS + 1)  # shape (G,)
    # pmf_h[s, g] = P(g goles | lambda=sub_h[s])
    pmf_h = poisson.pmf(goals[None, :], sub_h[:, None])  # (S, G)
    pmf_a = poisson.pmf(goals[None, :], sub_a[:, None])  # (S, G)

    # matrix[g_h, g_a] = mean over samples of pmf_h[s,g_h]*pmf_a[s,g_a]
    matrix = np.einsum("sh,sa->ha", pmf_h, pmf_a) / len(sub_h)

    total = matrix.sum()
    if total > 0:
        matrix /= total
    return matrix


# ═══════════════════════════════════════════════════════════════════════════════
# 5. ENSAMBLE Y MÉTRICAS FINALES
# ═══════════════════════════════════════════════════════════════════════════════

def ensemble_matrix(dc_matrix: np.ndarray, xgb_matrix: np.ndarray,
                    mcmc_matrix: np.ndarray,
                    weights: tuple[float, float, float] = (0.45, 0.35, 0.20)
                    ) -> np.ndarray:
    """Promedio ponderado de las tres matrices de probabilidad."""
    w_dc, w_xgb, w_mcmc = weights
    combined = w_dc * dc_matrix + w_xgb * xgb_matrix + w_mcmc * mcmc_matrix
    total = combined.sum()
    if total > 0:
        combined /= total
    return combined


def matrix_to_outcomes(matrix: np.ndarray) -> dict[str, float]:
    """Extrae probabilidades de victoria local, empate y victoria visitante."""
    p_home = float(np.tril(matrix, -1).sum())
    p_draw = float(np.diag(matrix).sum())
    p_away = float(np.triu(matrix, 1).sum())
    total = p_home + p_draw + p_away
    if total > 0:
        p_home, p_draw, p_away = p_home/total, p_draw/total, p_away/total
    return {"p_home": p_home, "p_draw": p_draw, "p_away": p_away}


def top_scorelines(matrix: np.ndarray, n: int = 10) -> list[dict]:
    """Retorna los n marcadores más probables como lista de dicts."""
    rows, cols = np.unravel_index(np.argsort(matrix, axis=None)[::-1][:n],
                                  matrix.shape)
    return [
        {"home": int(r), "away": int(c), "prob": float(matrix[r, c])}
        for r, c in zip(rows, cols)
    ]


def expected_goals(matrix: np.ndarray) -> tuple[float, float]:
    """xG esperado para local y visitante a partir de la matriz."""
    goals = np.arange(matrix.shape[0])
    xg_h = float((matrix.sum(axis=1) * goals).sum())
    xg_a = float((matrix.sum(axis=0) * goals).sum())
    return round(xg_h, 2), round(xg_a, 2)


def build_compact_summary(home: str, away: str,
                           outcomes: dict, xg_h: float, xg_a: float,
                           top3: list[dict],
                           dominant_model: str = "ensamble") -> dict:
    """
    Construye el dict compacto (<500 tokens) que se enviará al LLM.
    NUNCA incluye datos crudos ni DataFrames.
    """
    return {
        "match": f"{home} vs {away}",
        "model": dominant_model,
        "outcomes": {
            "home_win_%": round(outcomes["p_home"] * 100, 1),
            "draw_%":     round(outcomes["p_draw"] * 100, 1),
            "away_win_%": round(outcomes["p_away"] * 100, 1),
        },
        "xG": {"home": xg_h, "away": xg_a},
        "top3_scorelines": [
            f"{s['home']}-{s['away']} ({s['prob']*100:.1f}%)"
            for s in top3[:3]
        ],
    }


def mse_prediction(predicted_h: float, predicted_a: float,
                   actual_h: int, actual_a: int) -> float:
    """MSE entre marcador predicho y real."""
    return ((predicted_h - actual_h) ** 2 + (predicted_a - actual_a) ** 2) / 2


def suggest_half_life_adjustment(mse_history: list[dict]) -> dict:
    """
    Analiza errores recientes por modelo y sugiere ajustes de HALF_LIFE
    o redistribución de pesos del ensamble.
    """
    if len(mse_history) < 1:
        return {"sugerencia": "Cierra al menos un partido para activar el ajuste automático."}

    recent = mse_history[-10:]

    def safe_mean(lst):
        vals = [v for v in lst if v is not None and not np.isnan(float(v))]
        return float(np.mean(vals)) if vals else np.nan

    errors_dc   = [r.get("mse_dc")   for r in recent]
    errors_xgb  = [r.get("mse_xgb")  for r in recent]
    errors_mcmc = [r.get("mse_mcmc") for r in recent]
    avg_mse     = safe_mean([r.get("model_error") for r in recent])

    means = {
        "DC":   safe_mean(errors_dc),
        "XGB":  safe_mean(errors_xgb),
        "MCMC": safe_mean(errors_mcmc),
    }

    # Si no hay desglose por modelo, usamos ensamble como referencia
    valid_means = {k: v for k, v in means.items() if not np.isnan(v)}
    if valid_means:
        best = min(valid_means, key=lambda k: valid_means[k])
        sugerencia_pesos = f"Incrementar peso de '{best}' en el ensamble."
    else:
        best = "ensamble"
        sugerencia_pesos = "Acumula más partidos cerrados para comparar modelos."

    hl_suggestion = "mantener HALF_LIFE actual"
    if not np.isnan(avg_mse):
        if avg_mse > 2.0:
            hl_suggestion = "reducir HALF_LIFE (dar más peso a datos recientes)"
        elif avg_mse < 0.5:
            hl_suggestion = "aumentar HALF_LIFE (los datos históricos son estables)"

    n = len(recent)
    return {
        "partidos_analizados": n,
        "mse_ensamble_promedio": round(avg_mse, 4) if not np.isnan(avg_mse) else None,
        "mse_por_modelo": {k: round(v, 3) if not np.isnan(v) else "sin datos" for k, v in means.items()},
        "modelo_más_preciso": best,
        "half_life": hl_suggestion,
        "sugerencia_pesos": sugerencia_pesos,
    }
