"""
token_orchestrator.py — Capa de IA: gestión de contexto, memoria comprimida
y llamadas optimizadas al LLM (< 500 tokens por llamada).
"""

from __future__ import annotations

import json
import logging
import os
from datetime import datetime
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)

VAULT_PATH = Path(__file__).parent / "memory_vault.json"

# ─── Esquema estricto de registro ──────────────────────────────────────────────
_REQUIRED_KEYS = {"fecha", "local", "visitante", "goles_l", "goles_v"}
_PREDICTION_KEYS = {"fecha", "local", "visitante",
                    "pred_l", "pred_a", "estado",
                    "mse_ensemble", "mse_dc", "mse_xgb", "mse_mcmc"}


# ═══════════════════════════════════════════════════════════════════════════════
# 1. MEMORY VAULT — lectura / escritura anti-alucinaciones
# ═══════════════════════════════════════════════════════════════════════════════

def _load_vault() -> dict:
    """Carga el vault desde disco; crea estructura vacía si no existe."""
    if VAULT_PATH.exists():
        try:
            with open(VAULT_PATH, "r", encoding="utf-8") as f:
                return json.load(f)
        except (json.JSONDecodeError, OSError) as exc:
            logger.error("Error al leer memory_vault.json: %s", exc)
    return {"predictions": [], "performance": [], "meta": {}}


def _save_vault(vault: dict) -> None:
    """Persiste el vault en disco con formato legible."""
    try:
        with open(VAULT_PATH, "w", encoding="utf-8") as f:
            json.dump(vault, f, ensure_ascii=False, indent=2)
    except OSError as exc:
        logger.error("No se pudo guardar memory_vault.json: %s", exc)


def save_prediction(home: str, away: str,
                    pred_h: float, pred_a: float,
                    outcomes: dict,
                    top3: list[dict],
                    match_date: Optional[str] = None) -> str:
    """
    Guarda una predicción con estado 'pendiente'.
    Retorna el ID del registro para referencia posterior.
    """
    vault = _load_vault()
    record_id = f"{home.replace(' ', '_')}_vs_{away.replace(' ', '_')}_{datetime.now().strftime('%Y%m%d%H%M%S')}"
    record = {
        "id":          record_id,
        "fecha":       match_date or datetime.now().strftime("%Y-%m-%d"),
        "local":       home,
        "visitante":   away,
        "pred_l":      round(pred_h, 2),
        "pred_a":      round(pred_a, 2),
        "outcomes":    outcomes,
        "top3":        top3,
        "estado":      "pendiente",
        "mse_ensemble": None,
        "mse_dc":       None,
        "mse_xgb":      None,
        "mse_mcmc":     None,
        "goles_l_real": None,
        "goles_a_real": None,
    }
    vault["predictions"].append(record)
    _save_vault(vault)
    logger.info("Predicción guardada: %s", record_id)
    return record_id


def close_match(record_id: str,
                actual_h: int, actual_a: int,
                mse_dc: Optional[float] = None,
                mse_xgb: Optional[float] = None,
                mse_mcmc: Optional[float] = None,
                mse_ensemble: Optional[float] = None) -> dict:
    """
    Cierra un partido ingresando el marcador real.
    Calcula errores, actualiza el registro y retorna el análisis.
    """
    vault = _load_vault()
    record = next((r for r in vault["predictions"] if r["id"] == record_id), None)
    if record is None:
        raise KeyError(f"No se encontró la predicción '{record_id}'.")

    if record["estado"] == "cerrado":
        raise ValueError("Este partido ya fue cerrado.")

    record["estado"]      = "cerrado"
    record["goles_l_real"] = actual_h
    record["goles_a_real"] = actual_a
    record["mse_ensemble"] = mse_ensemble
    record["mse_dc"]       = mse_dc
    record["mse_xgb"]      = mse_xgb
    record["mse_mcmc"]     = mse_mcmc

    perf_entry = {
        "id":           record_id,
        "fecha":        record["fecha"],
        "local":        record["local"],
        "visitante":    record["visitante"],
        "goles_l":      actual_h,
        "goles_v":      actual_a,
        "model_error":  round(mse_ensemble, 4) if mse_ensemble is not None else None,
        "mse_dc":       round(mse_dc,       4) if mse_dc       is not None else None,
        "mse_xgb":      round(mse_xgb,      4) if mse_xgb      is not None else None,
        "mse_mcmc":     round(mse_mcmc,     4) if mse_mcmc     is not None else None,
    }
    vault["performance"].append(perf_entry)
    _save_vault(vault)
    return perf_entry


def get_pending_predictions() -> list[dict]:
    """Retorna todas las predicciones con estado 'pendiente'."""
    vault = _load_vault()
    return [r for r in vault["predictions"] if r["estado"] == "pendiente"]


def get_performance_history() -> list[dict]:
    """Retorna el historial de rendimiento (partidos cerrados)."""
    vault = _load_vault()
    return vault.get("performance", [])


def safe_lookup(record_id: str, field: str) -> str:
    """
    Busca un campo en una predicción guardada.
    Retorna 'Dato no disponible' si no existe — nunca inventa valores.
    """
    vault = _load_vault()
    record = next((r for r in vault["predictions"] if r["id"] == record_id), None)
    if record is None:
        return "Dato no disponible"
    value = record.get(field)
    return str(value) if value is not None else "Dato no disponible"


# ═══════════════════════════════════════════════════════════════════════════════
# 2. COMPRESIÓN DE CONTEXTO — anti-alucinaciones estricto
# ═══════════════════════════════════════════════════════════════════════════════

def compress_context(records: list[dict]) -> list[dict]:
    """
    Comprime una lista de registros al esquema mínimo estricto.
    NUNCA elimina: fecha, local, visitante, goles, model_error.
    Datos faltantes → None (no se inventan).
    """
    compressed = []
    for r in records:
        entry = {
            "fecha":        r.get("fecha"),
            "local":        r.get("local"),
            "visitante":    r.get("visitante"),
            "goles_l":      r.get("goles_l") if r.get("goles_l") is not None else r.get("goles_l_real"),
            "goles_v":      r.get("goles_v") if r.get("goles_v") is not None else r.get("goles_a_real"),
            "model_error":  r.get("model_error"),
        }
        # Validar que ningún campo crítico sea inventado
        for key in ("fecha", "local", "visitante"):
            if entry[key] is None:
                logger.warning("Campo crítico '%s' faltante en registro; descartado.", key)
                break
        else:
            compressed.append(entry)
    return compressed


# ═══════════════════════════════════════════════════════════════════════════════
# 3. PROMPT BUILDER — mantiene tokens < 500
# ═══════════════════════════════════════════════════════════════════════════════

def build_llm_prompt(compact_summary: dict,
                     performance_history: Optional[list[dict]] = None,
                     language: str = "es") -> str:
    """
    Construye el prompt final para el LLM usando únicamente el dict compacto.
    Estimado: ~350–450 tokens. Nunca incluye datos crudos.
    """
    match    = compact_summary.get("match", "Desconocido")
    model    = compact_summary.get("model", "ensamble")
    outcomes = compact_summary.get("outcomes", {})
    xg       = compact_summary.get("xG", {})
    top3     = compact_summary.get("top3_scorelines", [])

    perf_text = ""
    if performance_history:
        recent = performance_history[-5:]
        avg_err = _safe_avg([r.get("model_error") for r in recent])
        perf_text = (
            f"\n\nRendimiento reciente del modelo (últimos {len(recent)} partidos cerrados):\n"
            f"- Error cuadrático medio promedio: {avg_err}\n"
        )

    prompt = f"""Eres un analista deportivo experto. Genera un reporte conciso en {language} para el siguiente partido:

**Partido:** {match}
**Modelo usado:** {model}

**Probabilidades:**
- Victoria local: {outcomes.get('home_win_%', 'N/D')}%
- Empate: {outcomes.get('draw_%', 'N/D')}%
- Victoria visitante: {outcomes.get('away_win_%', 'N/D')}%

**Goles esperados (xG):**
- Local: {xg.get('home', 'N/D')}
- Visitante: {xg.get('away', 'N/D')}

**Top 3 marcadores más probables:**
{chr(10).join(f'- {s}' for s in top3)}{perf_text}

Instrucciones:
1. Describe brevemente las probabilidades y el marcador más probable.
2. Menciona el xG para contextualizar el nivel ofensivo esperado.
3. Sé conciso (máximo 150 palabras).
4. Si un dato no está disponible, di "Dato no disponible" — NUNCA inventes cifras.
"""
    return prompt.strip()


def _safe_avg(values: list) -> str:
    """Promedio seguro que maneja None y NaN."""
    nums = [v for v in values if v is not None]
    if not nums:
        return "Dato no disponible"
    return str(round(sum(nums) / len(nums), 4))


# ═══════════════════════════════════════════════════════════════════════════════
# 4. LLAMADA AL LLM (OPCIONAL) — con fallback sin API key
# ═══════════════════════════════════════════════════════════════════════════════

def call_llm(prompt: str, api_key: Optional[str] = None,
             model: str = "claude-haiku-4-5-20251001",
             max_tokens: int = 300) -> str:
    """
    Llama a la API de Anthropic con el prompt compacto.
    Si no hay API key, retorna un reporte generado localmente.
    """
    key = api_key or os.environ.get("ANTHROPIC_API_KEY")
    if not key:
        return _local_report(prompt)

    try:
        import anthropic
        client = anthropic.Anthropic(api_key=key)
        message = client.messages.create(
            model=model,
            max_tokens=max_tokens,
            messages=[{"role": "user", "content": prompt}],
        )
        return message.content[0].text
    except ImportError:
        logger.warning("Librería 'anthropic' no instalada. Usando reporte local.")
        return _local_report(prompt)
    except Exception as exc:
        logger.error("Error al llamar al LLM: %s", exc)
        return _local_report(prompt)


def _local_report(prompt: str) -> str:
    """
    Genera un reporte mínimo a partir del prompt cuando no hay LLM disponible.
    Extrae líneas clave del prompt para componer una respuesta estructurada.
    """
    lines = prompt.split("\n")
    relevant = [l.strip() for l in lines if l.strip().startswith(("-", "**")) and l.strip()]
    report = "**Reporte generado localmente** (sin API key):\n\n"
    report += "\n".join(relevant[:12])
    report += "\n\n*Para análisis narrativo completo, configure ANTHROPIC_API_KEY.*"
    return report
