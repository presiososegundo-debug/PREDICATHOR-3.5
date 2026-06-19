"""
app.py — Interfaz Streamlit de FootAgent PREDICATHOR 3.5.
UI limpia, modular y con bucle de auto-mejora integrado.
"""

from __future__ import annotations

import logging
import traceback
from typing import Optional

import numpy as np
import pandas as pd
import streamlit as st

# ─── Configuración de página ──────────────────────────────────────────────────
st.set_page_config(
    page_title="PREDICATHOR 3.5 — FootAgent by PICHOINDUSTRIES",
    page_icon="⚽",
    layout="wide",
    initial_sidebar_state="expanded",
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ─── Importaciones internas (con manejo de error claro) ───────────────────────
try:
    from core_models import (
        load_data, filter_teams, fit_dixon_coles, fit_xgb_poisson,
        mcmc_poisson_inference, predict_score_matrix_dc,
        predict_score_matrix_xgb, predict_score_matrix_mcmc,
        ensemble_matrix, matrix_to_outcomes, top_scorelines,
        expected_goals, build_compact_summary, mse_prediction,
        suggest_half_life_adjustment, HALF_LIFE_DEFAULT, MAX_GOALS,
    )
    from token_orchestrator import (
        save_prediction, close_match, get_pending_predictions,
        get_performance_history, build_llm_prompt, call_llm,
        compress_context, safe_lookup,
    )

except ImportError as exc:
    st.error(f"Error importando módulos internos: {exc}")
    st.stop()

# ═══════════════════════════════════════════════════════════════════════════════
# ESTADO DE SESIÓN
# ═══════════════════════════════════════════════════════════════════════════════

def init_session():
    defaults = {
        "df": None,
        "dc_model": None,
        "xgb_bundle": None,
        "teams_list": [],
        "last_matrix": None,
        "last_home": "",
        "last_away": "",
        "last_outcomes": {},
        "last_xg": (0.0, 0.0),
        "last_top10": [],
        "last_pred_id": None,
        "half_life": HALF_LIFE_DEFAULT,
        "api_key": "",
        "data_loaded": False,
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v

init_session()


# ═══════════════════════════════════════════════════════════════════════════════
# HELPERS DE VISUALIZACIÓN — deben estar definidos antes de ser llamados
# ═══════════════════════════════════════════════════════════════════════════════


def _export_prediction_png(matrix: np.ndarray, home: str, away: str,
                            outcomes: dict, xg_h: float, xg_a: float,
                            top10: list[dict]) -> bytes:
    """Genera imagen PNG con métricas, heatmap y top 10 marcadores."""
    import io
    import matplotlib.pyplot as plt
    import matplotlib.gridspec as gridspec
    import seaborn as sns

    fig = plt.figure(figsize=(14, 10), facecolor="#0e1117")
    gs = gridspec.GridSpec(3, 2, figure=fig,
                           height_ratios=[0.8, 4, 4],
                           hspace=0.45, wspace=0.35)

    text_kw = dict(color="white", fontsize=11)

    # ── Fila 0: título y métricas ─────────────────────────────────────────────
    ax_title = fig.add_subplot(gs[0, :])
    ax_title.axis("off")
    ax_title.set_facecolor("#0e1117")
    ax_title.text(0.5, 0.80, f"PREDICATHOR 3.5  —  {home} vs {away}",
                  ha="center", va="center", fontsize=16, fontweight="bold",
                  color="white", transform=ax_title.transAxes)
    ax_title.text(0.5, 0.42, "P I C H O I N D U S T R I E S",
                  ha="center", va="center", fontsize=9, fontweight="bold",
                  color="#f0a500", transform=ax_title.transAxes)
    metrics = (
        f"Victoria {home}: {outcomes['p_home']*100:.1f}%     "
        f"Empate: {outcomes['p_draw']*100:.1f}%     "
        f"Victoria {away}: {outcomes['p_away']*100:.1f}%     "
        f"xG: {xg_h} – {xg_a}"
    )
    ax_title.text(0.5, 0.15, metrics,
                  ha="center", va="center", fontsize=10, color="#aaaaaa",
                  transform=ax_title.transAxes)

    # ── Fila 1-2 izq: heatmap ────────────────────────────────────────────────
    ax_heat = fig.add_subplot(gs[1:, 0])
    ax_heat.set_facecolor("#0e1117")
    labels = [[f"{matrix[i,j]*100:.1f}%" for j in range(matrix.shape[1])]
              for i in range(matrix.shape[0])]
    sns.heatmap(matrix * 100, annot=labels, fmt="", ax=ax_heat,
                cmap="YlOrRd", linewidths=0.4,
                xticklabels=range(matrix.shape[1]),
                yticklabels=range(matrix.shape[0]),
                annot_kws={"size": 8},
                cbar_kws={"shrink": 0.7})
    ax_heat.set_xlabel(f"Goles {away}", **text_kw)
    ax_heat.set_ylabel(f"Goles {home}", **text_kw)
    ax_heat.set_title("Heatmap de probabilidades (%)", color="white", fontsize=12)
    ax_heat.tick_params(colors="white")
    ax_heat.figure.axes[-1].tick_params(colors="white")  # colorbar

    # ── Fila 1-2 der: top 10 barras ──────────────────────────────────────────
    ax_bar = fig.add_subplot(gs[1:, 1])
    ax_bar.set_facecolor("#0e1117")
    labels_bar = [f"{s['home']}-{s['away']}" for s in top10]
    probs_bar  = [s["prob"] * 100 for s in top10]
    colors_bar = [
        "#4e79a7" if s["home"] > s["away"] else
        "#f28e2b" if s["home"] < s["away"] else
        "#59a14f"
        for s in top10
    ]
    bars = ax_bar.barh(labels_bar[::-1], probs_bar[::-1], color=colors_bar[::-1])
    ax_bar.bar_label(bars, fmt="%.1f%%", padding=3, fontsize=8, color="white")
    ax_bar.set_xlabel("Probabilidad (%)", **text_kw)
    ax_bar.set_title(f"Top 10 marcadores", color="white", fontsize=12)
    ax_bar.set_xlim(0, max(probs_bar) * 1.3)
    ax_bar.tick_params(colors="white")
    ax_bar.spines[:].set_color("#444444")
    legend_elements = [
        plt.Rectangle((0, 0), 1, 1, color="#4e79a7", label=f"Gana {home}"),
        plt.Rectangle((0, 0), 1, 1, color="#f28e2b", label=f"Gana {away}"),
        plt.Rectangle((0, 0), 1, 1, color="#59a14f", label="Empate"),
    ]
    ax_bar.legend(handles=legend_elements, loc="lower right", fontsize=8,
                  facecolor="#1a1a2e", labelcolor="white")
    ax_bar.grid(axis="x", alpha=0.2, color="white")

    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=150, bbox_inches="tight",
                facecolor=fig.get_facecolor())
    plt.close(fig)
    buf.seek(0)
    return buf.read()


def _render_heatmap(matrix: np.ndarray, home: str, away: str):
    import matplotlib.pyplot as plt
    import seaborn as sns

    fig, ax = plt.subplots(figsize=(7, 5))
    labels = [[f"{matrix[i,j]*100:.1f}%" for j in range(matrix.shape[1])]
              for i in range(matrix.shape[0])]
    sns.heatmap(
        matrix * 100, annot=labels, fmt="", ax=ax,
        cmap="YlOrRd", linewidths=0.5,
        xticklabels=range(matrix.shape[1]),
        yticklabels=range(matrix.shape[0]),
    )
    ax.set_xlabel(f"Goles {away}", fontsize=11)
    ax.set_ylabel(f"Goles {home}", fontsize=11)
    ax.set_title("Heatmap de Probabilidades de Marcador (%)", fontsize=13)
    st.pyplot(fig)
    plt.close(fig)


def _render_top10(top10: list[dict], home: str, away: str):
    import matplotlib.pyplot as plt

    labels = [f"{s['home']}-{s['away']}" for s in top10]
    probs  = [s["prob"] * 100 for s in top10]
    colors = [
        "#4e79a7" if s["home"] > s["away"] else
        "#f28e2b" if s["home"] < s["away"] else
        "#59a14f"
        for s in top10
    ]

    fig, ax = plt.subplots(figsize=(7, 4))
    bars = ax.barh(labels[::-1], probs[::-1], color=colors[::-1])
    ax.bar_label(bars, fmt="%.1f%%", padding=3, fontsize=9)
    ax.set_xlabel("Probabilidad (%)")
    ax.set_title(f"Top 10 Marcadores — {home} vs {away}")
    ax.set_xlim(0, max(probs) * 1.25)
    legend_elements = [
        plt.Rectangle((0, 0), 1, 1, color="#4e79a7", label=f"Victoria {home}"),
        plt.Rectangle((0, 0), 1, 1, color="#f28e2b", label=f"Victoria {away}"),
        plt.Rectangle((0, 0), 1, 1, color="#59a14f", label="Empate"),
    ]
    ax.legend(handles=legend_elements, loc="lower right", fontsize=8)
    ax.grid(axis="x", alpha=0.3)
    st.pyplot(fig)
    plt.close(fig)


# ═══════════════════════════════════════════════════════════════════════════════
# SIDEBAR
# ═══════════════════════════════════════════════════════════════════════════════

with st.sidebar:
    st.title("⚽ PREDICATHOR 3.5")
    st.caption("FootAgent — Predicción Bayesiana de Fútbol Internacional")
    st.markdown(
        "<div style='text-align:center; color:#f0a500; font-size:13px; "
        "font-weight:bold; letter-spacing:2px; margin-top:-8px;'>"
        "PICHOINDUSTRIES</div>",
        unsafe_allow_html=True,
    )
    st.divider()

    # API Key (opcional)
    api_key_input = st.text_input(
        "API Key Anthropic (opcional)",
        value=st.session_state["api_key"],
        type="password",
        help="Para reporte narrativo con IA. Sin key, se genera reporte local.",
    )
    st.session_state["api_key"] = api_key_input

    # Half-life
    st.session_state["half_life"] = st.slider(
        "HALF_LIFE (días de decaimiento)",
        min_value=60, max_value=730,
        value=st.session_state["half_life"], step=30,
        help="Cuánto peso tienen los datos históricos vs. recientes.",
    )

    st.divider()

    # Cargar datos
    if st.button("Cargar / Actualizar Dataset", use_container_width=True, type="primary"):
        with st.spinner("Descargando resultados internacionales..."):
            try:
                df = load_data()
                st.session_state["df"] = df
                teams = sorted(set(df["home_team"]) | set(df["away_team"]))
                st.session_state["teams_list"] = teams
                st.session_state["data_loaded"] = True
                st.success(f"Dataset cargado: {len(df):,} partidos | {len(teams)} equipos")
            except Exception as exc:
                st.error(f"Error al cargar datos: {exc}")
                logger.error(traceback.format_exc())

    if st.session_state["data_loaded"]:
        st.caption(f"Dataset: {len(st.session_state['df']):,} partidos cargados")

    st.divider()
    st.markdown("**Navegación**")
    page = st.radio(
        "Sección",
        ["Predicción", "Próximos Partidos", "Cerrar Partido", "Rendimiento"],
        label_visibility="collapsed",
    )


# ═══════════════════════════════════════════════════════════════════════════════
# PÁGINA: PREDICCIÓN
# ═══════════════════════════════════════════════════════════════════════════════

if page == "Predicción":
    st.title("Predicción de Partido")

    if not st.session_state["data_loaded"]:
        st.info("Carga el dataset primero usando el botón en la barra lateral.")
        st.stop()

    teams = st.session_state["teams_list"]
    col1, col2 = st.columns(2)
    with col1:
        home = st.selectbox("Equipo Local", teams, index=teams.index("Argentina") if "Argentina" in teams else 0)
    with col2:
        away = st.selectbox("Equipo Visitante", teams, index=teams.index("Brazil") if "Brazil" in teams else 1)

    neutral = st.checkbox("Partido en sede neutral (elimina ventaja de local)", value=False)
    match_date = st.date_input("Fecha del partido")

    col_run, col_space = st.columns([1, 3])
    with col_run:
        run = st.button("Predecir", use_container_width=True, type="primary")

    if run:
        if home == away:
            st.warning("Selecciona dos equipos distintos.")
            st.stop()

        df = st.session_state["df"]
        half_life = st.session_state["half_life"]

        with st.spinner("Ajustando modelos..."):
            try:
                sub = filter_teams(df, home, away)
            except ValueError as exc:
                st.error(str(exc))
                st.stop()

            # ── Dixon-Coles ──────────────────────────────────────────────────
            try:
                dc_model = fit_dixon_coles(sub, half_life)
                dc_matrix = predict_score_matrix_dc(dc_model, home, away, neutral)
                st.session_state["dc_model"] = dc_model
            except Exception as exc:
                st.warning(f"Dixon-Coles falló: {exc}. Usando distribución uniforme.")
                logger.error(traceback.format_exc())
                dc_matrix = np.ones((MAX_GOALS + 1, MAX_GOALS + 1)) / (MAX_GOALS + 1) ** 2

            # ── XGBoost ─────────────────────────────────────────────────────
            try:
                xgb_bundle = fit_xgb_poisson(sub, half_life)
                xgb_matrix = predict_score_matrix_xgb(xgb_bundle, home, away, neutral)
                st.session_state["xgb_bundle"] = xgb_bundle
            except Exception as exc:
                st.warning(f"XGBoost falló: {exc}. Usando DC como fallback.")
                logger.error(traceback.format_exc())
                xgb_matrix = dc_matrix.copy()

            # ── MCMC H2H ────────────────────────────────────────────────────
            try:
                h2h = sub[
                    ((sub["home_team"] == home) & (sub["away_team"] == away)) |
                    ((sub["home_team"] == away) & (sub["away_team"] == home))
                ]
                mcmc_result = mcmc_poisson_inference(
                    h2h["home_score"].values, h2h["away_score"].values
                )
                mcmc_matrix = predict_score_matrix_mcmc(mcmc_result)
            except Exception as exc:
                st.warning(f"MCMC falló: {exc}. Usando DC como fallback.")
                logger.error(traceback.format_exc())
                mcmc_matrix = dc_matrix.copy()

            # ── Ensamble ────────────────────────────────────────────────────
            matrix = ensemble_matrix(dc_matrix, xgb_matrix, mcmc_matrix)
            outcomes = matrix_to_outcomes(matrix)
            top10 = top_scorelines(matrix, n=10)
            xg_h, xg_a = expected_goals(matrix)

            # Guardamos en sesión
            st.session_state.update({
                "last_matrix": matrix,
                "last_home": home,
                "last_away": away,
                "last_outcomes": outcomes,
                "last_xg": (xg_h, xg_a),
                "last_top10": top10,
            })

            # Guarda predicción en vault
            pred_id = save_prediction(
                home=home, away=away,
                pred_h=xg_h, pred_a=xg_a,
                outcomes=outcomes,
                top3=top10[:3],
                match_date=str(match_date),
            )
            st.session_state["last_pred_id"] = pred_id

        # ── Métricas ──────────────────────────────────────────────────────────
        st.subheader(f"Resultado: {home} vs {away}")
        mc1, mc2, mc3, mc4, mc5 = st.columns(5)
        mc1.metric("Victoria Local", f"{outcomes['p_home']*100:.1f}%")
        mc2.metric("Empate", f"{outcomes['p_draw']*100:.1f}%")
        mc3.metric("Victoria Visitante", f"{outcomes['p_away']*100:.1f}%")
        mc4.metric("xG Local", xg_h)
        mc5.metric("xG Visitante", xg_a)

        # ── Heatmap ──────────────────────────────────────────────────────────
        _render_heatmap(matrix, home, away)

        # ── Top 10 marcadores ─────────────────────────────────────────────────
        _render_top10(top10, home, away)

        # ── Reporte LLM ──────────────────────────────────────────────────────
        compact = build_compact_summary(home, away, outcomes, xg_h, xg_a, top10[:3])
        prompt = build_llm_prompt(compact, get_performance_history())
        with st.expander("Reporte narrativo (IA)", expanded=True):
            with st.spinner("Generando reporte..."):
                report = call_llm(prompt, api_key=st.session_state["api_key"])
            st.markdown(report)

        # ── Exportar ─────────────────────────────────────────────────────────
        png_bytes = _export_prediction_png(matrix, home, away, outcomes, xg_h, xg_a, top10)
        filename = f"prediccion_{home.replace(' ','_')}_vs_{away.replace(' ','_')}.png"
        st.download_button(
            label="Descargar resultado (PNG)",
            data=png_bytes,
            file_name=filename,
            mime="image/png",
            use_container_width=True,
        )

        st.caption(f"ID de predicción guardada: `{pred_id}`")


# ═══════════════════════════════════════════════════════════════════════════════
# PÁGINA: PRÓXIMOS PARTIDOS
# ═══════════════════════════════════════════════════════════════════════════════

elif page == "Próximos Partidos":
    st.title("Próximos Partidos")
    st.info(
        "Ingresa partidos manualmente para predecir múltiples encuentros en lote. "
        "El dataset debe estar cargado."
    )

    if not st.session_state["data_loaded"]:
        st.warning("Carga el dataset primero.")
        st.stop()

    teams = st.session_state["teams_list"]

    with st.form("batch_form"):
        st.subheader("Agregar partidos")
        cols = st.columns([2, 2, 1, 1])
        home_b = cols[0].selectbox("Local", teams, key="b_home")
        away_b = cols[1].selectbox("Visitante", teams, key="b_away")
        date_b = cols[2].date_input("Fecha", key="b_date")
        neutral_b = cols[3].checkbox("Neutral", key="b_neutral")
        submit_batch = st.form_submit_button("Agregar y Predecir")

    if submit_batch:
        if home_b == away_b:
            st.warning("Equipos deben ser distintos.")
        else:
            df = st.session_state["df"]
            half_life = st.session_state["half_life"]
            try:
                sub = filter_teams(df, home_b, away_b)
                dc_model = fit_dixon_coles(sub, half_life)
                dc_matrix = predict_score_matrix_dc(dc_model, home_b, away_b, neutral_b)
                xgb_bundle = fit_xgb_poisson(sub, half_life)
                xgb_matrix = predict_score_matrix_xgb(xgb_bundle, home_b, away_b, neutral_b)
                h2h = sub[
                    ((sub["home_team"] == home_b) & (sub["away_team"] == away_b)) |
                    ((sub["home_team"] == away_b) & (sub["away_team"] == home_b))
                ]
                mcmc_result = mcmc_poisson_inference(h2h["home_score"].values, h2h["away_score"].values)
                mcmc_matrix = predict_score_matrix_mcmc(mcmc_result)
                matrix = ensemble_matrix(dc_matrix, xgb_matrix, mcmc_matrix)
                outcomes = matrix_to_outcomes(matrix)
                top3 = top_scorelines(matrix, 3)
                xg_h, xg_a = expected_goals(matrix)

                save_prediction(home_b, away_b, xg_h, xg_a, outcomes, top3, str(date_b))

                st.success(f"Predicción guardada: **{home_b} vs {away_b}**")
                c1, c2, c3 = st.columns(3)
                c1.metric("Local gana", f"{outcomes['p_home']*100:.1f}%")
                c2.metric("Empate", f"{outcomes['p_draw']*100:.1f}%")
                c3.metric("Visitante gana", f"{outcomes['p_away']*100:.1f}%")
                st.write("Top 3:", " | ".join(f"{s['home']}-{s['away']} ({s['prob']*100:.1f}%)" for s in top3))

            except Exception as exc:
                st.error(f"Error: {exc}")
                logger.error(traceback.format_exc())

    # Tabla de pendientes
    st.subheader("Predicciones pendientes")
    pending = get_pending_predictions()
    if pending:
        rows = [{
            "ID": r["id"],
            "Fecha": r["fecha"],
            "Local": r["local"],
            "Visitante": r["visitante"],
            "xG L": r["pred_l"],
            "xG V": r["pred_a"],
        } for r in pending]
        st.dataframe(pd.DataFrame(rows), use_container_width=True)
    else:
        st.info("No hay predicciones pendientes.")


# ═══════════════════════════════════════════════════════════════════════════════
# PÁGINA: CERRAR PARTIDO
# ═══════════════════════════════════════════════════════════════════════════════

elif page == "Cerrar Partido":
    st.title("Cerrar Partido")
    st.write("Ingresa el marcador real para actualizar el rendimiento del modelo.")

    pending = get_pending_predictions()
    if not pending:
        st.info("No hay predicciones pendientes para cerrar.")
        st.stop()

    options = {r["id"]: f"{r['local']} vs {r['visitante']} ({r['fecha']})" for r in pending}
    selected_id = st.selectbox("Selecciona predicción", list(options.keys()),
                               format_func=lambda x: options[x])

    record = next(r for r in pending if r["id"] == selected_id)
    st.markdown(f"**Predicción:** xG {record['pred_l']} – {record['pred_a']}")

    col_h, col_a = st.columns(2)
    real_h = col_h.number_input("Goles locales (real)", min_value=0, max_value=20, step=1)
    real_a = col_a.number_input("Goles visitantes (real)", min_value=0, max_value=20, step=1)

    if st.button("Cerrar Partido", type="primary"):
        mse_ens = mse_prediction(record["pred_l"], record["pred_a"], real_h, real_a)
        try:
            perf = close_match(
                record_id=selected_id,
                actual_h=int(real_h),
                actual_a=int(real_a),
                mse_ensemble=mse_ens,
            )
            st.success("Partido cerrado correctamente.")
            st.metric("MSE del ensamble", round(mse_ens, 4))

            # Sugerencia de auto-mejora
            history = get_performance_history()
            suggestion = suggest_half_life_adjustment(history)
            with st.expander("Sugerencia de auto-mejora", expanded=True):
                st.json(suggestion)
                if suggestion.get("half_life") and "reducir" in suggestion["half_life"]:
                    new_hl = max(60, st.session_state["half_life"] - 60)
                    st.info(f"Se sugiere reducir HALF_LIFE a {new_hl} días.")
                elif suggestion.get("half_life") and "aumentar" in suggestion["half_life"]:
                    new_hl = min(730, st.session_state["half_life"] + 60)
                    st.info(f"Se sugiere aumentar HALF_LIFE a {new_hl} días.")
        except Exception as exc:
            st.error(f"Error al cerrar partido: {exc}")


# ═══════════════════════════════════════════════════════════════════════════════
# PÁGINA: RENDIMIENTO
# ═══════════════════════════════════════════════════════════════════════════════

elif page == "Rendimiento":
    st.title("Historial de Rendimiento")

    history = get_performance_history()
    if not history:
        st.info("Aún no hay partidos cerrados. Realiza predicciones y ciérralos.")
        st.stop()

    df_perf = pd.DataFrame(history)
    st.dataframe(df_perf, use_container_width=True)

    if "model_error" in df_perf.columns:
        valid = df_perf["model_error"].dropna()
        if not valid.empty:
            col1, col2, col3 = st.columns(3)
            col1.metric("MSE Promedio", round(valid.mean(), 4))
            col2.metric("MSE Mínimo", round(valid.min(), 4))
            col3.metric("MSE Máximo", round(valid.max(), 4))

            import matplotlib.pyplot as plt
            fig, ax = plt.subplots(figsize=(8, 3))
            ax.plot(valid.values, marker="o", linewidth=1.5, color="#4e79a7")
            ax.axhline(valid.mean(), color="red", linestyle="--", linewidth=1, label=f"Media: {valid.mean():.3f}")
            ax.set_title("Evolución del MSE del Ensamble")
            ax.set_xlabel("Partido cerrado #")
            ax.set_ylabel("MSE")
            ax.legend()
            ax.grid(alpha=0.3)
            st.pyplot(fig)
            plt.close(fig)

    suggestion = suggest_half_life_adjustment(history)
    with st.expander("Análisis de auto-mejora"):
        st.json(suggestion)


