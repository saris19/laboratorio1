"""
Interfaz web (Streamlit) para predecir con los modelos de regresión lineal múltiple.

Ejecutar:  streamlit run app.py
Requiere haber ejecutado antes:  python entrenamiento_modelos.py
"""
from pathlib import Path

import joblib
import pandas as pd
import streamlit as st

MODELS_DIR = Path(__file__).parent / "modelos"

# Etiquetas legibles y configuración de entrada por variable
FIELD_CONFIG = {
    "Dia": {"label": "Día (número de día)", "step": 1, "integer": True},
    "Inflacion": {"label": "Inflación diaria (ej. 0.02 = 2 %)", "step": 0.001, "format": "%.4f"},
    "Tasa_interes": {"label": "Tasa de interés diaria (%)", "step": 0.01, "format": "%.2f"},
    "Edad": {"label": "Edad (años)", "step": 1, "integer": True},
    "IMC": {"label": "Índice de masa corporal (IMC)", "step": 0.1, "format": "%.1f"},
    "Actividad_Fisica": {"label": "Actividad física (horas/semana)", "step": 1, "integer": True},
    "Temperatura": {"label": "Temperatura (°C)", "step": 0.1, "format": "%.1f"},
    "Hora": {"label": "Hora del día (1 a 24)", "step": 1, "integer": True, "min": 1, "max": 24},
    "Dia_Semana": {"label": "Día de la semana (1 = Lunes … 7 = Domingo)", "step": 1, "integer": True, "min": 1, "max": 7},
}

EXERCISE_OPTIONS = {
    "Dólar": "dolar",
    "Glucosa": "glucosa",
    "Energía": "energia",
}


@st.cache_resource
def load_bundle(key: str) -> dict:
    """Carga el modelo exportado (.joblib) junto con sus metadatos."""
    return joblib.load(MODELS_DIR / f"modelo_{key}.joblib")


st.set_page_config(page_title="Predicción con Regresión Lineal", page_icon="📈", layout="centered")
st.title("Predicción con regresión lineal múltiple")
st.caption("Laboratorio 1 · Minería de Datos (CRISP-DM)")

choice = st.radio("Selecciona el ejercicio", list(EXERCISE_OPTIONS), horizontal=True)
bundle = load_bundle(EXERCISE_OPTIONS[choice])
model = bundle["model"]

st.subheader(f"{bundle['title']}")
st.write(f"Variable a predecir: **{bundle['target']}** ({bundle['unit']})")

with st.form("prediction_form"):
    inputs = {}
    cols = st.columns(len(bundle["features"]))
    for col, feat in zip(cols, bundle["features"]):
        cfg = FIELD_CONFIG[feat]
        vmin, vmax, vmean = bundle["ranges"][feat]
        with col:
            if cfg.get("integer"):
                inputs[feat] = st.number_input(
                    cfg["label"],
                    min_value=cfg.get("min"),
                    max_value=cfg.get("max"),
                    value=int(round(vmean)),
                    step=cfg["step"],
                )
            else:
                inputs[feat] = st.number_input(
                    cfg["label"], value=float(round(vmean, 4)), step=cfg["step"], format=cfg["format"]
                )
            st.caption(f"Rango en datos: {vmin:g} – {vmax:g}")
    submitted = st.form_submit_button("Predecir", type="primary")

if submitted:
    X_new = pd.DataFrame([inputs], columns=bundle["features"])
    prediction = float(model.predict(X_new)[0])
    st.success(f"**{bundle['target']} estimado: {prediction:,.2f} {bundle['unit']}**")

    # Aviso si algún valor está fuera del rango de entrenamiento (extrapolación)
    out_of_range = [
        f for f in bundle["features"]
        if not (bundle["ranges"][f][0] <= inputs[f] <= bundle["ranges"][f][1])
    ]
    if out_of_range:
        st.warning(
            "Valores fuera del rango de entrenamiento: " + ", ".join(out_of_range)
            + ". La predicción es una extrapolación y puede ser menos confiable."
        )

with st.expander("Ver detalles del modelo"):
    m = bundle["metrics"]
    c1, c2, c3 = st.columns(3)
    c1.metric("R² (prueba)", f"{m['r2_test']:.4f}")
    c2.metric("MSE (prueba)", f"{m['mse_test']:,.2f}")
    c3.metric("RMSE (prueba)", f"{m['rmse_test']:,.2f}")
    coef_table = pd.DataFrame(
        {"Variable": bundle["features"], "Coeficiente": model.coef_}
    )
    st.write(f"Intercepto: **{model.intercept_:.4f}**")
    st.dataframe(coef_table, hide_index=True, use_container_width=True)
    equation = " + ".join(f"({c:.4f} × {f})" for f, c in zip(bundle["features"], model.coef_))
    st.code(f"{bundle['target']} = {model.intercept_:.4f} + {equation}", language="text")
