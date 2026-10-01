"""
Laboratorio 1 - Minería de Datos (CRISP-DM)
Regresión lineal múltiple para tres casos: Dólar, Glucosa y Energía.

Ejecutar:  python entrenamiento_modelos.py
Genera:    modelos/*.joblib, graficas/*.png, resultados/metricas.json
"""
import json
from pathlib import Path

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import statsmodels.api as sm
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split

BASE_DIR = Path(__file__).parent
DATA_DIR = BASE_DIR / "data"
MODELS_DIR = BASE_DIR / "modelos"
PLOTS_DIR = BASE_DIR / "graficas"
RESULTS_DIR = BASE_DIR / "resultados"
for folder in (MODELS_DIR, PLOTS_DIR, RESULTS_DIR):
    folder.mkdir(exist_ok=True)

RANDOM_STATE = 42
TEST_SIZE = 0.20

# Estilo de gráficas
POINT_COLOR = "#2a78d6"
LINE_COLOR = "#eb6834"
TEXT_COLOR = "#52514e"
plt.rcParams.update({
    "figure.facecolor": "#fcfcfb", "axes.facecolor": "#fcfcfb",
    "axes.edgecolor": "#c3c2b7", "axes.labelcolor": TEXT_COLOR,
    "xtick.color": TEXT_COLOR, "ytick.color": TEXT_COLOR,
    "axes.grid": True, "axes.axisbelow": True, "grid.color": "#e6e5e0", "grid.linewidth": 0.6,
    "axes.spines.top": False, "axes.spines.right": False,
    "font.size": 10, "axes.titlesize": 11, "axes.titleweight": "bold",
})

# Configuración de cada ejercicio
EXERCISES = {
    "dolar": {
        "title": "Precio del dólar",
        "file": "dolar_data.csv",
        "features": ["Dia", "Inflacion", "Tasa_interes"],
        "target": "Precio_Dolar",
        "unit": "COP",
    },
    "glucosa": {
        "title": "Nivel de glucosa",
        "file": "glucosa_data.csv",
        "features": ["Edad", "IMC", "Actividad_Fisica"],
        "target": "Nivel_Glucosa",
        "unit": "mg/dL",
    },
    "energia": {
        "title": "Consumo de energía",
        "file": "energia_data.csv",
        "features": ["Temperatura", "Hora", "Dia_Semana"],
        "target": "Consumo_Energia",
        "unit": "kWh",
    },
}


def understand_data(df: pd.DataFrame, name: str) -> dict:
    """Fase 2 CRISP-DM: comprensión de los datos."""
    print(f"\n--- Comprensión de datos: {name} ---")
    print(f"Registros: {df.shape[0]} | Columnas: {df.shape[1]}")
    print(f"Valores nulos: {int(df.isna().sum().sum())} | Duplicados: {int(df.duplicated().sum())}")
    print(df.describe().round(3))
    return {
        "rows": int(df.shape[0]),
        "nulls": int(df.isna().sum().sum()),
        "duplicates": int(df.duplicated().sum()),
        "describe": df.describe().round(4).to_dict(),
        "correlation": df.corr().round(4).to_dict(),
    }


def prepare_data(df: pd.DataFrame) -> pd.DataFrame:
    """Fase 3 CRISP-DM: preparación (limpieza básica)."""
    return df.dropna().drop_duplicates().reset_index(drop=True)


def train_and_evaluate(df: pd.DataFrame, features: list, target: str) -> dict:
    """Fases 4 y 5 CRISP-DM: modelado y evaluación."""
    X = df[features]
    y = df[target]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE
    )

    model = LinearRegression()
    model.fit(X_train, y_train)

    y_pred_train = model.predict(X_train)
    y_pred_test = model.predict(X_test)

    mse_test = mean_squared_error(y_test, y_pred_test)
    metrics = {
        "mse_train": float(mean_squared_error(y_train, y_pred_train)),
        "mse_test": float(mse_test),
        "rmse_test": float(np.sqrt(mse_test)),
        "r2_train": float(r2_score(y_train, y_pred_train)),
        "r2_test": float(r2_score(y_test, y_pred_test)),
        "n_train": int(len(X_train)),
        "n_test": int(len(X_test)),
    }

    # Coeficientes estandarizados: coef * desviación(X) / desviación(y).
    # Permiten comparar el impacto de variables con escalas distintas.
    std_coefs = model.coef_ * X.std().values / y.std()
    coefficients = {
        feat: {"coef": float(c), "std_coef": float(s)}
        for feat, c, s in zip(features, model.coef_, std_coefs)
    }

    # Significancia estadística (p-valores) de cada coeficiente con statsmodels (OLS)
    ols = sm.OLS(y_train, sm.add_constant(X_train)).fit()
    for feat in features:
        coefficients[feat]["p_value"] = float(ols.pvalues[feat])

    return {
        "model": model,
        "intercept": float(model.intercept_),
        "coefficients": coefficients,
        "metrics": metrics,
        "y_test": y_test.values,
        "y_pred_test": y_pred_test,
    }


def plot_relationships(df, features, target, key, title):
    """Dispersión de cada variable independiente vs. la dependiente con recta de tendencia."""
    fig, axes = plt.subplots(1, len(features), figsize=(5 * len(features), 4))
    for ax, feat in zip(axes, features):
        ax.scatter(df[feat], df[target], s=10, alpha=0.35, color=POINT_COLOR, edgecolors="none")
        slope, intercept = np.polyfit(df[feat], df[target], 1)
        x_line = np.linspace(df[feat].min(), df[feat].max(), 100)
        ax.plot(x_line, slope * x_line + intercept, color=LINE_COLOR, linewidth=2)
        r = df[feat].corr(df[target])
        ax.set_title(f"{feat} vs {target}  (r = {r:.3f})")
        ax.set_xlabel(feat)
        ax.set_ylabel(target)
    fig.suptitle(f"{title}: relación de cada variable con {target}", fontweight="bold", color="#0b0b0b")
    fig.tight_layout()
    path = PLOTS_DIR / f"{key}_relaciones.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


def plot_correlation(df, key, title):
    """Matriz de correlación (escala divergente azul-gris-naranja)."""
    from matplotlib.colors import LinearSegmentedColormap
    cmap = LinearSegmentedColormap.from_list("div", ["#2a78d6", "#f0efec", "#eb6834"])
    corr = df.corr()
    fig, ax = plt.subplots(figsize=(6, 5))
    im = ax.imshow(corr, cmap=cmap, vmin=-1, vmax=1)
    ax.set_xticks(range(len(corr)), corr.columns, rotation=30, ha="right")
    ax.set_yticks(range(len(corr)), corr.columns)
    ax.grid(False)
    for i in range(len(corr)):
        for j in range(len(corr)):
            ax.text(j, i, f"{corr.iloc[i, j]:.2f}", ha="center", va="center", color="#0b0b0b", fontsize=9)
    fig.colorbar(im, ax=ax, fraction=0.046)
    ax.set_title(f"{title}: matriz de correlación")
    fig.tight_layout()
    path = PLOTS_DIR / f"{key}_correlacion.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


def plot_real_vs_predicted(y_test, y_pred, key, title, unit):
    """Valores reales vs. predichos en el conjunto de prueba."""
    fig, ax = plt.subplots(figsize=(5.5, 5))
    ax.scatter(y_test, y_pred, s=12, alpha=0.4, color=POINT_COLOR, edgecolors="none")
    lims = [min(y_test.min(), y_pred.min()), max(y_test.max(), y_pred.max())]
    ax.plot(lims, lims, color=LINE_COLOR, linewidth=2, label="Predicción perfecta")
    ax.set_xlabel(f"Valor real ({unit})")
    ax.set_ylabel(f"Valor predicho ({unit})")
    ax.set_title(f"{title}: real vs. predicho (prueba)")
    ax.legend(frameon=False)
    fig.tight_layout()
    path = PLOTS_DIR / f"{key}_real_vs_predicho.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


def plot_importance(coefficients, key, title):
    """Impacto relativo: |coeficiente estandarizado| por variable."""
    names = list(coefficients)
    values = [abs(coefficients[n]["std_coef"]) for n in names]
    order = np.argsort(values)
    fig, ax = plt.subplots(figsize=(6, 3))
    ax.barh([names[i] for i in order], [values[i] for i in order], color=POINT_COLOR, height=0.5)
    for i, idx in enumerate(order):
        ax.text(values[idx], i, f"  {values[idx]:.3f}", va="center", color=TEXT_COLOR, fontsize=9)
    ax.set_xlabel("|Coeficiente estandarizado|")
    ax.set_title(f"{title}: impacto de cada variable")
    ax.grid(axis="y", visible=False)
    fig.tight_layout()
    path = PLOTS_DIR / f"{key}_importancia.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


def main():
    summary = {}
    for key, cfg in EXERCISES.items():
        df_raw = pd.read_csv(DATA_DIR / cfg["file"])
        data_info = understand_data(df_raw, cfg["title"])
        df = prepare_data(df_raw)

        result = train_and_evaluate(df, cfg["features"], cfg["target"])

        # Gráficas
        plot_relationships(df, cfg["features"], cfg["target"], key, cfg["title"])
        plot_correlation(df, key, cfg["title"])
        plot_real_vs_predicted(result["y_test"], result["y_pred_test"], key, cfg["title"], cfg["unit"])
        plot_importance(result["coefficients"], key, cfg["title"])

        # Fase 6 CRISP-DM: despliegue -> exportar modelo con metadatos para la app
        bundle = {
            "model": result["model"],
            "features": cfg["features"],
            "target": cfg["target"],
            "title": cfg["title"],
            "unit": cfg["unit"],
            "metrics": result["metrics"],
            "ranges": {f: [float(df[f].min()), float(df[f].max()), float(df[f].mean())] for f in cfg["features"]},
        }
        joblib.dump(bundle, MODELS_DIR / f"modelo_{key}.joblib")

        # Verificación: cargar el modelo exportado y comparar predicciones
        loaded = joblib.load(MODELS_DIR / f"modelo_{key}.joblib")
        sample = df[cfg["features"]].head(5)
        assert np.allclose(loaded["model"].predict(sample), result["model"].predict(sample))

        summary[key] = {
            "title": cfg["title"],
            "target": cfg["target"],
            "intercept": result["intercept"],
            "coefficients": result["coefficients"],
            "metrics": result["metrics"],
            "data": data_info,
        }

        print(f"\n=== Modelo: {cfg['title']} ===")
        print(f"Intercepto: {result['intercept']:.4f}")
        for feat, c in result["coefficients"].items():
            print(f"  {feat:18s} coef = {c['coef']:12.4f} | estandarizado = {c['std_coef']:.4f} | p = {c['p_value']:.4f}")
        m = result["metrics"]
        print(f"MSE prueba = {m['mse_test']:.4f} | RMSE = {m['rmse_test']:.4f} | R² prueba = {m['r2_test']:.4f}")

    with open(RESULTS_DIR / "metricas.json", "w", encoding="utf-8") as fh:
        json.dump(summary, fh, indent=2, ensure_ascii=False)
    print("\nModelos guardados en /modelos, gráficas en /graficas, métricas en /resultados.")


if __name__ == "__main__":
    main()
