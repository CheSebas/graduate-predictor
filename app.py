# app.py — Demo: FT-Transformer + MLP — Graduate Prediction
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from model_utils import (
    prepare_data,
    train_ft_transformer,
    train_mlp,
    evaluate,
    CATEGORICAL_COLS, CONTINUOUS_COLS, DEVICE
)

# ── Página ────────────────────────────────────────────────────
st.set_page_config(page_title="Graduate Predictor · FT-Transformer", layout="wide")

st.markdown("""
<style>
    .metric-card {
        background: #1e1e2e; border-radius: 10px;
        padding: 1rem; text-align: center;
    }
    .stProgress > div > div { background-color: #7B2D8B; }
</style>
""", unsafe_allow_html=True)

st.title("🎓 Graduate Predictor — FT-Transformer vs MLP")
st.caption(f"Dispositivo: **{DEVICE}** · Dataset: `ieee_online_learning_balanced.csv`")

# ── Sidebar ───────────────────────────────────────────────────
with st.sidebar:
    st.header("⚙️ Configuración")
    model_choice = st.selectbox(
        "Modelo", ["FT-Transformer", "MLP Base", "Comparar ambos"]
    )

    st.markdown("**FT-Transformer**")
    embed_dim  = st.select_slider("Embed dim",  [16, 32, 64], value=32)
    num_heads  = st.select_slider("Num heads",  [2, 4, 8],    value=8)
    num_blocks = st.select_slider("Num bloques",[1, 2, 4],    value=4)
    lr_ft      = st.number_input("Learning rate FT", value=1e-3, format="%.4f")
    epochs_ft  = st.slider("Épocas FT", 10, 150, 50)
    patience_ft= st.slider("Patience FT", 5, 30, 10)
    batch_ft   = st.selectbox("Batch FT", [128, 256, 512], index=1)

    st.markdown("**MLP**")
    epochs_mlp  = st.slider("Épocas MLP", 20, 200, 100)
    patience_mlp= st.slider("Patience MLP", 5, 30, 10)
    batch_mlp   = st.selectbox("Batch MLP", [32, 64, 128], index=0)

# ── Upload ────────────────────────────────────────────────────
uploaded = st.file_uploader(
    "📂 Subir CSV (`ieee_online_learning_balanced.csv`)", type=["csv"]
)

if not uploaded:
    st.info("Sube el archivo CSV para comenzar.")
    st.stop()

# ── Cargar y explorar ─────────────────────────────────────────
df_raw = pd.read_csv(uploaded)
st.success(f"Dataset cargado: **{df_raw.shape[0]:,}** filas × **{df_raw.shape[1]}** columnas")

with st.expander("🔍 Vista previa del dataset"):
    st.dataframe(df_raw.head(10), use_container_width=True)
    c1, c2 = st.columns(2)
    with c1:
        st.markdown("**Tipos de datos**")
        st.dataframe(df_raw.dtypes.rename("dtype").reset_index())
    with c2:
        st.markdown("**Nulos / Duplicados**")
        st.write(f"Nulos: {df_raw.isnull().sum().sum()}")
        st.write(f"Duplicados: {df_raw.duplicated().sum()}")

# ── EDA ───────────────────────────────────────────────────────
st.subheader("📊 Análisis Exploratorio")
col_a, col_b = st.columns(2)

with col_a:
    fig, ax = plt.subplots(figsize=(5, 3))
    vc = df_raw["graduate"].value_counts()
    colors = ["#7B2D8B", "#2D7B8B"]
    ax.bar(vc.index.astype(str), vc.values, color=colors, edgecolor="white", width=0.5)
    ax.set_title("Distribución de `graduate`", fontsize=12)
    ax.set_xlabel("graduate"); ax.set_ylabel("Frecuencia")
    for i, v in enumerate(vc.values):
        ax.text(i, v + 5, str(v), ha="center", fontweight="bold")
    st.pyplot(fig); plt.close()

with col_b:
    cont_present = [c for c in CONTINUOUS_COLS if c in df_raw.columns]
    fig, ax = plt.subplots(figsize=(5, 3))
    df_raw[cont_present].boxplot(ax=ax, patch_artist=True)
    ax.set_title("Distribución de variables continuas", fontsize=12)
    plt.xticks(rotation=30, ha="right")
    st.pyplot(fig); plt.close()

# Correlación continuas vs target
fig, ax = plt.subplots(figsize=(10, 3))
corr_data = df_raw[cont_present + ["graduate"]].corr()["graduate"].drop("graduate").sort_values()
bars = ax.barh(corr_data.index, corr_data.values,
               color=["#7B2D8B" if v >= 0 else "#2D7B8B" for v in corr_data.values])
ax.axvline(0, color="white", lw=0.8)
ax.set_title("Correlación de variables continuas con `graduate`", fontsize=12)
st.pyplot(fig); plt.close()

# ── Preparar datos ────────────────────────────────────────────
st.subheader("🔧 Preprocesamiento")
with st.spinner("Preparando datos (LabelEncode + Split 70/15/15 + Escalado)..."):
    data = prepare_data(df_raw.copy())

c1, c2, c3 = st.columns(3)
c1.metric("Train",      f"{len(data['X_train']):,} muestras")
c2.metric("Validación", f"{len(data['X_val']):,} muestras")
c3.metric("Test",       f"{len(data['X_test']):,} muestras")

# ── Entrenamiento ─────────────────────────────────────────────
st.subheader("🚀 Entrenamiento")

if st.button("▶ Iniciar entrenamiento", use_container_width=True):

    results = {}

    def run_model(name, train_fn, kwargs, model_type):
        st.markdown(f"#### Modelo: **{name}**")
        prog_bar  = st.progress(0)
        stat_cols = st.columns(3)
        ep_text   = stat_cols[0].empty()
        tl_text   = stat_cols[1].empty()
        vl_text   = stat_cols[2].empty()

        total_epochs = kwargs.get("epochs", 50)

        def cb(ep, tl, vl, vacc):
            frac = min(ep / total_epochs, 1.0)
            prog_bar.progress(frac)
            ep_text.metric("Época", ep)
            tl_text.metric("Train Loss", f"{tl:.4f}")
            vl_text.metric("Val Loss",   f"{vl:.4f}")

        model, history = train_fn(data, progress_cb=cb, **kwargs)
        prog_bar.progress(1.0)

        metrics, cm, curves = evaluate(model, data, model_type=model_type)
        results[name] = {
            "model": model, "history": history,
            "metrics": metrics, "cm": cm, "curves": curves,
            "model_type": model_type,
        }

        # ── Curvas de entrenamiento ──
        fig, axes = plt.subplots(1, 2, figsize=(12, 4))
        axes[0].plot(history["train_loss"], label="Train", color="#7B2D8B")
        axes[0].plot(history["val_loss"],   label="Val",   color="#2D7B8B")
        axes[0].set_title("Loss vs Epochs"); axes[0].legend()
        axes[0].set_xlabel("Epoch"); axes[0].set_ylabel("Loss")

        axes[1].plot(history["val_acc"], label="Val Accuracy", color="#7B2D8B")
        axes[1].set_title("Val Accuracy vs Epochs")
        axes[1].set_xlabel("Epoch"); axes[1].set_ylabel("Accuracy")
        plt.tight_layout()
        st.pyplot(fig); plt.close()

        # ── Métricas ──
        st.markdown("**Métricas sobre Test:**")
        mc = st.columns(5)
        for i, (k, v) in enumerate(metrics.items()):
            mc[i].metric(k, f"{v:.4f}")

        # ── Confusion Matrix ──
        fig, ax = plt.subplots(figsize=(4, 3))
        cmap = "Purples" if "Transformer" in name else "Blues"
        sns.heatmap(cm, annot=True, fmt="d", cmap=cmap, ax=ax)
        ax.set_title(f"Confusion Matrix — {name}")
        ax.set_xlabel("Predicted"); ax.set_ylabel("Actual")
        st.pyplot(fig); plt.close()

        # ── ROC + PR Curves ──
        fig, axes = plt.subplots(1, 2, figsize=(13, 4))
        c = curves

        axes[0].plot(c["fpr"], c["tpr"], lw=2, color="#7B2D8B",
                     label=f"AUC-ROC = {c['roc_auc']:.4f}")
        axes[0].plot([0,1],[0,1], "--", color="gray", label="Random")
        axes[0].fill_between(c["fpr"], c["tpr"], alpha=0.08, color="#7B2D8B")
        axes[0].set_title("ROC Curve"); axes[0].legend(loc="lower right")
        axes[0].set_xlabel("False Positive Rate"); axes[0].set_ylabel("True Positive Rate")
        axes[0].grid(True, linestyle="--", alpha=0.4)

        axes[1].plot(c["rec"], c["prec"], lw=2, color="#2D7B8B",
                     label=f"AP = {c['avg_prec']:.4f}")
        axes[1].axhline(c["prevalence"], color="gray", linestyle="--",
                        label=f"Baseline = {c['prevalence']:.2f}")
        axes[1].fill_between(c["rec"], c["prec"], alpha=0.08, color="#2D7B8B")
        axes[1].set_title("Precision-Recall Curve"); axes[1].legend(loc="upper right")
        axes[1].set_xlabel("Recall"); axes[1].set_ylabel("Precision")
        axes[1].grid(True, linestyle="--", alpha=0.4)

        plt.suptitle(f"Curvas de Validación — {name}", fontweight="bold", y=1.02)
        plt.tight_layout()
        st.pyplot(fig); plt.close()

    # ── Selección de modelos ──
    if model_choice in ("FT-Transformer", "Comparar ambos"):
        run_model(
            "FT-Transformer",
            train_ft_transformer,
            dict(epochs=epochs_ft, batch_size=batch_ft, embed_dim=embed_dim,
                 num_heads=num_heads, num_blocks=num_blocks,
                 lr=lr_ft, patience=patience_ft),
            model_type="ft",
        )

    if model_choice in ("MLP Base", "Comparar ambos"):
        run_model(
            "MLP Base",
            train_mlp,
            dict(epochs=epochs_mlp, batch_size=batch_mlp, patience=patience_mlp),
            model_type="mlp",
        )

    # ── Tabla comparativa ──
    if len(results) > 1:
        st.markdown("---")
        st.subheader("📋 Comparación de Modelos")
        rows = []
        for name, r in results.items():
            row = {"Modelo": name}
            row.update({k: f"{v:.4f}" for k, v in r["metrics"].items()})
            rows.append(row)
        st.dataframe(pd.DataFrame(rows).set_index("Modelo"), use_container_width=True)

    st.success("✅ Entrenamiento completado.")