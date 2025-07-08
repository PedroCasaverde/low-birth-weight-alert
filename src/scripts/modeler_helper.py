# 1. Future Imports
from __future__ import annotations

# 2. Standard Library Imports 
import logging
import re
import subprocess
from typing import Any
from warnings import simplefilter

import matplotlib.pyplot as plt

# 3. Third-Party Imports
import numpy as np
import pandas as pd
import seaborn as sns
from geopy.distance import great_circle
from imblearn.combine import SMOTETomek
from imblearn.ensemble import BalancedRandomForestClassifier
from imblearn.pipeline import Pipeline as ImbPipeline
from lightgbm import LGBMClassifier, early_stopping
from sklearn.base import clone
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    confusion_matrix,
    f1_score,
    matthews_corrcoef,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier, callback

# Ocultar advertencias
simplefilter("ignore", category=ConvergenceWarning)


#variables a eliminar
POST_BIRTH_PATTERNS: list[str] = [
    r"^bebe_",
    r"^edad_nino",
    r"^educacion_nino",
    r"^vacuna_",
    r"_bebe$",
    r"^dieta_nino",
    r"^dx_anemia_bebe",
    r"_cred$",
    r"_jarabe_",
    r"complicacion_",
    "lugar_parto",
    "parto_fue_cesarea",
    "consumio_micronutrientes_ult_7_dias",
    "tuvo_diarrea_ult_2_semanas",
    "tuvo_fiebre_ult_2_semanas",
    "tuvo_tos_ult_2_semanas",
    "tomo_suplemento_hierro",
    "hogar_beneficiario_beca18",
    "mujer_actualmente_embarazada",
]


def preparar_datos_bajo_peso(
    df: pd.DataFrame, test_size: float = 0.30, random_state: int = 42
) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    df = df.copy()

    # PASO 0: AISLAR LOS IDs PARA PROTEGERLOS DE TRANSFORMACIONES
    id_cols_a_preservar = [
        "id_hogar",
        "id_miembro_hogar",
        "id_cuestionario_mujer",
        "id_nacimiento",
    ]
    # Nos aseguramos de que las columnas existan en el DataFrame
    id_cols_a_preservar = [col for col in id_cols_a_preservar if col in df.columns]

    # Guardamos los IDs originales en un DataFrame separado
    df_ids = df[id_cols_a_preservar].copy()

    # Creamos un DataFrame de trabajo solo con las columnas a procesar
    df_procesar = df.drop(columns=id_cols_a_preservar, errors="ignore")

    # ───────── 1) Imputación (aplicada solo a df_procesar) ──────────
    for col in df_procesar.columns:
        serie = df_procesar[col]
        n_unique = serie.dropna().nunique()
        es_categ = (serie.dtype == "object") or (n_unique <= 9)

        if es_categ:
            modo = serie.mode(dropna=True)
            relleno = modo.iloc[0] if not modo.empty else 0
            df_procesar[col] = serie.fillna(relleno)
        else:
            mediana = serie.median(skipna=True)
            relleno = mediana if not np.isnan(mediana) else 0
            df_procesar[col] = serie.fillna(relleno)

    # ───────── 2) Codificación de texto (aplicada solo a df_procesar) ──
    for col in df_procesar.select_dtypes(include=["object"]).columns:
        s = df_procesar[col].astype(str).str.lower().str.strip()
        uniques = [u for u in s.unique() if u != "nan"]

        if set(uniques) <= {"si", "sí", "no", "0", "1"}:
            df_procesar[col] = s.map({"si": 1, "sí": 1, "no": 0, "1": 1, "0": 0}).astype(int)
        elif len(uniques) == 2:
            mapping = {val: idx for idx, val in enumerate(uniques)}
            df_procesar[col] = s.map(mapping).astype(int)
        else:
            df_procesar[col] = pd.Categorical(s).codes

    # ───────── 3) Variable objetivo bajo_peso ───────────────────────
    umbral = np.where(df_procesar["sexo_bebe"] == 1, 2500, 2400)
    df_procesar["bajo_peso"] = (df_procesar["peso_bebe_nacimiento_gr"] < umbral).astype(int)

    # ───────── 4) Quitar columnas post-nacimiento ───────────────────
    cols_a_eliminar = {
        c for patron in POST_BIRTH_PATTERNS for c in df_procesar.columns if re.search(patron, c)
    }
    df_procesar.drop(columns=cols_a_eliminar & set(df_procesar.columns), inplace=True)

    # PASO 5: RECOMBINAR IDs ORIGINALES CON DATOS PROCESADOS
    df_final = pd.concat([df_ids, df_procesar], axis=1)

    # ───────── 6) Split estratificado y eliminación de duplicados ────
    df_final["estrato"] = df_final["anio"].astype(str) + "_" + df_final["bajo_peso"].astype(str)

    print("Shape antes de eliminar duplicados:", df_final.shape)

    # Para la deduplicación, usamos todas las columnas EXCEPTO los IDs que preservamos
    cols_para_deduplicar = [col for col in df_final.columns if col not in id_cols_a_preservar]
    df_final = df_final.loc[~df_final[cols_para_deduplicar].duplicated(keep="first")].reset_index(
        drop=True
    )

    print("Shape después de eliminar duplicados:", df_final.shape)

    # `X` contendrá las características procesadas Y los IDs originales
    X = df_final.drop(columns=["bajo_peso", "estrato"])
    y = df_final["bajo_peso"]
    strat = df_final["estrato"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=strat
    )

    return X_train, X_test, y_train, y_test

def calcular_distancia(row):
    """
    Calcula la distancia great-circle en km y DEVUELVE UN ÚNICO NÚMERO.
    """
    try:
        # Define los puntos con formato (latitud, longitud)
        punto_origen = (row["latitud"], row["longitud"])
        punto_capital = (row["latitude"], row["longitude"])

        return great_circle(punto_origen, punto_capital).km

    except Exception as e:
        print(f"Error en una fila: {e}")



# ── logging ──────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)s  %(message)s",
    datefmt="%H:%M:%S",
)


# ── utilidades ───────────────────────────────────────────────────────────────
def _gpu_available() -> bool:
    """True si `nvidia-smi` detecta una GPU NVIDIA."""
    try:
        return (
            subprocess.run(
                ["nvidia-smi", "-L"],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                check=False,
            ).returncode
            == 0
        )
    except FileNotFoundError:
        return False


def _basic_metrics(y: np.ndarray, y_hat: np.ndarray, y_prob: np.ndarray) -> dict[str, float]:
    """Precision, Recall, Specificity, F1, ROC-AUC, PR-AUC, MCC."""
    tn, fp, fn, tp = confusion_matrix(y, y_hat).ravel()
    spec = tn / (tn + fp) if tn + fp else 0.0
    try:
        auc_roc = roc_auc_score(y, y_prob)
        auc_pr = average_precision_score(y, y_prob)
    except ValueError:  # ocurre si un fold no contiene ambas clases
        auc_roc = auc_pr = np.nan
    return {
        "Precision": precision_score(y, y_hat, zero_division=0),
        "Recall": recall_score(y, y_hat, zero_division=0),
        "Specificity": spec,
        "F1": f1_score(y, y_hat, zero_division=0),
        "AUC_ROC": auc_roc,
        "AUC_PR": auc_pr,
        "MCC": matthews_corrcoef(y, y_hat),
    }


def _select_threshold(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    metric: str = "f1",
) -> float:
    """Umbral que maximiza F1, recall o precision (según `metric`)."""
    prec, rec, thr = precision_recall_curve(y_true, y_prob)
    with np.errstate(divide="ignore", invalid="ignore"):
        f1 = np.where(prec + rec, 2 * prec * rec / (prec + rec), 0)

    metric_map = {"f1": f1, "recall": rec, "precision": prec}
    metric = metric.lower()
    if metric not in metric_map:
        raise ValueError(f"metric debe ser {list(metric_map)}")
    idx = int(np.argmax(metric_map[metric]))
    return float(thr[idx]) if idx < len(thr) else 0.5


# ── función principal ────────────────────────────────────────────────────────
def train_bajo_peso(
    X: pd.DataFrame,
    y: pd.Series,
    *,
    drop_cols: list[str] | None = None,
    random_state: int = 42,
    smote: bool = False,
    threshold_metric: str = "f1",
) -> tuple[Any, float, pd.DataFrame]:
    """Entrena 4 modelos y devuelve el mejor por AUC-PR."""
    drop_cols = drop_cols or []
    X = X.drop(columns=[c for c in drop_cols if c in X], errors="ignore")

    pos, neg = int(y.sum()), len(y) - int(y.sum())
    ratio = neg / max(pos, 1)
    scale_pos_weight = np.sqrt(ratio)
    logging.info("pos=%d  neg=%d  scale_pos_weight=%.2f", pos, neg, scale_pos_weight)

    use_gpu = _gpu_available()
    logging.info("GPU NVIDIA detectada: %s", use_gpu)

    # ── Steps de re-muestreo opcional ────────────────────────────────────────
    smt = SMOTETomek(sampling_strategy=0.2, random_state=random_state, n_jobs=-1)
    sampler_step = ("smt", smt) if smote else None

    # ── Constructores de modelos ─────────────────────────────────────────────
    def make_lgbm(**extra):
        return LGBMClassifier(
            objective="binary",
            device_type="gpu" if use_gpu else "cpu",
            n_estimators=300,
            learning_rate=0.1,
            num_leaves=24,
            min_child_samples=35,
            subsample=0.8,
            colsample_bytree=0.8,
            reg_alpha=0.5,  # solo reg_* para evitar warnings
            reg_lambda=10.0,
            is_unbalance=True,  # sustituye scale_pos_weight
            random_state=random_state,
            feature_fraction=0.7,
            n_jobs=-1,
            **extra,
        )

    def make_xgb(**extra):
        return XGBClassifier(
            eval_metric="aucpr",
            objective="binary:logistic",
            tree_method="hist",
            device="cuda" if use_gpu else "cpu",
            n_estimators=500,
            learning_rate=0.05,
            max_depth=4,
            min_child_weight=1,
            subsample=0.6,
            colsample_bytree=0.6,
            gamma=0,
            alpha=0.0,
            reg_lambda=10.0,
            scale_pos_weight=scale_pos_weight,
            random_state=random_state,
            n_jobs=-1,
            **extra,
        )

    # ── Diccionario de pipelines ────────────────────────────────────────────
    modelos: dict[str, ImbPipeline] = {
        "Balanced RF": ImbPipeline(
            steps=[
                *([sampler_step] if sampler_step else []),
                (
                    "clf",
                    BalancedRandomForestClassifier(
                        n_estimators=600,
                        max_depth=8,
                        min_samples_leaf=5,
                        bootstrap=False,
                        sampling_strategy="auto",  # explícito para v0.13
                        replacement=False,  # explícito para v0.13
                        random_state=random_state,
                        n_jobs=-1,
                    ),
                ),
            ]
        ),
        "LightGBM": ImbPipeline(
            steps=[
                *([sampler_step] if sampler_step else []),
                ("clf", make_lgbm()),
            ]
        ),
        "XGBoost": ImbPipeline(
            steps=[
                *([sampler_step] if sampler_step else []),
                ("clf", make_xgb()),
            ]
        ),
        "Regresión Logística": ImbPipeline(
            steps=[
                *([sampler_step] if sampler_step else []),
                ("sc", StandardScaler(with_mean=False)),
                (
                    "clf",
                    LogisticRegression(
                        solver="saga",
                        penalty="l1",
                        C=0.1,
                        max_iter=6000,
                        class_weight="balanced",
                        random_state=random_state,
                        n_jobs=-1,
                    ),
                ),
            ]
        ),
    }

    # ── Cross-validation ────────────────────────────────────────────────────
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=random_state)
    cv_results: dict[str, list[dict[str, float]]] = {n: [] for n in modelos}
    thresholds: dict[str, list[float]] = {n: [] for n in modelos}

    logging.info("Iniciando validación cruzada 5-fold…")
    for nombre, pipe_tpl in modelos.items():
        logging.info("• Modelo: %s", nombre)
        for tr_idx, val_idx in skf.split(X, y):
            X_tr, X_val = X.iloc[tr_idx], X.iloc[val_idx]
            y_tr, y_val = y.iloc[tr_idx], y.iloc[val_idx]

            pipe = clone(pipe_tpl)

            if nombre == "LightGBM":
                pipe.fit(
                    X_tr,
                    y_tr,
                    clf__eval_set=[(X_val, y_val)],
                    clf__eval_metric="aucpr",
                    clf__callbacks=[early_stopping(50, verbose=False)],
                )
            elif nombre == "XGBoost":
                pipe.set_params(
                    clf__callbacks=[callback.EarlyStopping(rounds=50, metric_name="aucpr")]
                )
                pipe.fit(X_tr, y_tr, clf__eval_set=[(X_val, y_val)], clf__verbose=False)
            else:
                pipe.fit(X_tr, y_tr)

            y_prob = pipe.predict_proba(X_val)[:, 1]
            thr = _select_threshold(y_val.to_numpy(), y_prob, threshold_metric)
            y_hat = (y_prob >= thr).astype(int)

            cv_results[nombre].append(_basic_metrics(y_val, y_hat, y_prob))
            thresholds[nombre].append(thr)

    # ── Consolidar resultados ───────────────────────────────────────────────
    metrics_df = (
        pd.concat({m: pd.DataFrame(lst) for m, lst in cv_results.items()})
        .groupby(level=0)
        .mean()
        .reset_index()
        .rename(columns={"index": "modelo"})
    )

    pct_cols = ["Precision", "Recall", "Specificity", "F1", "AUC_ROC", "AUC_PR"]
    metrics_df[pct_cols] = metrics_df[pct_cols].apply(
        lambda col: (col * 100).round(2).astype(str) + "%"
    )

    best_model_name = (
        metrics_df.assign(aucpr_num=lambda d: d["AUC_PR"].str.rstrip("%").astype(float))
        .sort_values("aucpr_num", ascending=False)
        .iloc[0]["modelo"]
    )
    logging.info("Mejor modelo según AUC-PR: %s", best_model_name)

    # ── Re-entrenar ganador sobre TODO el dataset ──────────────────────────
    best_pipe = clone(modelos[best_model_name])
    if hasattr(best_pipe.named_steps["clf"], "set_params"):
        best_pipe.set_params(clf__callbacks=None)
    best_pipe.fit(X, y)

    best_thresh = float(np.mean(thresholds[best_model_name]))
    logging.info("Umbral final óptimo (promedio CV): %.4f", best_thresh)

    return best_pipe, best_thresh, metrics_df
