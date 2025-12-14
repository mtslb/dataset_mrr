from utils.seed import set_seed
from sklearn.linear_model import Ridge
from sklearn.model_selection import KFold
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from src.utils.paths import GRAPHS

def run_model(df=None, dataset_path: str = None, n_splits=5):
    """
    RidgeRegression model pipeline (alpha=1.0)
    """
    import src.domain.data_processing as dp
    import src.domain.evaluation as ev
    from sklearn.preprocessing import StandardScaler 

    set_seed(42)

    # Load & Prétraitement (Identique à Linear/ElasticNet pour le Data Leakage)
    if df is None:
        if dataset_path is None:
            raise ValueError("Il faut fournir df ou dataset_path")
        df = dp.load_data(dataset_path)

    X, y = dp.split_features_targets(df)
    X = X.apply(pd.to_numeric, errors="coerce")
    X_imputed = X.fillna(X.mean())

    variances = X_imputed.var()
    cols_to_drop = variances[variances < 1e-6].index
    X_final = X_imputed.drop(columns=cols_to_drop)
    
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_final)

    # Targets
    y_score = y["y_score"]
    y_members = y["y_members"] # PAS DE LOG pour Ridge

    # Modèles (alpha fixé)
    alpha_val = 1.0 
    model_score = Ridge(alpha=alpha_val)
    model_members = Ridge(alpha=alpha_val)

    kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)

    # Collect preds
    all_y_true_score, all_y_pred_score = [], []
    all_y_true_members, all_y_pred_members = [], []

    # K-Fold y_score
    metrics_score_list = []
    for train_idx, val_idx in kf.split(X_scaled):
        X_train, X_val = X_scaled[train_idx], X_scaled[val_idx]
        y_train, y_val = y_score.iloc[train_idx], y_score.iloc[val_idx]
        model_score.fit(X_train, y_train)
        y_pred = model_score.predict(X_val)
        metrics_score_list.append(ev.evaluate_model(y_val, y_pred))
        all_y_true_score.append(y_val.to_numpy())
        all_y_pred_score.append(y_pred)

    metrics_score_avg = {k: np.mean([m[k] for m in metrics_score_list]) for k in metrics_score_list[0]}
    print("\n===== Ridge - y_score (K-FOLD) =====")
    print(metrics_score_avg)

    # K-Fold y_members (PAS DE LOG)
    metrics_members_list = []
    for train_idx, val_idx in kf.split(X_scaled):
        X_train, X_val = X_scaled[train_idx], X_scaled[val_idx]
        y_train, y_val = y_members.iloc[train_idx], y_members.iloc[val_idx]
        model_members.fit(X_train, y_train)
        y_pred = model_members.predict(X_val)
        metrics_members_list.append(ev.evaluate_model(y_val, y_pred))
        all_y_true_members.append(y_val.to_numpy())
        all_y_pred_members.append(y_pred)

    metrics_members_avg = {k: np.mean([m[k] for m in metrics_members_list]) for k in metrics_members_list[0]}
    print("\n===== Ridge - y_members (K-FOLD) =====")
    print(metrics_members_avg)

    # ... (Le reste du code de plotting et de return) ...
    return metrics_score_avg, metrics_members_avg