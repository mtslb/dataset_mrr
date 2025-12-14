from utils.seed import set_seed
from sklearn.linear_model import Lasso
from sklearn.model_selection import KFold
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from src.utils.paths import GRAPHS

def run_model(df=None, dataset_path: str = None, n_splits=5):
    """
    LassoRegression model pipeline (Log-transform pour y_members)
    """
    import src.domain.data_processing as dp
    import src.domain.evaluation as ev
    from sklearn.preprocessing import StandardScaler 

    set_seed(42)

    # Load & Prétraitement (Identique aux autres pour le Data Leakage)
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
    y_members_log = np.log1p(y["y_members"]) # LOG pour Lasso sur y_members

    # Modèles (alphas fixés)
    alpha_score = 0.01
    alpha_members = 0.1 
    model_score = Lasso(alpha=alpha_score, max_iter=50000)
    model_members = Lasso(alpha=alpha_members, max_iter=50000)

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
    print("\n===== Lasso - y_score (K-FOLD) =====")
    print(metrics_score_avg)

    # K-Fold y_members (LOG)
    metrics_members_list = []
    for train_idx, val_idx in kf.split(X_scaled):
        X_train, X_val = X_scaled[train_idx], X_scaled[val_idx]
        # Entraînement sur la target log-transformée
        model_members.fit(X_scaled[train_idx], y_members_log.iloc[train_idx]) 
        y_pred_log = model_members.predict(X_scaled[val_idx])
        y_pred = np.expm1(y_pred_log) # Inverse transformation
        y_val_real = y["y_members"].iloc[val_idx] # Évaluation sur la valeur réelle

        metrics_members_list.append(ev.evaluate_model(y_val_real, y_pred))
        all_y_true_members.append(y_val_real.to_numpy())
        all_y_pred_members.append(y_pred)

    metrics_members_avg = {k: np.mean([m[k] for m in metrics_members_list]) for k in metrics_members_list[0]}
    print("\n===== Lasso - y_members (LOG + K-FOLD) =====")
    print(metrics_members_avg)

    # ... (Le reste du code de plotting et de return) ...
    return metrics_score_avg, metrics_members_avg