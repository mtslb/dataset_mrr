from utils.seed import set_seed
from sklearn.linear_model import ElasticNet
from sklearn.model_selection import KFold
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from src.utils.paths import GRAPHS
# Assurez-vous que dp.StandardScaler est accessible

def run_model(df=None, dataset_path: str = None, n_splits=5):
    """
    ElasticNet pipeline corrigé pour éviter le Data Leakage en effectuant
    l'imputation et le scaling DANS la boucle K-Fold.
    """
    import src.domain.data_processing as dp
    import src.domain.evaluation as ev
    # Nécessite un accès direct à sklearn.preprocessing.StandardScaler si dp.StandardScaler n'est pas disponible.
    from sklearn.preprocessing import StandardScaler 

    set_seed(42)

    # Load (inchangé)
    if df is None:
        if dataset_path is None:
            raise ValueError("Il faut fournir df ou dataset_path")
        df = dp.load_data(dataset_path)

    X, y = dp.split_features_targets(df)
    if not isinstance(X, pd.DataFrame):
        X = pd.DataFrame(X)

    # --- ÉTAPE CRITIQUE : Préparation du DataFrame BRUT (NON SCALÉ/NON IMPUTÉ) ---
    # Conversions numériques pour traiter les erreurs/NaN
    X_raw = X.apply(pd.to_numeric, errors="coerce")
    
    # Suppression des features constantes basée sur l'ensemble complet (pour la cohérence des colonnes)
    # Note: On doit le faire avant KFold pour que l'indexation de y corresponde.
    variances = X_raw.var()
    cols_to_drop = variances[variances < 1e-6].index
    X_raw = X_raw.drop(columns=cols_to_drop)
    if len(cols_to_drop) > 0:
        print(f"⚠️ {len(cols_to_drop)} features constantes supprimées : {list(cols_to_drop)}")
    # X_raw est maintenant notre base de travail non imputée/non scalée.

    # Targets (inchangé)
    y_score = y["y_score"]
    y_members_log = np.log1p(y["y_members"])

    model_score = ElasticNet(alpha=0.01, l1_ratio=0.5, max_iter=50000)
    model_members = ElasticNet(alpha=0.1, l1_ratio=0.5, max_iter=50000)

    kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)

    # Collect preds (inchangé)
    all_y_true_score, all_y_pred_score = [], []
    all_y_true_members, all_y_pred_members = [], []

    scaler = StandardScaler()

    # K-Fold y_score
    metrics_score_list = []
    for train_idx, val_idx in kf.split(X_raw): # Split de X_raw (le DataFrame brut)
        X_train_raw, X_val_raw = X_raw.iloc[train_idx], X_raw.iloc[val_idx]
        y_train = y_score.iloc[train_idx]
        y_val_real = y_score.iloc[val_idx] # y_score est la target réelle ici

        # --- CORRECTION ANTI-LEAKAGE : Imputation et Scaling DANS la boucle ---
        
        # 1. Imputation (Calculée sur TRAIN UNIQUEMENT)
        train_mean = X_train_raw.mean()
        X_train_imputed = X_train_raw.fillna(train_mean)
        X_val_imputed = X_val_raw.fillna(train_mean) # Utilisation de la moyenne d'entraînement

        # 2. Scaling (Fit sur TRAIN, Transform sur TRAIN/VAL)
        X_train_scaled = scaler.fit_transform(X_train_imputed)
        X_val_scaled = scaler.transform(X_val_imputed)
        
        # Entraînement et prédiction
        model_score.fit(X_train_scaled, y_train)
        y_pred = model_score.predict(X_val_scaled)

        metrics_score_list.append(ev.evaluate_model(y_val_real, y_pred))
        all_y_true_score.append(y_val_real.to_numpy())
        all_y_pred_score.append(y_pred)

    metrics_score_avg = {k: np.mean([m[k] for m in metrics_score_list]) for k in metrics_score_list[0]}
    print("\n===== ElasticNet - y_score (K-FOLD, LEAKAGE-FREE) =====")
    print(metrics_score_avg)


    # K-Fold y_members (log)
    metrics_members_list = []
    for train_idx, val_idx in kf.split(X_raw): # Split de X_raw (le DataFrame brut)
        X_train_raw, X_val_raw = X_raw.iloc[train_idx], X_raw.iloc[val_idx]
        y_train_log = y_members_log.iloc[train_idx]
        y_val_real = y["y_members"].iloc[val_idx]
        
        # --- CORRECTION ANTI-LEAKAGE : Imputation et Scaling DANS la boucle ---
        train_mean = X_train_raw.mean()
        X_train_imputed = X_train_raw.fillna(train_mean)
        X_val_imputed = X_val_raw.fillna(train_mean)

        scaler = StandardScaler() # Réinitialiser le scaler (facultatif mais plus clair)
        X_train_scaled = scaler.fit_transform(X_train_imputed)
        X_val_scaled = scaler.transform(X_val_imputed)
        
        # Entraînement et prédiction
        model_members.fit(X_train_scaled, y_train_log)
        y_pred_log = model_members.predict(X_val_scaled)
        
        y_pred = np.expm1(y_pred_log) # Inverse transformation

        metrics_members_list.append(ev.evaluate_model(y_val_real, y_pred))
        all_y_true_members.append(y_val_real.to_numpy())
        all_y_pred_members.append(y_pred)

    metrics_members_avg = {k: np.mean([m[k] for m in metrics_members_list]) for k in metrics_members_list[0]}
    print("\n===== ElasticNet - y_members (LOG + K-FOLD, LEAKAGE-FREE) =====")
    print(metrics_members_avg)
    
    # ... (Le reste du code de plotting et de return) ...
    # Note: Les sections de plotting doivent utiliser all_y_true_score/members et all_y_pred_score/members
    # qui sont construits correctement par les boucles corrigées.

    return metrics_score_avg, metrics_members_avg