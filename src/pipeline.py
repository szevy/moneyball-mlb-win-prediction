import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import RidgeCV, LinearRegression
from sklearn.ensemble import RandomForestRegressor, StackingRegressor
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import GroupKFold

def feature_engineering(train_df, test_df):
    # Your feature engineering as before
    train_df['BA'] = train_df['H'] / train_df['AB'].replace(0, np.nan)
    test_df['BA'] = test_df['H'] / test_df['AB'].replace(0, np.nan)

    train_df['WHIP'] = (train_df['BBA'] + train_df['HA']) / (train_df['IPouts'] / 3).replace(0, np.nan)
    test_df['WHIP'] = (test_df['BBA'] + test_df['HA']) / (test_df['IPouts'] / 3).replace(0, np.nan)

    per_game = ['R','HR','BB','SO','SB','RA','ER','HA','HRA','BBA','SOA','E']
    for col in per_game:
        train_df[f'{col}_per_game'] = train_df[col] / train_df['G']
        test_df[f'{col}_per_game'] = test_df[col] / test_df['G']

    train_df['OPS_like'] = train_df['BA'] + train_df['BB_per_game']
    test_df['OPS_like'] = test_df['BA'] + test_df['BB_per_game']

    train_df['K_BB_ratio'] = train_df['SOA_per_game'] / (train_df['BBA_per_game'] + 1)
    test_df['K_BB_ratio'] = test_df['SOA_per_game'] / (test_df['BBA_per_game'] + 1)

    train_df['ERA_per_WHIP'] = train_df['ER'] / (train_df['WHIP'] + 1)
    test_df['ERA_per_WHIP'] = test_df['ER'] / (test_df['WHIP'] + 1)

    return train_df.fillna(0), test_df.fillna(0)

def prepare_features(train_df, test_df):
    excluded_cols = ['W', 'yearID', 'teamID', 'year_label', 'decade_label', 'win_bins']
    available_features = [col for col in train_df.columns if col not in excluded_cols and col in test_df.columns]

    per_game = ['R','HR','BB','SO','SB','RA','ER','HA','HRA','BBA','SOA','E']
    engineered = ['BA', 'WHIP', 'OPS_like', 'K_BB_ratio', 'ERA_per_WHIP'] + [f'{col}_per_game' for col in per_game]
    final_features = [col for col in available_features + engineered if col in train_df.columns and col in test_df.columns]

    X_train = train_df[final_features].copy()
    y_train = train_df['W'].copy()
    X_test = test_df[final_features].copy()

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    return X_train_scaled, y_train.values, X_test_scaled, scaler, final_features

def train_ridge(X_train, y_train):
    ridge = RidgeCV(alphas=np.logspace(-3, 3, 50), cv=5)
    ridge.fit(X_train, y_train)
    return ridge

def train_stack(X_train, y_train):
    ridge = RidgeCV(alphas=np.logspace(-3, 3, 50), cv=5)
    rf = RandomForestRegressor(n_estimators=100, random_state=42)
    stack = StackingRegressor(
        estimators=[('ridge', ridge), ('rf', rf)],
        final_estimator=RidgeCV(alphas=np.logspace(-3, 3, 50), cv=5)
    )
    stack.fit(X_train, y_train)
    return stack

def run_blending_cv(X_train, y_train, groups, n_splits=5):
    gkf = GroupKFold(n_splits=n_splits)

    oof_ridge = np.zeros(len(y_train))
    oof_stack = np.zeros(len(y_train))

    for fold, (train_idx, val_idx) in enumerate(gkf.split(X_train, y_train, groups)):
        print(f"Fold {fold+1}/{n_splits}")
        X_tr, X_val = X_train[train_idx], X_train[val_idx]
        y_tr, y_val = y_train[train_idx], y_train[val_idx]

        # Train ridge
        ridge = train_ridge(X_tr, y_tr)
        preds_ridge = ridge.predict(X_val)
        oof_ridge[val_idx] = preds_ridge
        print(f"  Ridge MAE fold: {mean_absolute_error(y_val, preds_ridge):.4f}")

        # Train stack
        stack = train_stack(X_tr, y_tr)
        preds_stack = stack.predict(X_val)
        oof_stack[val_idx] = preds_stack
        print(f"  Stack MAE fold: {mean_absolute_error(y_val, preds_stack):.4f}")

        # Store last fold actuals and predictions for visualization
        if fold == n_splits - 1:
            last_fold_y_val = y_val
            last_fold_y_pred = (preds_ridge + preds_stack) / 2  # simple average for visualization

    # Train meta-learner on OOF predictions
    meta_X = np.vstack([oof_ridge, oof_stack]).T
    meta_learner = LinearRegression()
    meta_learner.fit(meta_X, y_train)

    print(f"\nMeta-learner Coefs: {meta_learner.coef_}, Intercept: {meta_learner.intercept_}")

    # Blended OOF predictions and MAE
    oof_blended = meta_learner.predict(meta_X)
    final_oof_mae = mean_absolute_error(y_train, oof_blended)
    print(f"Final blended OOF MAE: {final_oof_mae:.4f}")

    # --- Visualization (Last Validation Fold) ---
    if last_fold_y_val is not None:
        plt.figure(figsize=(10, 6))
        plt.scatter(last_fold_y_val, last_fold_y_pred, alpha=0.6)
        min_val = min(last_fold_y_val.min(), last_fold_y_pred.min())
        max_val = max(last_fold_y_val.max(), last_fold_y_pred.max())
        plt.plot([min_val, max_val], [min_val, max_val], 'r--')
        plt.xlabel("Actual Wins")
        plt.ylabel("Predicted Wins")
        plt.title("Actual vs Predicted Wins (Last Validation Fold)")
        plt.grid(True)
        plt.show()

        plt.figure(figsize=(10, 6))
        residuals = last_fold_y_val - last_fold_y_pred
        plt.scatter(last_fold_y_pred, residuals, alpha=0.6)
        plt.axhline(0, color='red', linestyle='--')
        plt.xlabel("Predicted Wins")
        plt.ylabel("Residuals")
        plt.title("Residual Plot (Last Validation Fold)")
        plt.grid(True)
        plt.show()
    else:
        print("No validation fold data for plotting.")
        
    return meta_learner

def train_full_models(X_train, y_train):
    ridge = train_ridge(X_train, y_train)
    stack = train_stack(X_train, y_train)
    return ridge, stack

def create_submission(meta_learner, ridge, stack, X_test, test_df, filename='submission_meta_blend.csv'):
    preds_ridge = ridge.predict(X_test)
    preds_stack = stack.predict(X_test)
    meta_X_test = np.vstack([preds_ridge, preds_stack]).T
    blended_preds = meta_learner.predict(meta_X_test)

    submission = pd.DataFrame({
        'ID': test_df['ID'],
        'W': blended_preds
    })

    os.makedirs('../data', exist_ok=True)
    submission_path = os.path.join('../data', filename)
    submission.to_csv(submission_path, index=False)
    print(f"✅ Submission saved to {submission_path}")

def run_pipeline(data_dir='../data'):
    train_df = pd.read_csv(os.path.join(data_dir, 'train.csv'))
    test_df = pd.read_csv(os.path.join(data_dir, 'test.csv'))

    train_df, test_df = feature_engineering(train_df, test_df)
    X_train, y_train, X_test, scaler, features = prepare_features(train_df, test_df)

    meta_learner = run_blending_cv(X_train, y_train, groups=train_df['yearID'].values)

    ridge, stack = train_full_models(X_train, y_train)

    create_submission(meta_learner, ridge, stack, X_test, test_df)

if __name__ == '__main__':
    run_pipeline()
