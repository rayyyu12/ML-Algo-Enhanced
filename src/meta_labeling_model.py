# src/meta_labeling_model.py

import pandas as pd
import numpy as np
import lightgbm as lgb
# from sklearn.model_selection import TimeSeriesSplit # Only needed for HPT within this file
from sklearn.metrics import accuracy_score, classification_report, roc_auc_score
from sklearn.preprocessing import StandardScaler
import joblib
import os
import traceback # For detailed error printing

# --- prepare_meta_labeling_data function commented out as logic is in notebook ---
# def prepare_meta_labeling_data(features_df, trade_log_df, feature_cols):
#     """Aligns features with trade outcomes using searchsorted."""
#     # ... (Implementation using searchsorted as shown in the fixed notebook cell) ...
#     # Returns X, y


# --- UPDATED train_meta_labeling_model function (with HPT params argument) ---
def train_meta_labeling_model(X_train, y_train, X_val, y_val, feature_cols,
                              model_save_path, scaler_save_path, lgbm_params={}):
    """
    Trains a LightGBM classifier for meta-labeling, handling class imbalance,
    using validation set for early stopping, retraining on combined data, and saving.
    Accepts tuned hyperparameters via lgbm_params.

    Args:
        X_train (pd.DataFrame): Training features.
        y_train (pd.Series): Training labels (0 or 1).
        X_val (pd.DataFrame or None): Validation features.
        y_val (pd.Series or None): Validation labels.
        feature_cols (list): List of feature column names used.
        model_save_path (str): Full path to save the trained LightGBM model.
        scaler_save_path (str): Full path to save the final fitted StandardScaler.
        lgbm_params (dict, optional): Dictionary of hyperparameters found by Optuna/tuning.
                                      Defaults to {}.

    Returns:
        tuple: (trained LightGBM model, fitted StandardScaler) or (None, None) on error.
    """
    print("Starting meta-labeling model training...")
    if X_train is None or X_train.empty or y_train is None or y_train.empty:
        print("Error: Training data (X_train or y_train) is missing or empty.")
        return None, None

    # --- Handle Class Imbalance ---
    neg_count = (y_train == 0).sum()
    pos_count = (y_train == 1).sum()
    print(f"Training class counts: Negative (0): {neg_count}, Positive (1): {pos_count}")
    if neg_count == 0 or pos_count == 0:
        print("Warning: Training data contains only one class. Model might not be meaningful.")
        lgbm_scale_pos_weight = 1
    else:
        lgbm_scale_pos_weight = neg_count / pos_count
        print(f"Calculated scale_pos_weight for LightGBM: {lgbm_scale_pos_weight:.4f}")

    # --- Scaling ---
    scaler_train = StandardScaler()
    # Ensure feature_cols match X_train columns before scaling
    if not all(col in X_train.columns for col in feature_cols):
        missing_train_scale = [col for col in feature_cols if col not in X_train.columns]
        print(f"ERROR: Missing features in X_train for scaling: {missing_train_scale}")
        return None, None
    X_train_scaled = scaler_train.fit_transform(X_train[feature_cols])
    print("Scaler fitted on training data.")

    # Transform validation data if it exists
    X_val_scaled = None
    eval_set = None
    early_stopping_callback = None # Initialize
    if X_val is not None and not X_val.empty and y_val is not None and not y_val.empty:
        if not all(col in X_val.columns for col in feature_cols):
            missing_val_scale = [col for col in feature_cols if col not in X_val.columns]
            print(f"ERROR: Missing features in X_val for scaling: {missing_val_scale}")
            return None, None
        X_val_scaled = scaler_train.transform(X_val[feature_cols])
        print("Validation data scaled.")
        eval_set = [(X_train_scaled, y_train), (X_val_scaled, y_val)]
        # Use early stopping rounds from HPT config if available, else default
        # Note: HPT config isn't passed here, using fixed value or could enhance later
        early_stopping_rounds = lgbm_params.get('early_stopping_rounds', 20) # Default to 20 if not in params
        early_stopping_callback = lgb.early_stopping(early_stopping_rounds, verbose=1)
    else:
        print("No validation data provided or validation data is empty.")
        eval_set = [(X_train_scaled, y_train)]


    # --- Initial Training (Uses tuned params, applies early stopping) ---
    print("Starting initial LightGBM training (using potentially tuned params)...")
    # Define base params, then update with tuned params
    base_lgbm_params = {
        'random_state': 42,
        'scale_pos_weight': lgbm_scale_pos_weight, # Calculated earlier
        'objective': 'binary',
        'metric': 'auc',
        'verbosity': -1 # Keep fitting quiet unless verbose needed
        # Add default n_estimators if not provided by tuning
        # 'n_estimators': 100 # Default if not in lgbm_params
    }
    # Update base params with tuned params passed in lgbm_params
    # Important: Ensure lgbm_params doesn't contain invalid keys for LGBMClassifier
    current_params = base_lgbm_params.copy()
    current_params.update(lgbm_params)
    print(f"Parameters for initial fit: {current_params}")


    lgbm = lgb.LGBMClassifier(**current_params)

    try:
        lgbm.fit(
            X_train_scaled, y_train,
            eval_set=eval_set,
            eval_metric='auc',
            callbacks=[early_stopping_callback] if early_stopping_callback else []
        )
        print("Initial training complete.")
        best_iter = -1
        # Check attribute existence before accessing
        if early_stopping_callback and hasattr(lgbm, 'best_iteration_') and lgbm.best_iteration_ is not None and lgbm.best_iteration_ > 0:
             best_iter = lgbm.best_iteration_
             print(f"Early stopping used. Best iteration: {best_iter}")
        elif hasattr(lgbm, 'n_estimators_'):
             best_iter = lgbm.n_estimators_ # Use actual trained estimators if no early stopping
             print(f"No early stopping or ran full course. Using n_estimators: {best_iter}")
        else:
             best_iter = current_params.get('n_estimators', 100) # Fallback to param or default
             print(f"Could not determine best iteration, using n_estimators param or default: {best_iter}")


        # --- Retrain Final Model on Combined Data ---
        if X_val is not None and not X_val.empty:
            print("Retraining final model on combined Train + Validation data...")
            X_combined_original = pd.concat([X_train, X_val]) # Use pd.concat
            y_combined = pd.concat([y_train, y_val])
        else:
            print("Retraining final model on Training data only...")
            X_combined_original = X_train
            y_combined = y_train

        # Fit a NEW scaler on the combined *original* data
        final_scaler = StandardScaler()
        if not all(col in X_combined_original.columns for col in feature_cols):
             missing_final_scale = [col for col in feature_cols if col not in X_combined_original.columns]
             print(f"ERROR: Missing features in combined data for final scaling: {missing_final_scale}")
             return None, None
        X_combined_scaled = final_scaler.fit_transform(X_combined_original[feature_cols])
        print("Final scaler fitted on combined data.")


        # Create a new final LGBM model instance using tuned params
        final_lgbm_params = current_params.copy() # Start with params used for initial fit
        final_lgbm_params['n_estimators'] = best_iter # Set estimators based on best_iter found
        print(f"Final model parameters: {final_lgbm_params}")

        final_lgbm = lgb.LGBMClassifier(**final_lgbm_params)

        # Train the final model
        final_lgbm.fit(X_combined_scaled, y_combined)
        print("Final model retrained.")

        # --- Save the FINAL Model and FINAL Scaler ---
        try:
            os.makedirs(os.path.dirname(model_save_path), exist_ok=True)
            os.makedirs(os.path.dirname(scaler_save_path), exist_ok=True)
            joblib.dump(final_lgbm, model_save_path)
            joblib.dump(final_scaler, scaler_save_path)
            print(f"Final Meta-labeling model saved to: {model_save_path}")
            print(f"Final Scaler saved to: {scaler_save_path}")
            return final_lgbm, final_scaler # Return the *final* model and scaler

        except Exception as save_err:
            print(f"ERROR saving final model or scaler: {save_err}")
            return None, None

    except Exception as fit_err:
        print(f"ERROR during LightGBM fitting: {fit_err}")
        traceback.print_exc()
        return None, None


# --- predict_trade_outcome function (Keep as before) ---
def predict_trade_outcome(features, model, scaler, feature_cols):
    """Predicts win probability for a potential trade."""
    if isinstance(features, pd.Series):
        if not all(col in features.index for col in feature_cols):
             missing_pred = [col for col in feature_cols if col not in features.index]
             # print(f"Warning: Missing features for prediction: {missing_pred}") # Can be noisy
             return 0, 0.0
        feature_values = features[feature_cols].values.reshape(1, -1)
    elif isinstance(features, pd.DataFrame):
        if not all(col in features.columns for col in feature_cols):
             missing_pred = [col for col in feature_cols if col not in features.columns]
             # print(f"Warning: Missing features for prediction: {missing_pred}") # Can be noisy
             return 0, 0.0
        feature_values = features[feature_cols].iloc[0].values.reshape(1, -1)
    else:
        print("Warning: Invalid type for 'features' in predict_trade_outcome.")
        return 0, 0.0

    # Check for NaNs or Infs before scaling/prediction
    if np.isfinite(feature_values).all():
        try:
            X_scaled = scaler.transform(feature_values)
            # Add check for model readiness if needed (e.g., model.fitted_)
            probability = model.predict_proba(X_scaled)[0, 1]
            prediction = model.predict(X_scaled)[0]
            return prediction, probability
        except Exception as pred_err:
             print(f"Error during prediction: {pred_err}")
             return 0, 0.0
    else:
        # print(f"Warning: Non-finite values in features for prediction: {feature_values}") # Debugging
        return 0, 0.0