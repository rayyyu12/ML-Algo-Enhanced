# src/regime_model.py
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import joblib
import os # Import os

# --- Keep the existing predict_regime function ---
def predict_regime(data_point, model, scaler, features=['Volatility_20_lag1', 'ATR_lag1']):
    """Predicts regime for a single data point (e.g., a row of features)."""
    X = data_point[features].values.reshape(1, -1)
    X_scaled = scaler.transform(X)
    return model.predict(X_scaled)[0]


# --- UPDATED train_regime_model function ---
def train_regime_model(features_df, n_regimes, features, model_save_path, scaler_save_path):
    """
    Trains a K-Means model for regime detection and saves the model and scaler.

    Args:
        features_df (pd.DataFrame): DataFrame containing features for training.
        n_regimes (int): Number of regimes (clusters) to find.
        features (list): List of feature column names to use for clustering.
        model_save_path (str): Full path where the trained KMeans model should be saved.
        scaler_save_path (str): Full path where the fitted StandardScaler should be saved.

    Returns:
        tuple: (DataFrame with 'Regime' column added, trained KMeans model, fitted StandardScaler)
               Returns (None, None, None) on error.
    """
    print(f"Starting regime model training using features: {features}")
    if features_df is None or features_df.empty:
        print("Error: Input features_df is None or empty.")
        return None, None, None
    if not all(f in features_df.columns for f in features):
        missing = [f for f in features if f not in features_df.columns]
        print(f"Error: Missing required features in DataFrame: {missing}")
        return None, None, None

    try:
        X = features_df[features].copy() # Use copy to avoid SettingWithCopyWarning

        # Handle potential NaNs before scaling
        if X.isnull().any().any():
            print(f"Warning: Found {X.isnull().sum().sum()} NaNs in training features. Filling with median.")
            # Fill NaNs using the median of the training data column
            for col in features:
                 if X[col].isnull().any():
                      median_val = X[col].median()
                      X[col].fillna(median_val, inplace=True)

        if X.empty:
             print("Error: No valid data left after NaN handling (if any).")
             return None, None, None


        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)

        # Use 'lloyd' explicitly if needed, n_init='auto' or 10 are common
        kmeans = KMeans(n_clusters=n_regimes, random_state=42, n_init=10, algorithm='lloyd')
        kmeans.fit(X_scaled)

        # --- Save the model and scaler using provided paths ---
        # Ensure parent directory exists
        os.makedirs(os.path.dirname(model_save_path), exist_ok=True)
        os.makedirs(os.path.dirname(scaler_save_path), exist_ok=True)

        joblib.dump(kmeans, model_save_path)
        joblib.dump(scaler, scaler_save_path)
        print(f"Regime model saved to: {model_save_path}")
        print(f"Scaler saved to: {scaler_save_path}")
        # ----------------------------------------------------

        # Add regime labels back to the input df (use the non-scaled data index)
        # Use .loc to avoid potential index misalignment if NaNs were dropped differently
        regime_labels = kmeans.predict(X_scaled)
        # Create a Series with the same index as X to assign back correctly
        regime_series = pd.Series(regime_labels, index=X.index, name='Regime')
        features_df_with_regime = features_df.copy() # Create copy to add column safely
        features_df_with_regime['Regime'] = regime_series # Assign based on index

        # Analyze cluster centers to manually label regimes (e.g., low vol, high vol)
        centers_scaled = kmeans.cluster_centers_
        # Use try-except for inverse_transform in case scaling wasn't actually needed (e.g., only one feature)
        try:
            centers_original = scaler.inverse_transform(centers_scaled)
        except Exception: # Catch potential errors if inverse_transform is not applicable
             print("Warning: Could not inverse_transform cluster centers.")
             centers_original = centers_scaled # Show scaled centers if inverse fails

        print("\nRegime Cluster Centers (Original Scale - approx if NaNs were filled):")
        print(pd.DataFrame(centers_original, columns=features))
        print("-" * 30)

        return features_df_with_regime, kmeans, scaler

    except Exception as e:
        print(f"An unexpected error occurred during regime model training: {e}")
        import traceback
        traceback.print_exc() # Print full traceback for debugging
        return None, None, None