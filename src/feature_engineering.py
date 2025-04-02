# src/feature_engineering.py

import pandas as pd
import numpy as np # Import numpy first
import datetime as dt # For time comparisons
import os # For path joining in test block
import yaml # To load config if run standalone for testing
import traceback # For detailed error printing in test block

# --- WORKAROUND for pandas-ta/numpy 2.0 incompatibility ---
# Manually add the 'NaN' attribute to the numpy module before pandas-ta
# (version 0.3.14b0 or similar) tries to import it.
if not hasattr(np, 'NaN'):
    np.NaN = np.nan
    print("Applied NumPy workaround for pandas-ta compatibility.")
# --- End Workaround ---

try:
    import pandas_ta as ta # Now import pandas-ta AFTER the patch
except ImportError as e:
    print("ERROR: Failed to import pandas_ta even after workaround.")
    print("Please ensure pandas-ta is installed (`pip install pandas-ta`)")
    raise e
except Exception as e:
    print(f"An unexpected error occurred during pandas_ta import: {e}")
    raise e


# --- Indicator Calculation Functions ---
def calculate_atr(high, low, close, period=14):
    """Calculates Average True Range, returns NaN Series on error."""
    if not all(isinstance(i, pd.Series) for i in [high, low, close]):
        print(f"Error calc ATR_{period}: Inputs must be pandas Series")
        return pd.Series(np.nan, index=high.index if isinstance(high, pd.Series) else None)
    try:
        return ta.atr(high, low, close, length=period)
    except Exception as e:
        print(f"Error calc ATR_{period}: {e}")
        return pd.Series(np.nan, index=high.index)

def calculate_rsi(close, period=14):
    """Calculates Relative Strength Index, returns NaN Series on error."""
    if not isinstance(close, pd.Series):
        print(f"Error calc RSI_{period}: Input must be a pandas Series")
        return pd.Series(np.nan, index=close.index if isinstance(close, pd.Series) else None)
    try:
        return ta.rsi(close, length=period)
    except Exception as e:
        print(f"Error calc RSI_{period}: {e}")
        return pd.Series(np.nan, index=close.index)

def calculate_sma(close, period=50):
    """Calculates Simple Moving Average, returns NaN Series on error."""
    if not isinstance(close, pd.Series):
        print(f"Error calc SMA_{period}: Input must be a pandas Series")
        return pd.Series(np.nan, index=close.index if isinstance(close, pd.Series) else None)
    try:
        return ta.sma(close, length=period)
    except Exception as e:
        print(f"Error calc SMA_{period}: {e}")
        return pd.Series(np.nan, index=close.index)

def calculate_volume_sma(volume, period=20):
    """Calculates Simple Moving Average for Volume, returns NaN Series on error."""
    if not isinstance(volume, pd.Series):
        print(f"Error calc VolSMA_{period}: Input must be a pandas Series")
        return pd.Series(np.nan, index=volume.index if isinstance(volume, pd.Series) else None)
    try:
        return ta.sma(volume, length=period)
    except Exception as e:
        print(f"Error calc VolSMA_{period}: {e}")
        return pd.Series(np.nan, index=volume.index)

def calculate_volatility(close, period=20):
    """Calculates rolling volatility, returns NaN Series on error."""
    if not isinstance(close, pd.Series):
        print(f"Error calc Vol_{period}: Input must be a pandas Series")
        return pd.Series(np.nan, index=close.index if isinstance(close, pd.Series) else None)
    try:
        log_returns = np.log(close / close.shift(1))
        annualization_factor = np.sqrt(252 * 24 * 60) # Standard factor
        return log_returns.rolling(window=period).std() * annualization_factor
    except Exception as e:
        print(f"Error calc Vol_{period}: {e}")
        return pd.Series(np.nan, index=close.index)

def calculate_roc(close, period=10):
    """Calculates Rate of Change, returns NaN Series on error."""
    if not isinstance(close, pd.Series):
        print(f"Error calc ROC_{period}: Input must be a pandas Series")
        return pd.Series(np.nan, index=close.index if isinstance(close, pd.Series) else None)
    try:
        return ta.roc(close, length=period)
    except Exception as e:
        print(f"Error calc ROC_{period}: {e}")
        return pd.Series(np.nan, index=close.index)

def calculate_adx(high, low, close, period=14):
    """Calculates Average Directional Index (ADX), returns NaN Series on error."""
    if not all(isinstance(i, pd.Series) for i in [high, low, close]):
        print(f"Error calc ADX_{period}: Inputs must be pandas Series")
        return pd.Series(np.nan, index=high.index if isinstance(high, pd.Series) else None)
    try:
        adx_df = ta.adx(high, low, close, length=period)
        adx_col = f'ADX_{period}'
        # Check for expected column names robustly
        if adx_col in adx_df.columns: return adx_df[adx_col]
        elif 'ADX' in adx_df.columns: return adx_df['ADX'] # Fallback name
        else: print(f"Warning: ADX column not found in ta.adx output. Cols: {adx_df.columns}"); return pd.Series(np.nan, index=high.index)
    except Exception as e_adx:
        print(f"Error calc ADX_{period}: {e_adx}")
        return pd.Series(np.nan, index=high.index)

# --- CORRECTED calculate_macd ---
def calculate_macd(close, fast=12, slow=26, signal=9):
     """Calculates MACD, returns DataFrame with consistent double-parameter names or NaNs on error."""
     if not isinstance(close, pd.Series):
          print(f"Error calc MACD: Input must be pandas Series")
          index = close.index if isinstance(close, pd.Series) else None
          cols = [f'MACD_{fast}_{slow}_{signal}_{fast}_{slow}_{signal}', f'MACDh_{fast}_{slow}_{signal}_{fast}_{slow}_{signal}', f'MACDs_{fast}_{slow}_{signal}_{fast}_{slow}_{signal}']
          return pd.DataFrame(np.nan, index=index, columns=cols)
     try:
          macd_df = ta.macd(close, fast=fast, slow=slow, signal=signal)
          # Determine original columns (handle potential variations in ta library versions)
          original_cols = macd_df.columns
          if len(original_cols) != 3:
               print(f"Warning: Unexpected number of columns from ta.macd: {original_cols}. Cannot reliably rename.")
               return macd_df # Return as is, might cause issues later
          # Create the CORRECT desired names (with double parameters)
          new_cols = {
              original_cols[0]: f'MACD_{fast}_{slow}_{signal}_{fast}_{slow}_{signal}',  # MACD Line
              original_cols[1]: f'MACDh_{fast}_{slow}_{signal}_{fast}_{slow}_{signal}', # Histogram
              original_cols[2]: f'MACDs_{fast}_{slow}_{signal}_{fast}_{slow}_{signal}'  # Signal Line
          }
          macd_df.rename(columns=new_cols, inplace=True)
          return macd_df
     except Exception as e_macd:
          print(f"Error calculating MACD: {e_macd}")
          cols = [f'MACD_{fast}_{slow}_{signal}_{fast}_{slow}_{signal}', f'MACDh_{fast}_{slow}_{signal}_{fast}_{slow}_{signal}', f'MACDs_{fast}_{slow}_{signal}_{fast}_{slow}_{signal}']
          return pd.DataFrame(np.nan, index=close.index, columns=cols)


# --- Main Feature Adding Function ---
def add_features(df, config):
    """Adds technical indicators and contextual features based on config."""
    if not isinstance(df, pd.DataFrame): raise TypeError("Input df must be pandas DataFrame")
    if not isinstance(config, dict): raise TypeError("Input config must be dictionary")

    fp = config.get('feature_params', {})
    sp = config.get('strategy_params', {})
    orb_start_time_cfg = sp.get('openingRangeStartTime', 830); orb_end_time_cfg = sp.get('openingRangeEndTime', 845)
    market_close_time_cfg = sp.get('marketCloseTime', 1500)

    # Convert times
    try:
        st = str(orb_start_time_cfg).zfill(4); et = str(orb_end_time_cfg).zfill(4); mct = str(market_close_time_cfg).zfill(4)
        orb_start_time = dt.time(int(st[:2]), int(st[2:])); orb_end_time = dt.time(int(et[:2]), int(et[2:]))
        market_close_time = dt.time(int(mct[:2]), int(mct[2:])); market_open_time = dt.time(8, 30) # Assuming 8:30 CST open
    except Exception as time_err:
         print(f"ERROR converting config times: {time_err}"); raise

    print("Calculating base indicators...")
    # --- Extract Periods ---
    atr_p = fp.get('atr_period', 14); atr_lp = fp.get('atr_long_period', 50)
    rsi_p = fp.get('rsi_period', 7); sma_f_p = fp.get('sma_fast_period', 15); sma_s_p = fp.get('sma_slow_period', 20)
    vol_p = fp.get('volatility_period', 20); vol_sma_p = fp.get('volume_sma_period', 20)
    roc_p = fp.get('roc_period', 10); adx_p = fp.get('adx_period', 14)
    macd_f = fp.get('macd_fast', 12); macd_s = fp.get('macd_slow', 26); macd_sig = fp.get('macd_signal', 9)

    # --- Calculate Base Indicators (with checks for required input columns) ---
    required_ohlcv = ['Open', 'High', 'Low', 'Close', 'Volume']
    if not all(col in df.columns for col in required_ohlcv):
        missing_ohlcv = [col for col in required_ohlcv if col not in df.columns]
        raise ValueError(f"Input DataFrame missing required OHLCV columns: {missing_ohlcv}")

    df[f'ATR_{atr_p}'] = calculate_atr(df['High'], df['Low'], df['Close'], period=atr_p)
    df[f'ATR_{atr_lp}'] = calculate_atr(df['High'], df['Low'], df['Close'], period=atr_lp)
    df[f'RSI_{rsi_p}'] = calculate_rsi(df['Close'], period=rsi_p)
    df[f'SMA_{sma_f_p}'] = calculate_sma(df['Close'], period=sma_f_p)
    df[f'SMA_{sma_s_p}'] = calculate_sma(df['Close'], period=sma_s_p)
    df[f'Volume_SMA_{vol_sma_p}'] = calculate_volume_sma(df['Volume'], period=vol_sma_p)
    df[f'Volatility_{vol_p}'] = calculate_volatility(df['Close'], period=vol_p)
    df[f'ROC_{roc_p}'] = calculate_roc(df['Close'], period=roc_p)
    df[f'ADX_{adx_p}'] = calculate_adx(df['High'], df['Low'], df['Close'], period=adx_p)
    macd_df = calculate_macd(df['Close'], fast=macd_f, slow=macd_s, signal=macd_sig)
    df = pd.concat([df, macd_df], axis=1)

    # --- Calculate Ratio/Normalized Features ---
    epsilon = 1e-9 # Use a smaller epsilon to avoid issues with very small ATRs
    if f'ATR_{atr_p}' in df.columns and f'ATR_{atr_lp}' in df.columns: df[f'ATR_Ratio_{atr_p}_{atr_lp}'] = df[f'ATR_{atr_p}'] / (df[f'ATR_{atr_lp}'] + epsilon)
    else: df[f'ATR_Ratio_{atr_p}_{atr_lp}'] = np.nan
    if f'SMA_{sma_f_p}' in df.columns and f'ATR_{atr_p}' in df.columns: df[f'Close_vs_SMA_{sma_f_p}_Norm_ATR_{atr_p}'] = (df['Close'] - df[f'SMA_{sma_f_p}']) / (df[f'ATR_{atr_p}'] + epsilon)
    else: df[f'Close_vs_SMA_{sma_f_p}_Norm_ATR_{atr_p}'] = np.nan
    if f'SMA_{sma_s_p}' in df.columns and f'ATR_{atr_p}' in df.columns: df[f'Close_vs_SMA_{sma_s_p}_Norm_ATR_{atr_p}'] = (df['Close'] - df[f'SMA_{sma_s_p}']) / (df[f'ATR_{atr_p}'] + epsilon)
    else: df[f'Close_vs_SMA_{sma_s_p}_Norm_ATR_{atr_p}'] = np.nan

    print("Calculating time features...")
    # --- Calculate Time Features ---
    df['Hour'] = df.index.hour; df['Minute'] = df.index.minute; df['DayOfWeek'] = df.index.dayofweek
    time_component = df.index.time
    def minutes_since_open(t): return ((t.hour*60+t.minute+t.second/60.0) - (market_open_time.hour*60+market_open_time.minute)) if t >= market_open_time else np.nan
    df['Time_Since_Open'] = pd.Series(time_component, index=df.index).apply(minutes_since_open)
    def minutes_to_close(t): return ((market_close_time.hour*60+market_close_time.minute) - (t.hour*60+t.minute+t.second/60.0)) if t < market_close_time else 0
    df['Time_to_Close'] = pd.Series(time_component, index=df.index).apply(minutes_to_close)

    print("Calculating daily and ORB context features...")
    # --- Calculate Daily and ORB Context ---
    df['Date'] = df.index.date # Use temporary Date column for grouping
    # Use transform for efficiency if possible, otherwise cummax/cummin is fine
    df['Daily_High'] = df.groupby(df['Date'])['High'].cummax()
    df['Daily_Low'] = df.groupby(df['Date'])['Low'].cummin()

    # Calculate ORB stats per day
    daily_orb_stats = df[(df.index.time >= orb_start_time) & (df.index.time < orb_end_time)]\
        .groupby(df[(df.index.time >= orb_start_time) & (df.index.time < orb_end_time)].index.date)\
        .agg(ORB_High=('High', 'max'), ORB_Low=('Low', 'min'))

    if not daily_orb_stats.empty:
        daily_orb_stats['ORB_Size'] = daily_orb_stats['ORB_High'] - daily_orb_stats['ORB_Low']
        daily_orb_stats['ORB_Size'] = daily_orb_stats['ORB_Size'].where(daily_orb_stats['ORB_Size'] > 0, np.nan)
        # Get Long ATR value around ORB end time for normalization
        atr_col_long = f'ATR_{atr_lp}'
        if atr_col_long in df.columns:
            atr_orb_end_series = df.loc[df.index.time == orb_end_time, atr_col_long]
            atr_near_orb_end = atr_orb_end_series.groupby(atr_orb_end_series.index.date).first()
            daily_orb_stats = daily_orb_stats.merge(atr_near_orb_end.rename(f'{atr_col_long}_at_ORB_End'), left_index=True, right_index=True, how='left')
            daily_orb_stats[f'ORB_Size_Norm_ATR_{atr_lp}'] = daily_orb_stats['ORB_Size'] / (daily_orb_stats[f'{atr_col_long}_at_ORB_End'] + epsilon)
            # Merge ORB stats back
            df = df.merge(daily_orb_stats, left_on='Date', right_index=True, how='left')
            df.drop(columns=[f'{atr_col_long}_at_ORB_End'], inplace=True, errors='ignore') # Drop temporary merge column
        else:
             print(f"Warning: Long ATR column {atr_col_long} not found for ORB size normalization.")
             # Add ORB cols with NaNs if ATR was missing
             df = df.merge(daily_orb_stats[['ORB_High', 'ORB_Low', 'ORB_Size']], left_on='Date', right_index=True, how='left')
             df[f'ORB_Size_Norm_ATR_{atr_lp}'] = np.nan

        # Calculate Normalized Distance from ORB Edges (only valid after ORB)
        df['Dist_from_ORB_Low_Norm'] = np.where((df['ORB_Size'] > epsilon) & (df.index.time >= orb_end_time), (df['Close'] - df['ORB_Low']) / df['ORB_Size'], np.nan)
        df['Dist_from_ORB_High_Norm'] = np.where((df['ORB_Size'] > epsilon) & (df.index.time >= orb_end_time), (df['ORB_High'] - df['Close']) / df['ORB_Size'], np.nan)
    else:
        print("Warning: No data found within ORB time range. ORB features will be NaN.")
        # Add ORB columns as NaN if no ORB data exists at all
        for col in ['ORB_High', 'ORB_Low', 'ORB_Size', f'ORB_Size_Norm_ATR_{atr_lp}', 'Dist_from_ORB_Low_Norm', 'Dist_from_ORB_High_Norm']:
            if col not in df.columns: df[col] = np.nan

    df.drop(columns=['Date'], inplace=True) # Drop temporary Date column


    print("Calculating strategy filter and interaction features...")
    # --- Calculate Strategy Filter Features ---
    sma_f_col = f'SMA_{sma_f_p}'; sma_s_col = f'SMA_{sma_s_p}'
    df['IsTrendingUp'] = np.where(df[sma_f_col] > df[sma_s_col], 1, 0) if sma_f_col in df.columns and sma_s_col in df.columns else 0

    vol_thresh_mult = sp.get('volumeThreshold', 1.5); vol_sma_col = f'Volume_SMA_{vol_sma_p}'
    df['IsHighVolume'] = np.where((df[vol_sma_col] > epsilon) & (~df[vol_sma_col].isnull()) & (df['Volume'] > vol_thresh_mult * df[vol_sma_col]), 1, 0) if 'Volume' in df.columns and vol_sma_col in df.columns else 0

    # --- Calculate Interaction Features ---
    vol_col = f'Volatility_{vol_p}'; adx_col = f'ADX_{adx_p}'
    df['Interaction_Vol_ADX'] = df[vol_col] * df[adx_col] if vol_col in df.columns and adx_col in df.columns else np.nan
    rsi_col = f'RSI_{rsi_p}'; dist_orb_col = 'Dist_from_ORB_Low_Norm'
    df['Interaction_RSI_DistORB'] = df[rsi_col] * df[dist_orb_col] if rsi_col in df.columns and dist_orb_col in df.columns else np.nan

    print("Lagging features...")
    # --- Lag Features for Modeling ---
    feature_cols_to_lag = [
        f'ATR_{atr_p}', f'ATR_{atr_lp}', f'RSI_{rsi_p}', sma_f_col, sma_s_col,
        vol_sma_col, vol_col, f'ROC_{roc_p}', adx_col,
        # Use the CORRECT double-parameter base names generated by calculate_macd
        f'MACD_{macd_f}_{macd_s}_{macd_sig}_{macd_f}_{macd_s}_{macd_sig}',
        f'MACDh_{macd_f}_{macd_s}_{macd_sig}_{macd_f}_{macd_s}_{macd_sig}',
        f'MACDs_{macd_f}_{macd_s}_{macd_sig}_{macd_f}_{macd_s}_{macd_sig}',
        f'ATR_Ratio_{atr_p}_{atr_lp}', f'Close_vs_SMA_{sma_f_p}_Norm_ATR_{atr_p}', f'Close_vs_SMA_{sma_s_p}_Norm_ATR_{atr_p}',
        'Hour', 'Minute', 'DayOfWeek', 'Time_Since_Open', 'Time_to_Close',
        'Volume', 'Daily_High', 'Daily_Low', 'ORB_High', 'ORB_Low', 'ORB_Size', f'ORB_Size_Norm_ATR_{atr_lp}',
        'Dist_from_ORB_Low_Norm', 'Dist_from_ORB_High_Norm',
        'Interaction_Vol_ADX', 'Interaction_RSI_DistORB',
        'IsTrendingUp', 'IsHighVolume'
    ]
    lagged_cols_created = []
    for col in feature_cols_to_lag:
        if col in df.columns:
            lagged_col_name = f'{col}_lag1'; df[lagged_col_name] = df[col].shift(1); lagged_cols_created.append(lagged_col_name)
        else: print(f"Warning: Base column '{col}' not found for lagging.")

    # --- Clean up ---
    print("Dropping rows with initial NaNs from essential indicators...")
    initial_rows = df.shape[0]
    # Define essential indicators whose initial NaNs MUST be dropped (choose longest lookbacks)
    essential_indicators = [ f'ATR_{atr_lp}', sma_s_col, vol_col, adx_col, f'MACD_{macd_f}_{macd_s}_{macd_sig}_{macd_f}_{macd_s}_{macd_sig}']
    essential_lagged_cols = [f'{col}_lag1' for col in essential_indicators if f'{col}_lag1' in lagged_cols_created]
    if essential_lagged_cols:
        df.dropna(subset=essential_lagged_cols, inplace=True)
        print(f"Dropped {initial_rows - df.shape[0]} rows (essential indicator lags).")
    else: print("Warning: No essential lagged indicator columns found to drop NaNs by.")

    # Forward fill contextual features (handle potential missing cols)
    context_cols = ['ORB_High_lag1', 'ORB_Low_lag1', 'ORB_Size_lag1', f'ORB_Size_Norm_ATR_{atr_lp}_lag1',
                    'Daily_High_lag1', 'Daily_Low_lag1',
                    'Dist_from_ORB_Low_Norm_lag1', 'Dist_from_ORB_High_Norm_lag1',
                    'Time_Since_Open_lag1', 'Time_to_Close_lag1',
                    'Interaction_RSI_DistORB_lag1'] # Interaction depends on ORB context
    context_cols_present = [c for c in context_cols if c in df.columns]
    if context_cols_present:
         print("Forward filling contextual lagged features within each day...")
         # Use fillna(method='ffill') grouped by date; more explicit than just ffill()
         df[context_cols_present] = df.groupby(df.index.date, group_keys=False)[context_cols_present].fillna(method='ffill')
         # Final drop of any remaining NaNs (e.g., first day, interaction dependencies)
         final_check_cols = essential_lagged_cols + context_cols_present # Check all potentially problematic cols
         initial_rows_ffill = df.shape[0]
         df.dropna(subset=final_check_cols, inplace=True)
         print(f"Dropped {initial_rows_ffill - df.shape[0]} rows after ffill/final context check.")
         print(f"Final shape after NaN handling: {df.shape}")

    print("Feature engineering complete.")
    return df

# --- Example Usage (Keep as before, minor adjustment) ---
if __name__ == '__main__':
    print("\nRunning feature_engineering.py directly for testing...")
    test_config = None
    try:
        config_path = os.path.join(os.path.dirname(__file__), '..', 'config', 'config.yaml')
        with open(config_path, 'r') as f: test_config = yaml.safe_load(f)
        print("Test config loaded.")
    except Exception as e: print(f"Could not load config for testing: {e}")

    if test_config:
        # Create sample data
        start_dt = '2023-01-09 07:00'; periods=2000
        sample_dates = pd.date_range(start=start_dt, periods=periods, freq='1min')
        sample_data = { 'Open': np.random.rand(periods)*2+15000, 'Close': np.random.rand(periods)*2+15000, 'Volume': np.random.randint(50, 500, periods) }
        sample_df = pd.DataFrame(sample_data, index=sample_dates)
        sample_df['High'] = sample_df[['Open', 'Close']].max(axis=1) + np.random.rand(periods)*0.5
        sample_df['Low'] = sample_df[['Open', 'Close']].min(axis=1) - np.random.rand(periods)*0.5
        print("Sample DataFrame created.")

        try:
             features_test_df = add_features(sample_df.copy(), test_config)
             print("\nFeatures added to sample DataFrame (sample rows around ORB time):")
             pd.set_option('display.max_columns', None); pd.set_option('display.max_rows', 25)
             # Display sample including lagged cols
             print(features_test_df['2023-01-09 08:40':'2023-01-09 09:05'])#.filter(regex='MACD|ORB|Dist|Time_|ADX|Ratio|Norm|lag1'))
             print("\nChecking final columns and NaNs:")
             print(features_test_df.info(verbose=True, show_counts=True)) # More detailed info
             print("\nColumns still containing NaNs (if any):")
             final_nans_test = features_test_df.isnull().sum()
             print(final_nans_test[final_nans_test > 0])
        except Exception as e:
             print(f"Error during feature calculation test: {e}")
             traceback.print_exc()
    else: print("Skipping feature calculation test as config failed to load.")