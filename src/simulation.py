# src/simulation.py

import pandas as pd
import numpy as np
import joblib
import yaml
from tqdm import tqdm
import os
from datetime import time, date # Import date for combine
import traceback # For detailed error printing

# Import necessary functions from other src modules using absolute imports from project root
# This relies on the project root directory being in sys.path (done in the notebook)
from src.regime_model import predict_regime
from src.meta_labeling_model import predict_trade_outcome
from src.position_sizing import get_position_size
from src.utils import calculate_performance_metrics

class OrbMlSimulator:
    def __init__(self, config_path='../config/config.yaml'): # Default relative path assumes running from notebook dir parent
        """Initializes the simulator by loading config, models, and data."""
        print("Initializing ORB ML Simulator...")
        self.config = None
        self.features_df = None
        self.regime_model = None
        self.regime_scaler = None
        self.meta_model = None
        self.meta_scaler = None

        try:
            # Try absolute path first, then relative
            if not os.path.exists(config_path):
                 # Construct path assuming script is in src, config is in project_root/config
                 script_dir = os.path.dirname(__file__)
                 proj_root = os.path.abspath(os.path.join(script_dir, '..'))
                 config_path = os.path.join(proj_root, 'config', 'config.yaml')

            if not os.path.exists(config_path):
                 raise FileNotFoundError(f"Config file not found at specified or default path: {config_path}")

            with open(config_path, 'r') as f:
                self.config = yaml.safe_load(f)
            print(f"Configuration loaded from: {config_path}")

            # --- Construct Absolute Paths based on Config Path ---
            config_dir = os.path.dirname(config_path)
            project_root_sim = os.path.abspath(os.path.join(config_dir, '..'))

            model_dir_rel = self.config['data_paths'].get('models_dir', 'models/') # Get relative path from config
            proc_dir_rel = self.config['data_paths'].get('processed_dir', 'data/processed/')

            model_dir_abs = os.path.join(project_root_sim, model_dir_rel)
            processed_dir_abs = os.path.join(project_root_sim, proc_dir_rel)
            print(f"Simulator using absolute model path: {model_dir_abs}")
            print(f"Simulator using absolute processed data path: {processed_dir_abs}")
            # -----------------------------------------------------

            # --- Load Models and Scalers ---
            print("Loading models and scalers...")
            # Use absolute paths
            regime_model_path = os.path.join(model_dir_abs, self.config['model_files']['regime_model'])
            regime_scaler_path = os.path.join(model_dir_abs, self.config['model_files']['regime_scaler'])
            meta_model_path = os.path.join(model_dir_abs, self.config['model_files']['meta_model'])
            meta_scaler_path = os.path.join(model_dir_abs, self.config['model_files']['meta_scaler'])

            print(f" Attempting to load regime model from: {regime_model_path}")
            self.regime_model = joblib.load(regime_model_path)
            print(f" Attempting to load regime scaler from: {regime_scaler_path}")
            self.regime_scaler = joblib.load(regime_scaler_path)
            print(f" Attempting to load meta model from: {meta_model_path}")
            self.meta_model = joblib.load(meta_model_path)
            print(f" Attempting to load meta scaler from: {meta_scaler_path}")
            self.meta_scaler = joblib.load(meta_scaler_path)
            print("Models and scalers loaded successfully.")
            # --------------------------------

            # --- Load Feature Data ---
            print("Loading feature data...")
            # Use absolute path
            feature_file_abs = os.path.join(processed_dir_abs, self.config['processed_files']['features_labels'])
            print(f"Attempting to load feature data from: {feature_file_abs}")
            self.features_df = pd.read_parquet(feature_file_abs)

            # Ensure index is datetime and sorted
            if not pd.api.types.is_datetime64_any_dtype(self.features_df.index):
                 self.features_df.index = pd.to_datetime(self.features_df.index)
            if not self.features_df.index.is_monotonic_increasing:
                 print("Warning: Feature data index not sorted. Sorting...")
                 self.features_df.sort_index(inplace=True)

            print(f"Feature data loaded: {self.features_df.shape[0]} rows.")

            # Ensure required feature columns needed by simulation logic exist
            required_sim_cols = ['Open', 'High', 'Low', 'Close']
            if 'regime_model_params' in self.config and 'features' in self.config['regime_model_params']:
                required_sim_cols.extend(self.config['regime_model_params']['features'])
            if 'meta_model_params' in self.config and 'features' in self.config['meta_model_params']:
                 required_sim_cols.extend(self.config['meta_model_params']['features'])
            # Remove duplicates, handle potential None values if keys missing
            required_sim_cols = sorted(list(set(filter(None, required_sim_cols))))

            missing_sim_cols = [col for col in required_sim_cols if col not in self.features_df.columns]
            if missing_sim_cols:
                print(f"ERROR: Feature data missing required columns needed by simulation: {missing_sim_cols}")
                raise ValueError("Feature data missing required columns for simulation")
            print("Required simulation feature columns found in data.")
            # ------------------------

            # --- Strategy & Simulation Parameters ---
            print("Setting up strategy parameters...")
            self.params = self.config.get('strategy_params', {})
            self.sim_params = self.config.get('simulation_params', {})
            self.meta_params = self.config.get('meta_model_params', {})
            self.pos_sizing_params = self.config.get('position_sizing_params', {})

            # Convert times to datetime.time objects for comparison
            st = str(self.params.get('openingRangeStartTime', 830)).zfill(4)
            et = str(self.params.get('openingRangeEndTime', 845)).zfill(4)
            ct = str(self.params.get('cutoffTime', 930)).zfill(4)
            mt = str(self.params.get('marketCloseTime', 1500)).zfill(4)
            self.orb_start_time = time(int(st[:2]), int(st[2:]))
            self.orb_end_time = time(int(et[:2]), int(et[2:]))
            self.entry_cutoff_time = time(int(ct[:2]), int(ct[2:]))
            self.market_close_time = time(int(mt[:2]), int(mt[2:]))
            print(f" ORB Time: {self.orb_start_time} - {self.orb_end_time}")
            print(f" Cutoff: {self.entry_cutoff_time}, Market Close: {self.market_close_time}")

            self.simulated_trades = []
            self._reset_simulation_state()
            print("Simulator initialization complete.")
            # ---------------------------------------

        except FileNotFoundError as fnf_err:
             print(f"ERROR during initialization: File not found - {fnf_err}")
             traceback.print_exc()
             raise # Re-raise error to stop execution
        except KeyError as key_err:
             print(f"ERROR during initialization: Missing key in configuration - {key_err}")
             traceback.print_exc()
             raise
        except Exception as init_err:
             print(f"ERROR during simulator initialization: {init_err}")
             traceback.print_exc()
             raise


    def _reset_simulation_state(self):
        """Resets variables that track the current position and trade state."""
        self.current_pos = 0
        self.entry_price = 0.0
        self.stop_loss = 0.0
        self.take_profit = 0.0
        self.position_size = 0
        self.entry_time = None
        self.entry_bar_index = -1
        # Add MAE/MFE tracking if needed
        # self.current_trade_high = -float('inf')
        # self.current_trade_low = float('inf')

    def _reset_daily_vars(self):
        """Resets variables tracked daily."""
        self.daily_entry_taken = False
        self.opening_range_high = 0.0
        self.opening_range_low = float('inf')
        self.opening_range_complete = False

    def _calculate_orb(self, current_day_data):
        """Calculates the opening range H/L for the current day."""
        if current_day_data is None or current_day_data.empty:
            self.opening_range_complete = False
            return
        try:
            orb_mask = (current_day_data.index.time >= self.orb_start_time) & \
                       (current_day_data.index.time < self.orb_end_time)
            daily_orb_data = current_day_data.loc[orb_mask] # Use .loc for boolean mask

            if not daily_orb_data.empty:
                self.opening_range_high = daily_orb_data['High'].max()
                self.opening_range_low = daily_orb_data['Low'].min()
                if pd.notna(self.opening_range_high) and pd.notna(self.opening_range_low) and self.opening_range_high > self.opening_range_low:
                    self.opening_range_complete = True
                else:
                    # print(f"Warning: Invalid ORB on {current_day_data.index[0].date()} (H:{self.opening_range_high}, L:{self.opening_range_low}).")
                    self.opening_range_complete = False
            else:
                self.opening_range_complete = False
        except Exception as e:
            # print(f"ERROR calculating ORB for {current_day_data.index[0].date()}: {e}") # Can be noisy
            self.opening_range_complete = False

    def _record_trade(self, exit_time, exit_price, exit_reason):
        """Records the details of a completed trade."""
        if self.current_pos == 0 or self.entry_time is None: return

        profit_per_contract = (exit_price - self.entry_price) if self.current_pos == 1 else (self.entry_price - exit_price)
        # Incorporate commission/slippage from config if they exist
        commission = self.sim_params.get('commission_per_contract', 0.0) * abs(self.position_size) * 2 # Entry + Exit
        slippage = self.sim_params.get('slippage_per_trade', 0.0) # Assume per trade (entry+exit combined) or adjust logic
        total_profit = (profit_per_contract * abs(self.position_size)) - commission - slippage

        trade_details = {
            "Entry time": self.entry_time,
            "Exit time": exit_time,
            "Market pos.": "Long" if self.current_pos == 1 else "Short",
            "Qty": self.position_size,
            "Entry price": self.entry_price,
            "Exit price": exit_price,
            "Profit": total_profit,
            "Exit reason": exit_reason,
            # Add other details if needed
            # "Entry bar index": self.entry_bar_index,
            # "Exit bar index": self.current_bar_index
        }
        self.simulated_trades.append(trade_details)
        self._reset_simulation_state()


    def _check_exits(self, timestamp, current_bar_high, current_bar_low, current_bar_close):
         """Checks for Stop Loss, Take Profit, or End of Day exits."""
         if self.current_pos == 0: return

         exit_price = 0.0
         exit_reason = None
         current_time = timestamp.time()

         # Basic Intra-bar check (more sophisticated needed for exact fills)
         # Assumes worst price for SL (triggers if low/high touches SL)
         # Assumes best price for TP (triggers if low/high touches TP)
         if self.current_pos == 1: # Long
             if current_bar_low <= self.stop_loss:
                 exit_price = self.stop_loss; exit_reason = "Stop loss"
             elif current_bar_high >= self.take_profit:
                 exit_price = self.take_profit; exit_reason = "Profit target"
         elif self.current_pos == -1: # Short
             if current_bar_high >= self.stop_loss:
                 exit_price = self.stop_loss; exit_reason = "Stop loss"
             elif current_bar_low <= self.take_profit:
                 exit_price = self.take_profit; exit_reason = "Profit target"

         # Check End of Day Exit (force close at EOD bar close)
         if exit_reason is None and current_time >= self.market_close_time:
             exit_price = current_bar_close
             exit_reason = "Exit on session close"

         if exit_reason:
             self._record_trade(timestamp, exit_price, exit_reason)


    def _check_entry(self, timestamp, current_row):
        """Checks ORB breakout signals, applies ML filters, determines size, enters trade."""
        if self.current_pos != 0 or not self.opening_range_complete or self.daily_entry_taken: return

        current_time = timestamp.time()
        if not (self.orb_end_time <= current_time <= self.entry_cutoff_time): return

        close_price = current_row['Close']
        break_high = close_price > self.opening_range_high
        break_low = close_price < self.opening_range_low

        if break_high or break_low:
            # --- Potential Entry Signal: Apply ML Filters ---
            features_now = current_row # Use the current row which has lagged features
            regime_feature_names = self.config['regime_model_params']['features']
            meta_feature_names = self.config['meta_model_params']['features']

            # Ensure features needed for prediction are not NaN
            required_pred_features = list(set(regime_feature_names + [f for f in meta_feature_names if f != 'Regime']))
            if features_now[required_pred_features].isnull().any():
                # print(f"Skipping signal at {timestamp} due to NaN in required prediction features.")
                return

            # Predict Regime
            try:
                 current_regime = predict_regime(features_now, self.regime_model, self.regime_scaler, regime_feature_names)
            except Exception as e_regime: print(f"ERROR predicting regime at {timestamp}: {e_regime}"); return

            # Add regime for meta-model prediction
            features_now_with_regime = features_now.copy() # Avoid potential SettingWithCopyWarning
            features_now_with_regime['Regime'] = current_regime

            # Predict Meta-Label (Win Probability)
            try:
                 pred_win_class, proba_win = predict_trade_outcome(features_now_with_regime, self.meta_model, self.meta_scaler, meta_feature_names)
            except Exception as e_meta: print(f"ERROR predicting meta-label at {timestamp}: {e_meta}"); return

            # Make Decision (Filter based on probability threshold)
            prob_threshold = self.meta_params.get('probability_threshold', 0.5) # Default 0.5 if not in config
            if proba_win >= prob_threshold:
                # Determine Position Size
                size = get_position_size(
                    confidence=proba_win,
                    regime=current_regime,
                    base_size=self.params.get('basePositionSize', 1), # Default 1
                    confidence_threshold=self.pos_sizing_params.get('confidence_threshold', prob_threshold), # Use same threshold
                    regime_map=self.pos_sizing_params.get('regime_multipliers')
                )

                if size > 0:
                    # --- Execute Simulated Entry ---
                    self.entry_price = close_price
                    self.position_size = size
                    self.entry_time = timestamp
                    self.entry_bar_index = self.current_bar_index
                    rr_ratio = self.params.get('riskRewardRatio', 2.0) # Default 2.0

                    # Set SL/TP based on ORB range
                    if break_high:
                        self.current_pos = 1
                        self.stop_loss = self.opening_range_low
                        stop_distance = self.entry_price - self.stop_loss
                        if stop_distance <= 0: # Safety check
                             # print(f"Warning: Zero/Negative stop distance on Long entry {timestamp}. Skipping.")
                             self._reset_simulation_state(); return
                        self.take_profit = self.entry_price + (stop_distance * rr_ratio)
                        # print(f"{timestamp}: ML LONG @{self.entry_price:.2f}|Sz:{size}|SL:{self.stop_loss:.2f}|TP:{self.take_profit:.2f}|P:{proba_win:.2f}|R:{current_regime}")
                    elif break_low:
                        self.current_pos = -1
                        self.stop_loss = self.opening_range_high
                        stop_distance = self.stop_loss - self.entry_price
                        if stop_distance <= 0: # Safety check
                             # print(f"Warning: Zero/Negative stop distance on Short entry {timestamp}. Skipping.")
                             self._reset_simulation_state(); return
                        self.take_profit = self.entry_price - (stop_distance * rr_ratio)
                        # print(f"{timestamp}: ML SHORT @{self.entry_price:.2f}|Sz:{size}|SL:{self.stop_loss:.2f}|TP:{self.take_profit:.2f}|P:{proba_win:.2f}|R:{current_regime}")

                    self.daily_entry_taken = True
            # else: (Optional: log skipped trades)
            #    print(f"{timestamp}: Signal skipped by Meta-Label. Prob: {proba_win:.2f} < {prob_threshold}")


    def run_simulation(self, start_date=None, end_date=None):
        """Runs the backtesting simulation loop."""
        print("\nStarting simulation run...")
        # Validate data and models loaded
        if self.features_df is None or self.meta_model is None or self.meta_scaler is None or self.regime_model is None or self.regime_scaler is None:
             print("ERROR: Cannot run simulation. Data or models not loaded correctly during initialization.")
             return None

        # Filter data by date
        sim_data = self.features_df.copy()
        if start_date: sim_data = sim_data[sim_data.index >= pd.to_datetime(start_date)]
        if end_date: sim_data = sim_data[sim_data.index <= pd.to_datetime(end_date)]

        if sim_data.empty: print("ERROR: No data available for the simulation period."); return None
        print(f"Simulating from {sim_data.index.min()} to {sim_data.index.max()}...")

        self._reset_simulation_state()
        self._reset_daily_vars()
        current_day = None
        daily_data_cache = None # Cache daily data for ORB calc

        # Iterate through each bar/row
        print("Processing bars...")
        for i, (timestamp, row) in enumerate(tqdm(sim_data.iterrows(), total=sim_data.shape[0], desc="Simulating")):
            try:
                self.current_bar_index = i
                today = timestamp.date()

                # --- Daily Reset and ORB Calculation ---
                if today != current_day:
                    if self.current_pos != 0: # Close EOD positions if held overnight (shouldn't happen with EOD exit)
                        print(f"Warning: Closing position carried overnight at {timestamp}.")
                        self._record_trade(timestamp, row['Open'], "Overnight Close")

                    self._reset_daily_vars()
                    current_day = today
                    # Efficiently get data for the current day once
                    daily_data_cache = sim_data.loc[sim_data.index.date == today]
                    if not daily_data_cache.empty:
                        self._calculate_orb(daily_data_cache)
                    else: self.opening_range_complete = False

                # --- Process Bar ---
                # Skip processing if ORB not yet complete for the day
                if not self.opening_range_complete and timestamp.time() >= self.orb_end_time:
                    continue

                # Check exits before entries
                self._check_exits(timestamp, row['High'], row['Low'], row['Close'])

                # Check entries only if flat
                if self.current_pos == 0:
                    self._check_entry(timestamp, row)

            except Exception as loop_err:
                 print(f"\nERROR processing bar at {timestamp}: {loop_err}")
                 traceback.print_exc()
                 # Decide whether to continue or stop simulation on error
                 # continue # Skip bar on error
                 # break # Stop simulation on error
                 raise # Re-raise to stop immediately


        print("\nSimulation finished.")
        if not self.simulated_trades:
            print("Warning: No trades were simulated.")
            return pd.DataFrame()

        results_df = pd.DataFrame(self.simulated_trades)
        if not results_df.empty:
             # Ensure correct types and sort
             results_df['Entry time'] = pd.to_datetime(results_df['Entry time'])
             results_df['Exit time'] = pd.to_datetime(results_df['Exit time'])
             results_df.set_index('Exit time', inplace=True)
             results_df.sort_index(inplace=True)
             print(f"Total Trades Simulated: {len(results_df)}")
        return results_df

    def calculate_performance(self, results_df=None, initial_capital=100000):
        """Calculates performance metrics for the simulated trades."""
        if results_df is None:
            if not self.simulated_trades: print("No trades available to calculate performance."); return {}
            results_df = pd.DataFrame(self.simulated_trades)
            # Need Exit time as index if calculating from self.simulated_trades
            if 'Exit time' in results_df.columns:
                results_df['Exit time'] = pd.to_datetime(results_df['Exit time'])
                results_df = results_df.set_index('Exit time').sort_index()


        if results_df is None or results_df.empty:
             print("No trades to analyze.")
             return {}

        # Use the utility function
        metrics = calculate_performance_metrics(results_df, initial_capital)
        print("\n--- Simulation Performance Metrics ---")
        for key, value in metrics.items():
            # Format numbers nicely
            if isinstance(value, (int, float)):
                 print(f"{key+':':<30} {value:,.2f}")
            else:
                 print(f"{key+':':<30} {value}")
        print("------------------------------------")
        return metrics


# --- Example Usage (if run directly for testing) ---
if __name__ == '__main__':
    print("\nRunning simulation script directly for testing...")
    try:
        # Assume config is in ../config relative to this script in src
        script_dir = os.path.dirname(__file__)
        config_file_path = os.path.join(script_dir, '..', 'config', 'config.yaml')
        simulator = OrbMlSimulator(config_path=config_file_path)

        # Define simulation period (e.g., test period from config)
        sim_start = simulator.config.get('data_split', {}).get('validation_end') # Start after validation
        sim_end = None # Run till end of data

        simulated_trades_df = simulator.run_simulation(start_date=sim_start, end_date=sim_end)

        if simulated_trades_df is not None and not simulated_trades_df.empty:
            performance = simulator.calculate_performance(simulated_trades_df)

            # Save results (relative to project root)
            project_root = os.path.abspath(os.path.join(script_dir, '..'))
            results_dir_rel = simulator.config.get('data_paths',{}).get('results_dir', 'results')
            results_dir_abs = os.path.join(project_root, results_dir_rel)
            os.makedirs(results_dir_abs, exist_ok=True)

            log_path = os.path.join(results_dir_abs, 'backtest_log_ml_TEST.csv') # Add TEST suffix
            simulated_trades_df.to_csv(log_path)
            print(f"\nSimulated trade log saved to {log_path}")

            perf_path = os.path.join(results_dir_abs, 'performance_ml_enhanced_TEST.txt')
            with open(perf_path, 'w') as f:
                for key, value in performance.items(): f.write(f"{key}: {value}\n")
            print(f"Performance metrics saved to {perf_path}")
        else:
            print("No trades generated during simulation.")

    except Exception as main_err:
         print(f"\nFATAL ERROR in simulation test run: {main_err}")
         traceback.print_exc()