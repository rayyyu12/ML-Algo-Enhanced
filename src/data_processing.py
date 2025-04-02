# src/data_processing.py
import pandas as pd
import numpy as np # Needed for np.where
from datetime import datetime
import glob
import os

def parse_custom_datetime(date_str, time_str):
    """Parses the custom date (YYYYMMDD) and time (HHMMSS) format."""
    try:
        # Ensure inputs are strings and handle potential float representation
        date_str = str(int(float(date_str)))
        time_str = str(int(float(time_str))).zfill(6) # Ensure time is 6 digits (e.g., 90300 -> 090300)

        year = int(date_str[:4])
        month = int(date_str[4:6])
        day = int(date_str[6:8])
        hour = int(time_str[:2])
        minute = int(time_str[2:4])
        second = int(time_str[4:6])
        # Potential timezone awareness needed here - assume UTC or consistent exchange time for now
        # If data is CST, consider localizing:
        # import pytz
        # dt = datetime(year, month, day, hour, minute, second)
        # cst = pytz.timezone('America/Chicago')
        # return cst.localize(dt)
        return datetime(year, month, day, hour, minute, second)
    except Exception as e:
        # print(f"Error parsing datetime: Date='{date_str}', Time='{time_str}'. Error: {e}")
        # Return NaT (Not a Time) on error, which pandas handles well
        return pd.NaT

def load_price_data(data_dir, file_pattern="NQ*.txt"):
    """
    Loads and processes historical price data from multiple files matching a pattern.

    Args:
        data_dir (str): The directory containing the raw price files (e.g., 'data/raw/').
        file_pattern (str): Glob pattern to match the price files (e.g., "NQ*.txt").

    Returns:
        pandas.DataFrame or None: A single DataFrame containing all price data,
                                  sorted chronologically, or None if an error occurs.
    """
    all_files = glob.glob(os.path.join(data_dir, file_pattern))
    if not all_files:
        print(f"Error: No files found matching pattern '{file_pattern}' in directory '{data_dir}'")
        return None

    print(f"Found {len(all_files)} price data files to process...")
    df_list = []

    for i, filepath in enumerate(sorted(all_files)): # Sort files to process chronologically
        filename = os.path.basename(filepath)
        print(f"  Processing file {i+1}/{len(all_files)}: {filename}")
        try:
            # Adjust separator if needed (e.g., might be ',' if '.Last.txt' implies CSV)
            # Assuming the format remains: YYYYMMDD HHMMSS;Open;High;Low;Close;Volume
            df_single = pd.read_csv(filepath, sep=';', header=None,
                                    names=['DateTimeStr', 'Open', 'High', 'Low', 'Close', 'Volume'],
                                    on_bad_lines='warn') # Report issues with lines

            if 'DateTimeStr' not in df_single.columns or df_single['DateTimeStr'].isnull().all():
                 print(f"    Warning: 'DateTimeStr' column missing or empty in {filename}. Skipping file.")
                 continue

            # Split DateTimeStr into Date and Time parts robustly
            split_dt = df_single['DateTimeStr'].astype(str).str.split(' ', n=1, expand=True)
            # Handle cases where split might not produce 2 columns if format is unexpected
            if split_dt.shape[1] == 2:
                 df_single['DateStr'] = split_dt[0]
                 df_single['TimeStr'] = split_dt[1]
            else:
                 print(f"    Warning: Could not split DateTimeStr properly in {filename}. Skipping file.")
                 continue


            # Parse the combined datetime using the corrected function
            df_single['Timestamp'] = df_single.apply(lambda row: parse_custom_datetime(row['DateStr'], row['TimeStr']), axis=1)

            # Drop rows where timestamp parsing failed
            df_single.dropna(subset=['Timestamp'], inplace=True)

            if df_single.empty:
                print(f"    Warning: No valid timestamps after parsing in {filename}")
                continue

            df_single.set_index('Timestamp', inplace=True)
            df_single = df_single[['Open', 'High', 'Low', 'Close', 'Volume']] # Ensure column order & drop intermediate cols

            # Convert to numeric, coercing errors
            for col in df_single.columns:
                df_single[col] = pd.to_numeric(df_single[col], errors='coerce')

            df_single.dropna(inplace=True) # Drop rows with non-numeric OHLCV

            if not df_single.empty:
                df_list.append(df_single)
            else:
                print(f"    Warning: No valid numeric data after processing {filename}")

        except FileNotFoundError:
            print(f"    Error: File not found {filepath}")
        except pd.errors.EmptyDataError:
            print(f"    Warning: File is empty {filepath}")
        except Exception as e:
            print(f"    Error processing file {filepath}: {e}")
            import traceback
            traceback.print_exc() # Print full traceback for unexpected errors
            # Decide if you want to skip the file or stop the process
            # continue # Skip this file

    if not df_list:
        print("Error: No data could be loaded from any files.")
        return None

    # Concatenate all DataFrames and sort chronologically
    print("Concatenating dataframes...")
    full_df = pd.concat(df_list)
    print("Sorting by timestamp...")
    full_df.sort_index(inplace=True)

    # Check for and remove duplicate index entries (e.g., overlapping data)
    initial_rows = len(full_df)
    if full_df.index.has_duplicates:
        print("Removing duplicate timestamp entries...")
        full_df = full_df[~full_df.index.duplicated(keep='first')]
        removed_rows = initial_rows - len(full_df)
        print(f"Removed {removed_rows} duplicate timestamp entries.")


    print(f"\nFinished loading price data: {full_df.shape[0]} rows from {full_df.index.min()} to {full_df.index.max()}")

    return full_df


# --- CORRECTED load_trade_data function ---
def load_trade_data(filepath):
    """Loads the historical trade log and correctly handles currency formatting."""
    print(f"\nAttempting to load trade data from: {filepath}")
    try:
        # Specify dtype='str' initially to prevent pandas from guessing types poorly
        df = pd.read_csv(filepath, dtype='str')
        print(f"Loaded trade data: {df.shape[0]} trades")

        # --- Time Conversion ---
        print("Processing Entry/Exit times...")
        time_cols = ['Entry time', 'Exit time']
        # Try specific format first, fallback to generic parser
        specific_format = '%m/%d/%Y %I:%M:%S %p'
        parsed_ok = True
        for col in time_cols:
            if col in df.columns:
                 try:
                      df[col] = pd.to_datetime(df[col], format=specific_format)
                 except (ValueError, TypeError):
                      print(f"Warning: Could not parse '{col}' with format '{specific_format}', trying generic parser.")
                      try:
                           df[col] = pd.to_datetime(df[col])
                      except (ValueError, TypeError) as e_gen:
                           print(f"ERROR: Failed to parse '{col}' even with generic parser: {e_gen}")
                           df[col] = pd.NaT # Assign NaT on failure
                           parsed_ok = False
            else:
                 print(f"Warning: Column '{col}' not found in trade log.")
                 parsed_ok = False # Mark as not fully parsed if time cols missing

        if parsed_ok: print("Parsed Entry/Exit times successfully.")
        df.dropna(subset=time_cols, inplace=True) # Drop rows where time parsing failed
        if df.empty:
             print("ERROR: No valid trade entries left after time parsing/dropping NaNs.")
             return None


        # --- Corrected Cleaning for currency/numeric columns ---
        print("Processing currency/numeric columns...")
        # Define columns expected to have currency format potentially
        currency_cols = ['Profit', 'Cum. net profit', 'MAE', 'MFE', 'ETD', 'Commission', 'Entry price', 'Exit price']
        numeric_cols = ['Qty', 'Bars'] # Other numeric cols

        for col in currency_cols:
             if col in df.columns:
                  # Fill NaNs (which might be actual NaN or string 'nan') with '0' string
                  df[col] = df[col].fillna('0')
                  # Convert to string type for cleaning
                  s = df[col].astype(str)
                  # Store original strings for debugging if needed
                  # original_s = s.copy()
                  # Remove $ and thousand separators (,)
                  s = s.str.replace(r'[$,]', '', regex=True).str.strip()
                  # Check if it contains parentheses for negative
                  is_negative = s.str.contains(r'^\(.*\)$', regex=True) # Check start/end parenthesis
                  # Remove parentheses
                  s = s.str.replace(r'[()]', '', regex=True)
                  # Convert to numeric, coercing errors to NaN
                  numeric_s = pd.to_numeric(s, errors='coerce')
                  # Where conversion failed, try to see original value
                  # failed_indices = numeric_s.isnull() & ~original_s.isin(['', 'nan', 'None'])
                  # if failed_indices.any():
                  #    print(f"Warning: Could not convert some values in column '{col}':")
                  #    print(original_s[failed_indices].value_counts().head())
                  # Fill resulting NaNs with 0
                  numeric_s = numeric_s.fillna(0)
                  # Apply negative sign where needed based on original format
                  df[col] = np.where(is_negative, -numeric_s, numeric_s)
                  # print(f"Processed column: {col}") # Optional: uncomment for verbose output
             else:
                  print(f"Warning: Currency column '{col}' not found.")

        # Process other purely numeric columns
        for col in numeric_cols:
             if col in df.columns:
                  df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0).astype(int) # Convert to int
             else:
                  print(f"Warning: Numeric column '{col}' not found.")


        # Add a binary win/loss column for meta-labeling target
        # Consider a threshold slightly above 0 if needed to exclude tiny wins/losses
        win_threshold = 0.01 # Example: require > $0.01 profit to be a 'win'
        if 'Profit' in df.columns:
             df['Win'] = (df['Profit'] > win_threshold).astype(int)
             print(f"Calculated 'Win' column (using Profit > {win_threshold})")
             print("Sample 'Profit' and 'Win' values after cleaning:")
             print(df[['Profit', 'Win']].head(15)) # Show more rows
             print("\nProfit column stats:")
             print(df['Profit'].describe())
             print("\nWin column distribution:")
             print(df['Win'].value_counts(normalize=True))
        else:
             print("ERROR: 'Profit' column not found or processed correctly, cannot create 'Win' column.")
             return None # Cannot proceed without Profit

        # Final check for unexpected NaNs
        if df.isnull().any().any():
             print("\nWarning: NaNs found after processing:")
             print(df.isnull().sum()[df.isnull().sum() > 0])


        print(f"\nCleaned trade data. Final shape: {df.shape[0]} trades")
        # Sort trades by entry time - important for meta-labeling preparation
        df.sort_values(by='Entry time', inplace=True)
        return df

    except FileNotFoundError:
        print(f"Error: Trade data file not found at {filepath}")
        return None
    except Exception as e:
        print(f"Error loading/cleaning trade data: {e}")
        import traceback
        traceback.print_exc()
        return None
# --- End corrected function ---