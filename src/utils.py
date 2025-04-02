# src/utils.py
import pandas as pd
import numpy as np

def calculate_performance_metrics(trade_log_df, initial_capital=100000):
    """
    Calculates common performance metrics from a trade log DataFrame.
    Assumes trade_log_df has 'Profit' and 'Exit time' columns.
    """
    if trade_log_df.empty:
        return {
            "Total Net Profit": 0,
            "Gross Profit": 0,
            "Gross Loss": 0,
            "Profit Factor": 0,
            "Total Number of Trades": 0,
            "Percent Profitable": 0,
            "# Winning Trades": 0,
            "# Losing Trades": 0,
            "Avg Trade Net Profit": 0,
            "Avg Winning Trade": 0,
            "Avg Losing Trade": 0,
            "Ratio Avg Win / Avg Loss": 0,
            "Max Drawdown": 0,
            "Sharpe Ratio (Risk-Free=0)": 0,
            # Add Sortino, etc. if needed
        }

    total_profit = trade_log_df['Profit'].sum()
    gross_profit = trade_log_df[trade_log_df['Profit'] > 0]['Profit'].sum()
    gross_loss = trade_log_df[trade_log_df['Profit'] < 0]['Profit'].sum() # Already negative

    profit_factor = abs(gross_profit / gross_loss) if gross_loss != 0 else np.inf

    num_trades = len(trade_log_df)
    num_wins = len(trade_log_df[trade_log_df['Profit'] > 0])
    num_losses = len(trade_log_df[trade_log_df['Profit'] < 0])

    percent_profitable = (num_wins / num_trades) * 100 if num_trades > 0 else 0

    avg_trade = total_profit / num_trades if num_trades > 0 else 0
    avg_win = gross_profit / num_wins if num_wins > 0 else 0
    avg_loss = gross_loss / num_losses if num_losses > 0 else 0 # Already negative

    ratio_win_loss = abs(avg_win / avg_loss) if avg_loss != 0 else np.inf

    # Calculate Drawdown
    trade_log_df = trade_log_df.sort_values(by='Exit time') # Ensure sorted
    trade_log_df['Cumulative Profit'] = trade_log_df['Profit'].cumsum()
    trade_log_df['Equity Curve'] = initial_capital + trade_log_df['Cumulative Profit']
    trade_log_df['Running Max Equity'] = trade_log_df['Equity Curve'].cummax()
    trade_log_df['Drawdown'] = trade_log_df['Running Max Equity'] - trade_log_df['Equity Curve']
    max_drawdown = trade_log_df['Drawdown'].max()

    # Calculate Sharpe Ratio (Simplified - assumes daily returns if data spans long period)
    # A proper calculation requires daily equity values. This is an approximation based on trades.
    # Consider using libraries like `empyrical` or `pyfolio` if more precise metrics are needed.
    # Approximation: Use profit per trade as "returns"
    if num_trades > 1:
        profit_std_dev = trade_log_df['Profit'].std()
        sharpe_ratio = (avg_trade / profit_std_dev) * np.sqrt(252) if profit_std_dev != 0 else 0 # Annualized guess
    else:
        sharpe_ratio = 0


    metrics = {
        "Total Net Profit": total_profit,
        "Gross Profit": gross_profit,
        "Gross Loss": gross_loss,
        "Profit Factor": profit_factor,
        "Total Number of Trades": num_trades,
        "Percent Profitable": percent_profitable,
        "# Winning Trades": num_wins,
        "# Losing Trades": num_losses,
        "Avg Trade Net Profit": avg_trade,
        "Avg Winning Trade": avg_win,
        "Avg Losing Trade": avg_loss,
        "Ratio Avg Win / Avg Loss": ratio_win_loss,
        "Max Drawdown": max_drawdown,
        "Sharpe Ratio (Approx.)": sharpe_ratio,
    }
    return metrics