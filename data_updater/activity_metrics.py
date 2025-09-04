"""
Market Activity Metrics Calculation

This module calculates various activity metrics for Polymarket tokens:
1. Volume of activity from trades for the last N days
2. Decay weighted volume (more recent trades have higher emphasis)
3. Volume inside some fixed spread
4. Trade frequency (average in the last N days)
5. Number of unique traders and transactions

Note: In Polymarket terminology:
- A market has two tokens (Yes/No outcomes)
- Each token represents one side of the bet
- The trades API uses token_id in the 'market' parameter
"""

import pandas as pd
import numpy as np
import requests
import time
from datetime import datetime, timedelta
from typing import Dict, List, Tuple
from logan import Logan
from configuration import TCNF


def get_market_trades_data(condition_id: str) -> pd.DataFrame:
    """
    Fetch trades data for a given market (condition_id) over the configured lookback period.
    
    Args:
        condition_id: The condition ID (market ID) to fetch trades for
        
    Returns:
        DataFrame with trade data including timestamps, prices, sizes, etc.
    """
    try:
        # Calculate timestamp for lookback period
        end_time = int(datetime.now().timestamp())
        start_time = int((datetime.now() - timedelta(days=TCNF.ACTIVITY_LOOKBACK_DAYS)).timestamp())
        
        # Fetch trades from Polymarket Data API for the entire market (condition_id)
        url = "https://data-api.polymarket.com/trades"
        params = {
            'market': condition_id,  # Use condition_id (market) to get all trades for the market
            'limit': 10000  # Get maximum trades for analysis
        }
        
        response = requests.get(url, params=params)
        response.raise_for_status()
        
        trades_data = response.json()
        if not trades_data:
            raise Exception(f"No trades data found for market {condition_id}")
            
        trades_df = pd.DataFrame(trades_data)
        
        if trades_df.empty:
            raise Exception(f"Empty trades data for market {condition_id}")
        
        # Convert timestamps to datetime (API uses 'timestamp' field)
        trades_df['match_time'] = pd.to_datetime(trades_df['timestamp'], unit='s')
        trades_df['price'] = pd.to_numeric(trades_df['price'])
        trades_df['size'] = pd.to_numeric(trades_df['size'])
        
        return trades_df
        
    except Exception as e:
        Logan.error(
            f"Error fetching trades data for market {condition_id}: {e}",
            namespace="data_updater.activity_metrics",
            exception=e
        )
        return pd.DataFrame()


def calculate_volume_metrics(trades_df: pd.DataFrame) -> Dict[str, float]:
    """
    Calculate various volume metrics from trades data.
    
    Args:
        trades_df: DataFrame with trade data
        
    Returns:
        Dictionary with volume metrics
    """
    if trades_df.empty:
        return {
            'total_volume': 0.0,
            'volume_usd': 0.0,
            'decay_weighted_volume': 0.0,
            'avg_daily_volume': 0.0
        }
    
    # Calculate total volume (sum of all trade sizes)
    total_volume = trades_df['size'].sum()
    
    # Calculate volume in USD terms (size * price)
    volume_usd = (trades_df['size'] * trades_df['price']).sum()
    
    # Calculate decay weighted volume (more recent trades weighted higher)
    current_time = datetime.now()
    trades_df['hours_ago'] = (current_time - trades_df['match_time']).dt.total_seconds() / 3600
    
    # Use exponential decay with configured half-life
    decay_factor = 0.5 ** (trades_df['hours_ago'] / TCNF.DECAY_HALF_LIFE_HOURS)
    decay_weighted_volume = (trades_df['size'] * decay_factor).sum()
    
    # Average daily volume
    avg_daily_volume = total_volume / max(TCNF.ACTIVITY_LOOKBACK_DAYS, 1)
    
    return {
        'total_volume': round(total_volume, 2),
        'volume_usd': round(volume_usd, 2),
        'decay_weighted_volume': round(decay_weighted_volume, 2),
        'avg_daily_volume': round(avg_daily_volume, 2)
    }




def calculate_trade_frequency(trades_df: pd.DataFrame) -> Dict[str, float]:
    """
    Calculate trade frequency metrics.
    
    Args:
        trades_df: DataFrame with trade data
        
    Returns:
        Dictionary with frequency metrics
    """
    if trades_df.empty:
        return {
            'total_trades': 0,
            'avg_trades_per_day': 0.0,
            'avg_trades_per_hour': 0.0
        }
    
    total_trades = len(trades_df)
    avg_trades_per_day = total_trades / max(TCNF.ACTIVITY_LOOKBACK_DAYS, 1)
    avg_trades_per_hour = total_trades / max(TCNF.ACTIVITY_LOOKBACK_DAYS * 24, 1)
    
    return {
        'total_trades': total_trades,
        'avg_trades_per_day': round(avg_trades_per_day, 2),
        'avg_trades_per_hour': round(avg_trades_per_hour, 4)
    }


# FIRST TODO: Remove data not available in the new API
def calculate_unique_participants(trades_df: pd.DataFrame) -> Dict[str, int]:
    """
    Calculate number of unique traders and transactions.
    
    Args:
        trades_df: DataFrame with trade data
        
    Returns:
        Dictionary with participant metrics
    """
    if trades_df.empty:
        return {
            'unique_makers': 0,
            'unique_takers': 0,
            'unique_traders': 0,
            'unique_transactions': 0
        }
    
    # Count unique wallets (the API uses 'proxyWallet' field)
    unique_wallets = trades_df['proxyWallet'].nunique() if 'proxyWallet' in trades_df.columns else 0
    
    # For this API, we don't have separate maker/taker addresses, just proxyWallet
    unique_makers = 0  # Not available in this API
    unique_takers = 0  # Not available in this API
    unique_traders = unique_wallets
    
    # Unique transactions (using 'transactionHash' field)
    unique_transactions = trades_df['transactionHash'].nunique() if 'transactionHash' in trades_df.columns else len(trades_df)
    
    return {
        'unique_makers': unique_makers,
        'unique_takers': unique_takers,
        'unique_traders': unique_traders,
        'unique_transactions': unique_transactions
    }


def calculate_market_activity_metrics(condition_id: str, best_bid: float, best_ask: float) -> Dict[str, float]:
    """
    Calculate all activity metrics for a given market.
    
    Args:
        condition_id: The condition ID (market ID) to analyze
        best_bid: Current best bid price
        best_ask: Current best ask price
        
    Returns:
        Dictionary with all activity metrics
    """
    try:
        # Fetch trades data for the entire market
        trades_df = get_market_trades_data(condition_id)
        
        if trades_df.empty:
            Logan.warn(
                f"No trades data found for market {condition_id}",
                namespace="data_updater.activity_metrics"
            )
            
        # Calculate all metrics
        volume_metrics = calculate_volume_metrics(trades_df)
        frequency_metrics = calculate_trade_frequency(trades_df)
        participant_metrics = calculate_unique_participants(trades_df)
        
        # Combine all metrics
        all_metrics = {
            **volume_metrics,
            **frequency_metrics,
            **participant_metrics
        }
        
        Logan.info(
            f"Calculated activity metrics for market {condition_id}: {len(trades_df)} trades over {TCNF.ACTIVITY_LOOKBACK_DAYS} days",
            namespace="data_updater.activity_metrics"
        )
        
        return all_metrics
        
    except Exception as e:
        Logan.error(
            f"Error calculating activity metrics for market {condition_id}: {e}",
            namespace="data_updater.activity_metrics",
            exception=e
        )
        return {}


def add_activity_metrics_to_market_data(market_row: Dict) -> Dict:
    """
    Add activity metrics to a single market data row.
    This calculates metrics for the entire market (both tokens combined).
    
    Args:
        market_row: Dictionary containing market data
        
    Returns:
        Market row enhanced with activity metrics
    """
    condition_id = market_row.get('condition_id')
    best_bid = market_row.get('best_bid', 0)
    best_ask = market_row.get('best_ask', 0)
    
    if not condition_id:
        Logan.warn(
            "No condition_id found in market row, skipping activity metrics",
            namespace="data_updater.activity_metrics"
        )
        return market_row
    
    # Calculate metrics for the entire market over configured lookback period
    activity_metrics = calculate_market_activity_metrics(condition_id, best_bid, best_ask)
    
    # Add metrics to market row
    enhanced_row = market_row.copy()
    enhanced_row.update(activity_metrics)
    
    return enhanced_row


