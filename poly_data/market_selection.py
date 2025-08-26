"""
Market Selection and Position Sizing Logic

This module contains the logic for:
1. Further filtering markets from the "Selected Markets" sheet data
2. Determining how much money to invest in each market

These are currently stub implementations that can be filled in with custom logic.
"""

from typing import Dict, Optional, Tuple
from dataclasses import dataclass
import pandas as pd
import poly_data.global_state as global_state


@dataclass
class PositionSizeResult:
    """Result of position sizing calculation"""
    trade_size: float
    max_size: float
    risk_multiplier: float
    sizing_reason: str


@dataclass 
class Position:
    """Represents a trading position"""
    size: float
    avgPrice: float


def filter_selected_markets(markets_df: pd.DataFrame) -> pd.DataFrame:
    """
    Apply additional filtering logic to markets from the Selected Markets sheet.
    
    This function allows you to implement custom logic to further filter the markets
    that come from the Google Sheets "Selected Markets" tab. You can add criteria like:
    - Market-specific risk checks
    - Liquidity requirements
    - Volatility thresholds
    - Time-based filters (e.g., markets closing soon)
    - Portfolio balance considerations
    
    Args:
        markets_df: DataFrame containing markets from Selected Markets sheet
                   merged with All Markets data
    
    Returns:
        Filtered DataFrame with markets that should be actively traded
        
    Example filtering logic you might implement:
    - Remove markets with volatility_sum > custom_threshold
    - Remove markets with insufficient liquidity
    - Remove markets that conflict with existing positions
    - Apply custom scoring/ranking logic
    """
    
    # STUB IMPLEMENTATION - Currently returns all markets unchanged
    # TODO: Implement your custom filtering logic here
    
    print(f"Market filter stub: Processing {len(markets_df)} markets from Selected Markets")
    
    # Example stub logic (commented out):
    # filtered_df = markets_df[
    #     (markets_df['volatility_sum'] < 15) &  # Lower volatility threshold
    #     (markets_df['attractiveness_score'] > 5) &  # Higher attractiveness requirement
    #     (markets_df['best_ask'] - markets_df['best_bid'] < 0.05)  # Tighter spread requirement
    # ]
    
    # For now, return all markets unchanged
    filtered_df = markets_df.copy()
    
    print(f"Market filter stub: Returning {len(filtered_df)} markets after filtering")
    return filtered_df


def calculate_position_size(
    market_row: pd.Series, 
    current_positions: Dict[str, Dict[str, float]], 
    total_portfolio_value: Optional[float] = None
) -> PositionSizeResult:
    """
    Calculate how much money to invest in a specific market.
    
    This function determines the position size for each market based on:
    - Market characteristics (volatility, liquidity, attractiveness)
    - Current portfolio state
    - Risk management rules
    - Available capital
    
    Args:
        market_row: Single row from the filtered markets DataFrame containing
                   all market data (rewards, volatility, etc.)
        current_positions: Current positions across all tokens
                         {token_id: {'size': float, 'avgPrice': float}}
        total_portfolio_value: Total portfolio value for percentage-based sizing
    
    Returns:
        Position sizing information
              
    Example sizing logic you might implement:
    - Higher attractiveness_score -> larger position
    - Higher volatility -> smaller position  
    - Existing position -> smaller additional size
    - Portfolio concentration limits
    - Kelly criterion or other position sizing formulas
    """
    
    # STUB IMPLEMENTATION - Currently uses market's existing trade_size
    # TODO: Implement your custom position sizing logic here
    
    market_question: str = market_row.get('question', 'Unknown Market')
    base_trade_size: float = float(market_row.get('trade_size', 100))  # Default fallback
    existing_max_size: float = float(market_row.get('max_size', base_trade_size))
    
    print(f"Position sizing stub for: {market_question[:50]}...")
    
    # Example stub logic (commented out):
    # attractiveness: float = float(market_row.get('attractiveness_score', 1))
    # volatility: float = float(market_row.get('volatility_sum', 10))
    # 
    # # Higher attractiveness -> larger size, higher volatility -> smaller size
    # risk_multiplier: float = min(2.0, max(0.5, attractiveness / 10)) * min(1.0, 20 / max(volatility, 1))
    # 
    # # Check for existing positions to avoid overconcentration
    # token1_pos: float = current_positions.get(str(market_row['token1']), {}).get('size', 0.0)
    # token2_pos: float = current_positions.get(str(market_row['token2']), {}).get('size', 0.0)
    # existing_exposure: float = abs(token1_pos) + abs(token2_pos)
    # 
    # if existing_exposure > base_trade_size * 0.5:
    #     risk_multiplier *= 0.5  # Reduce size if already have exposure
    
    # For now, use existing trade_size logic
    risk_multiplier: float = 1.0
    sizing_reason: str = "Using default trade_size from sheet (stub implementation)"
    
    result = PositionSizeResult(
        trade_size=base_trade_size,
        max_size=existing_max_size,
        risk_multiplier=risk_multiplier,
        sizing_reason=sizing_reason
    )
    
    print(f"Position sizing stub result: trade_size={result.trade_size}, max_size={result.max_size}, multiplier={result.risk_multiplier}")
    
    return result


def get_active_markets() -> pd.DataFrame:
    """
    Get the list of markets that should be actively traded, after applying
    both sheet selection and custom filtering logic.
    
    Returns:
        Final filtered markets ready for trading
    """
    if global_state.df is None or len(global_state.df) == 0:
        return pd.DataFrame()
    
    # Apply custom market filtering
    filtered_markets: pd.DataFrame = filter_selected_markets(global_state.df)
    
    return filtered_markets


def get_market_position_sizing(market_row: pd.Series) -> PositionSizeResult:
    """
    Get position sizing information for a specific market.
    
    Args:
        market_row: Market data row
        
    Returns:
        Position sizing information
    """
    return calculate_position_size(
        market_row, 
        global_state.positions,
        total_portfolio_value=None  # Could calculate this if needed
    )


def get_enhanced_market_row(condition_id: str) -> Optional[pd.Series]:
    """
    Get market row data with enhanced position sizing information.
    
    Args:
        condition_id: The condition ID of the market
        
    Returns:
        Enhanced market row with position sizing data, or None if not found
    """
    if global_state.selected_markets_df is None:
        return None
    
    # Find the market in selected_markets_df 
    matching_markets = global_state.selected_markets_df[
        global_state.selected_markets_df['condition_id'] == condition_id
    ]
    
    if len(matching_markets) == 0:
        return None
    
    market_row = matching_markets.iloc[0].copy()
    
    # Get position sizing information
    position_size_info = global_state.market_position_sizes.get(condition_id)
    if position_size_info:
        # Override trade_size and max_size with calculated values
        calculated_trade_size = position_size_info.trade_size * position_size_info.risk_multiplier
        market_row['trade_size'] = calculated_trade_size
        market_row['max_size'] = position_size_info.max_size
        market_row['risk_multiplier'] = position_size_info.risk_multiplier
        market_row['sizing_reason'] = position_size_info.sizing_reason
    
    return market_row