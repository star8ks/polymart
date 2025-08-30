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
import numpy as np


INVESTMENT_CEILING = 2000
MAX_POSITION_MULT = 3
BUDGET_MULT = 3
MARKET_COUNT = 30

@dataclass
class PositionSizeResult:
    """Result of position sizing calculation"""
    trade_size: float
    max_size: float


@dataclass 
class Position:
    """Represents a trading position"""
    size: float
    avgPrice: float


def filter_selected_markets(markets_df: pd.DataFrame) -> pd.DataFrame:
    df = markets_df.copy()
    if 'attractiveness_score' not in df.columns:
        raise Exception("attractiveness_score column missing")

    df['attractiveness_score'] = pd.to_numeric(df['attractiveness_score'], errors='coerce')
    df_sorted = df.sort_values(by='attractiveness_score', ascending=False, na_position='last')

    top_n = MARKET_COUNT
    return df_sorted.head(top_n).reset_index(drop=True)


def calculate_position_sizes(): 
    total_liquidity = global_state.available_liquidity
    budget = total_liquidity * BUDGET_MULT
    total_sharpe = global_state.selected_markets_df['attractiveness_score'].sum()

    global_state.market_position_sizes = {}
    for _, row in global_state.selected_markets_df.iterrows():
        condition_id = str(row['condition_id'])
        sharpe = row['attractiveness_score']

        size = budget * (sharpe / total_sharpe)
        
        global_state.market_position_sizes[condition_id] = PositionSizeResult(
            trade_size=size,
            max_size=size * MAX_POSITION_MULT
        )
    
    floors = {row['condition_id']: float(row.get('min_size')) for _, row in global_state.selected_markets_df.iterrows()}
    ceilings = {row['condition_id']: INVESTMENT_CEILING for _, row in global_state.selected_markets_df.iterrows()}

    try:
        global_state.market_position_sizes = redistribute_for_bounds(global_state.market_position_sizes, floors, ceilings)
    except Exception as e:
        print(f"Error redistributing for bounds: {e}")
        global_state.market_position_sizes = filter_out_outbound_markets(global_state.market_position_sizes, floors, ceilings)
        
def filter_out_outbound_markets(position_sizes: dict[str, PositionSizeResult], floors: dict[str, float], ceilings: dict[str, float]):
    positions = position_sizes.copy()
    for k, v in positions.items():
        if v.trade_size < floors[k] or v.trade_size > ceilings[k]:
            positions[k].trade_size = 0
    return positions

def redistribute_for_bounds(position_sizes: dict[str, PositionSizeResult], floors: dict[str, float], ceilings: dict[str, float], tol: float = 1e-12, max_iter: int = 100) -> dict[str, PositionSizeResult]:
    """
    Redistribute trade sizes so each respects its own [floor_i, ceiling_i] bound
    while preserving the total sum of the original trade sizes.

    This solves the projection onto {x : floors <= x <= ceilings, sum(x) = sum(initial)}
    via a 1-D root-finding on the Lagrange multiplier.

    Args:
        position_sizes: Mapping of id -> PositionSizeResult (uses trade_size as the weight)
        floors: Mapping of id -> lower bound for trade_size
        ceilings: Mapping of id -> upper bound for trade_size
        tol: Tolerance for sum matching
        max_iter: Max iterations for bisection

    Returns:
        Dict[id, float]: Adjusted trade sizes satisfying bounds and target sum
    """

    if position_sizes is None or len(position_sizes) == 0:
        return {}

    keys = list(position_sizes.keys())
    w = np.asarray([float(position_sizes[k].trade_size) for k in keys], dtype=float)

    n = w.size
    lower = np.empty(n, dtype=float)
    upper = np.empty(n, dtype=float)
    for i, k in enumerate(keys):
        lower[i] = float(floors.get(k, -np.inf))
        upper[i] = float(ceilings.get(k, np.inf))

    # Ensure valid bounds
    if not np.all(lower <= upper + 1e-15):
        raise ValueError("Some lower bounds exceed upper bounds.")

    target_sum = float(w.sum())

    # Feasibility check
    finite_lower = np.where(np.isfinite(lower), lower, -1e300)
    finite_upper = np.where(np.isfinite(upper), upper, 1e300)
    min_possible = finite_lower.sum()
    max_possible = finite_upper.sum()
    if target_sum < min_possible - 1e-12 or target_sum > max_possible + 1e-12:
        raise ValueError(
            f"Infeasible target sum {target_sum:.6g}; must be in [{min_possible:.6g}, {max_possible:.6g}] given the bounds."
        )

    def sum_after_shift(lmbda: float) -> float:
        return float(np.clip(w + lmbda, lower, upper).sum())

    # Bracket lambda using bound-shift extremes (use finite parts to guarantee finite bracket)
    shift_lo = lower - w
    shift_hi = upper - w
    finite_lo = np.where(np.isfinite(shift_lo), shift_lo, 0.0)
    finite_hi = np.where(np.isfinite(shift_hi), shift_hi, 0.0)
    lo = np.min(finite_lo) - 1.0
    hi = np.max(finite_hi) + 1.0

    # Bisection on lambda
    for _ in range(max_iter):
        mid = 0.5 * (lo + hi)
        s = sum_after_shift(mid)
        if abs(s - target_sum) <= tol:
            lam = mid
            break
        if s < target_sum:
            lo = mid
        else:
            hi = mid
    else:
        lam = 0.5 * (lo + hi)

    x = np.clip(w + lam, lower, upper)

    # Final tiny normalization for residual
    diff = target_sum - float(x.sum())
    if abs(diff) > tol:
        free = (x > lower + 1e-14) & (x < upper - 1e-14)
        if np.any(free):
            x[free] += diff / free.sum()
            x = np.clip(x, lower, upper)

    return {k: PositionSizeResult(trade_size=float(x[i]), max_size=float(x[i]) * MAX_POSITION_MULT) for i, k in enumerate(keys)}

def get_enhanced_market_row(condition_id: str) -> Optional[pd.Series]:
    """
    Get market row data with enhanced position sizing information.
    
    Args:
        condition_id: The condition ID of the market
        
    Returns:
        Enhanced market row with position sizing data, or None if not found
    """
    active_markets = global_state.get_active_markets()
    if active_markets is None:
        return None
    
    # Find the market in selected_markets_df 
    matching_markets = active_markets[
        active_markets['condition_id'] == condition_id
    ]
    
    if len(matching_markets) == 0:
        return None
    
    market_row = matching_markets.iloc[0].copy()
    
    # Get position sizing information
    position_size_info = global_state.market_position_sizes.get(condition_id)
    if position_size_info:
        # Override trade_size and max_size with calculated values
        market_row['trade_size'] = position_size_info.trade_size
        market_row['max_size'] = position_size_info.max_size
    
    return market_row