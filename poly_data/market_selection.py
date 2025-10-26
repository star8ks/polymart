"""
Market Selection and Position Sizing Logic

This module contains the logic for:
1. Further filtering markets from the "Selected Markets" sheet data
2. Determining how much money to invest in each market

These are currently stub implementations that can be filled in with custom logic.
"""

from typing import Any, Dict, Optional, Tuple
from dataclasses import dataclass
import threading
import time
from copy import deepcopy

import numpy as np
import pandas as pd

import poly_data.global_state as global_state
from configuration import TCNF
from gspread_dataframe import set_with_dataframe
from logan import Logan
from poly_utils.google_utils import get_spreadsheet


@dataclass
class PositionSizeResult:
    """Container for trade sizing outputs."""

    trade_size: float
    max_size: float


_market_snapshot_cache_lock = threading.Lock()
_market_snapshot_cache: Dict[str, Tuple[float, Dict[str, Any]]] = {}


def _normalize_snapshot_values(row_dict: Dict[str, Any]) -> Dict[str, Any]:
    cleaned: Dict[str, Any] = {}
    for key, value in row_dict.items():
        if pd.isna(value):
            cleaned[key] = None
        elif isinstance(value, str):
            cleaned[key] = value.strip()
        else:
            cleaned[key] = value

    if 'condition_id' in cleaned:
        cleaned['condition_id'] = str(cleaned['condition_id'])

    return cleaned


def _download_market_snapshot(condition_id: str) -> Optional[Dict[str, Any]]:
    condition_id = str(condition_id)
    if not condition_id:
        return None

    try:
        spreadsheet = get_spreadsheet()
    except FileNotFoundError:
        try:
            spreadsheet = get_spreadsheet(read_only=True)
        except Exception as exc:
            Logan.error(
                f"Unable to open spreadsheet in read-only mode while refreshing market {condition_id}",
                namespace="poly_data.market_selection",
                exception=exc
            )
            return None
    except Exception as exc:
        Logan.error(
            f"Failed to open spreadsheet while refreshing market {condition_id}",
            namespace="poly_data.market_selection",
            exception=exc
        )
        return None

    try:
        worksheet = spreadsheet.worksheet("All Markets")
    except Exception as exc:
        Logan.error(
            f"Unable to access 'All Markets' sheet while refreshing market {condition_id}",
            namespace="poly_data.market_selection",
            exception=exc
        )
        return None

    row_dict: Optional[Dict[str, Any]] = None

    try:
        finder = getattr(worksheet, "find", None)
        row_values_method = getattr(worksheet, "row_values", None)
        if callable(finder) and callable(row_values_method):
            cell = finder(condition_id)
            if cell:
                header = row_values_method(1)
                row_values = row_values_method(cell.row)
                if header:
                    if len(row_values) < len(header):
                        row_values = row_values + \
                            [''] * (len(header) - len(row_values))
                    row_dict = dict(zip(header, row_values))
                elif row_values:
                    row_dict = {'condition_id': condition_id}
    except Exception:
        Logan.warn(
            f"Direct row lookup failed for market {condition_id}, falling back to full sheet fetch",
            namespace="poly_data.market_selection"
        )

    if row_dict is None:
        try:
            records = worksheet.get_all_records()
        except Exception as exc:
            Logan.error(
                f"Failed to fetch sheet records while refreshing market {condition_id}",
                namespace="poly_data.market_selection",
                exception=exc
            )
            return None

        if not records:
            return None

        try:
            df = pd.DataFrame(records)
        except Exception as exc:
            Logan.error(
                f"Failed to convert sheet records to DataFrame while refreshing market {condition_id}",
                namespace="poly_data.market_selection",
                exception=exc
            )
            return None

        if 'condition_id' not in df.columns:
            Logan.warn(
                "Sheet snapshot missing 'condition_id' column",
                namespace="poly_data.market_selection"
            )
            return None

        df['condition_id'] = df['condition_id'].astype(str)
        matches = df[df['condition_id'] == condition_id]
        if matches.empty:
            return None

        row_dict = matches.iloc[0].to_dict()

    if row_dict is None:
        return None

    row_dict['condition_id'] = condition_id
    return _normalize_snapshot_values(row_dict)


def get_latest_market_snapshot(condition_id: str, max_age: Optional[float] = None) -> Optional[Dict[str, Any]]:
    """Fetch a fresh market snapshot from Google Sheets with optional caching."""

    condition_id = str(condition_id)
    if not condition_id:
        return None

    ttl = TCNF.MARKET_ROW_REFRESH_SECONDS if max_age is None else max_age
    now = time.time()

    if ttl and ttl > 0:
        with _market_snapshot_cache_lock:
            cached = _market_snapshot_cache.get(condition_id)
            if cached and now - cached[0] <= ttl:
                return deepcopy(cached[1])

    snapshot = _download_market_snapshot(condition_id)
    if snapshot is None:
        return None

    with _market_snapshot_cache_lock:
        _market_snapshot_cache[condition_id] = (now, snapshot)

    return deepcopy(snapshot)


def filter_selected_markets(markets_df: pd.DataFrame) -> pd.DataFrame:
    """
    Filter and select markets based on various criteria while honoring manual overrides.

    Applies the following filters in order:
    1. Volatility filtering (volatility_sum threshold)
    2. Attractiveness score minimum threshold
    3. Midpoint range filtering (avoid extreme probabilities)
    4. Top-N selection by attractiveness_score (manual selections always included)
    """
    df = markets_df.copy()
    initial_count = len(df)

    # Ensure required columns exist
    required_cols = [
        'attractiveness_score',
        'volatility_sum',
        'best_bid',
        'best_ask',
        'rewards_daily_rate',
        'min_size',
        'max_spread',
        'condition_id'
    ]
    missing_cols = [col for col in required_cols if col not in df.columns]
    if missing_cols:
        raise Exception(f"Required columns missing: {missing_cols}")

    # Define activity columns that might be present
    activity_cols = [
        'total_volume',
        'volume_usd',
        'decay_weighted_volume',
        'avg_trades_per_day',
        'unique_traders'
    ]

    # Convert columns to numeric, handling any string/NaN values
    numeric_cols = [
        'attractiveness_score',
        'volatility_sum',
        'best_bid',
        'best_ask',
        'gm_reward_per_100',
        'market_order_imbalance',
        'rewards_daily_rate',
        'min_size',
        'max_spread',
        'reward_half_spread',
        'reward_bid_floor',
        'reward_ask_ceiling'
    ] + [col for col in activity_cols if col in df.columns]

    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')

    # Calculate midpoint if not already present
    if 'midpoint' not in df.columns:
        df['midpoint'] = (df['best_bid'] + df['best_ask']) / 2

    # Detect manual overrides (column name normalized to manual_select)
    manual_col = next(
        (
            col
            for col in df.columns
            if col and col.strip().lower().replace(' ', '_') == 'manual_select'
        ),
        None
    )

    manual_rows = pd.DataFrame(columns=df.columns)
    manual_ids: list[str] = []
    if manual_col:
        manual_series = df[manual_col].fillna('').astype(str).str.strip()
        manual_series = manual_series.replace(
            {'nan': '', 'NaN': '', 'None': ''})
        manual_mask = manual_series != ''
        if manual_mask.any():
            manual_rows = df[manual_mask].copy()
            manual_ids = [str(cid)
                          for cid in manual_rows['condition_id'].astype(str)]
            Logan.info(
                f"Manual overrides detected: {len(manual_ids)} markets flagged via '{manual_col}'",
                namespace="poly_data.market_selection"
            )
            Logan.info(
                f"Manual override IDs: {manual_ids}",
                namespace="poly_data.market_selection"
            )

    manual_id_set = set(manual_ids)

    def ensure_manual(current_df: pd.DataFrame) -> pd.DataFrame:
        if not manual_id_set:
            return current_df

        current_df = current_df.copy()
        current_ids = set(current_df['condition_id'].astype(str))
        missing_ids = [cid for cid in manual_ids if cid not in current_ids]
        if missing_ids:
            additions = manual_rows[
                manual_rows['condition_id'].astype(str).isin(missing_ids)
            ].copy()
            additions = additions.reindex(
                columns=current_df.columns, fill_value=pd.NA)
            current_df = pd.concat(
                [additions, current_df], ignore_index=True, sort=False)
        return current_df

    df = ensure_manual(df)

    # Focus on markets with active liquidity rewards
    df = df[
        (df['rewards_daily_rate'] > 0)
        & (df['min_size'] > 0)
        & (df['max_spread'] > 0)
    ].copy()
    df = ensure_manual(df)
    Logan.info(
        f"After rewards filter: {len(df)}/{initial_count} markets offering incentives",
        namespace="poly_data.market_selection"
    )

    # Filter by volatility sum
    if TCNF.MAX_VOLATILITY_SUM > 0:
        df = df[df['volatility_sum'] <= TCNF.MAX_VOLATILITY_SUM].copy()
        df = ensure_manual(df)
        avg_attractiveness = df['attractiveness_score'].mean() if len(
            df) > 0 else 0
        avg_gm_reward = df['gm_reward_per_100'].mean() if len(df) > 0 else 0
        Logan.info(
            f"After volatility filter (≤{TCNF.MAX_VOLATILITY_SUM}): {len(df)}/{initial_count} markets "
            f"(avg attractiveness: {avg_attractiveness:.2f}, avg GM reward: {avg_gm_reward:.2f})",
            namespace="poly_data.market_selection"
        )

    # Filter by minimum attractiveness score
    if TCNF.MIN_ATTRACTIVENESS_SCORE > 0:
        df = df[df['attractiveness_score'] >=
                TCNF.MIN_ATTRACTIVENESS_SCORE].copy()
        df = ensure_manual(df)
        avg_attractiveness = df['attractiveness_score'].mean() if len(
            df) > 0 else 0
        avg_gm_reward = df['gm_reward_per_100'].mean() if len(df) > 0 else 0
        Logan.info(
            f"After attractiveness filter (≥{TCNF.MIN_ATTRACTIVENESS_SCORE}): {len(df)}/{initial_count} markets "
            f"(avg attractiveness: {avg_attractiveness:.2f}, avg GM reward: {avg_gm_reward:.2f})",
            namespace="poly_data.market_selection"
        )

    # Filter by midpoint range (avoid extreme probabilities)
    df = df[
        (df['midpoint'] >= TCNF.MIN_PRICE_LIMIT)
        & (df['midpoint'] <= TCNF.MAX_PRICE_LIMIT)
    ].copy()
    df = ensure_manual(df)
    avg_attractiveness = df['attractiveness_score'].mean() if len(
        df) > 0 else 0
    avg_gm_reward = df['gm_reward_per_100'].mean() if len(df) > 0 else 0
    Logan.info(
        f"After midpoint filter ({TCNF.MIN_PRICE_LIMIT}-{TCNF.MAX_PRICE_LIMIT}): {len(df)}/{initial_count} markets "
        f"(avg attractiveness: {avg_attractiveness:.2f}, avg GM reward: {avg_gm_reward:.2f})",
        namespace="poly_data.market_selection"
    )

    # Filter out markets with spread greater than max spread
    if 'spread' in df.columns:
        df = df[(df['spread'] * 100 <= df['max_spread'])].copy()
        df = ensure_manual(df)
        avg_attractiveness = df['attractiveness_score'].mean() if len(
            df) > 0 else 0
        avg_gm_reward = df['gm_reward_per_100'].mean() if len(df) > 0 else 0
        Logan.info(
            f"After max spread filter (spread <= max spread): {len(df)}/{initial_count} markets "
            f"(avg attractiveness: {avg_attractiveness:.2f}, avg GM reward: {avg_gm_reward:.2f})",
            namespace="poly_data.market_selection"
        )

    # Filter by market order imbalance
    df = df[df['market_order_imbalance'].abs(
    ) <= TCNF.MAX_MARKET_ORDER_IMBALANCE].copy()
    df = ensure_manual(df)
    avg_attractiveness = df['attractiveness_score'].mean() if len(
        df) > 0 else 0
    avg_gm_reward = df['gm_reward_per_100'].mean() if len(df) > 0 else 0
    Logan.info(
        f"After market order imbalance filter (≤{TCNF.MAX_MARKET_ORDER_IMBALANCE}): {len(df)}/{initial_count} markets "
        f"(avg attractiveness: {avg_attractiveness:.2f}, avg GM reward: {avg_gm_reward:.2f})",
        namespace="poly_data.market_selection"
    )

    # Activity-based filtering (only if activity metrics are available)
    if 'total_volume' in df.columns:
        condition = df['total_volume'].fillna(0) >= TCNF.MIN_TOTAL_VOLUME
        df = df[condition].copy()
        df = ensure_manual(df)
        avg_attractiveness = df['attractiveness_score'].mean() if len(
            df) > 0 else 0
        avg_gm_reward = df['gm_reward_per_100'].mean() if len(df) > 0 else 0
        Logan.info(
            f"After total volume filter (≥{TCNF.MIN_TOTAL_VOLUME}): {len(df)}/{initial_count} markets "
            f"(avg attractiveness: {avg_attractiveness:.2f}, avg GM reward: {avg_gm_reward:.2f})",
            namespace="poly_data.market_selection"
        )

        if 'volume_usd' in df.columns:
            condition = df['volume_usd'].fillna(0) >= TCNF.MIN_VOLUME_USD
            df = df[condition].copy()
            df = ensure_manual(df)
            avg_attractiveness = df['attractiveness_score'].mean() if len(
                df) > 0 else 0
            avg_gm_reward = df['gm_reward_per_100'].mean() if len(
                df) > 0 else 0
            Logan.info(
                f"After USD volume filter (≥{TCNF.MIN_VOLUME_USD}): {len(df)}/{initial_count} markets "
                f"(avg attractiveness: {avg_attractiveness:.2f}, avg GM reward: {avg_gm_reward:.2f})",
                namespace="poly_data.market_selection"
            )

        if 'decay_weighted_volume' in df.columns:
            condition = df['decay_weighted_volume'].fillna(
                0) >= TCNF.MIN_DECAY_WEIGHTED_VOLUME
            df = df[condition].copy()
            df = ensure_manual(df)
            avg_attractiveness = df['attractiveness_score'].mean() if len(
                df) > 0 else 0
            avg_gm_reward = df['gm_reward_per_100'].mean() if len(
                df) > 0 else 0
            Logan.info(
                f"After decay-weighted volume filter (≥{TCNF.MIN_DECAY_WEIGHTED_VOLUME}): {len(df)}/{initial_count} markets "
                f"(avg attractiveness: {avg_attractiveness:.2f}, avg GM reward: {avg_gm_reward:.2f})",
                namespace="poly_data.market_selection"
            )

        if 'avg_trades_per_day' in df.columns:
            condition = df['avg_trades_per_day'].fillna(
                0) >= TCNF.MIN_AVG_TRADES_PER_DAY
            df = df[condition].copy()
            df = ensure_manual(df)
            avg_attractiveness = df['attractiveness_score'].mean() if len(
                df) > 0 else 0
            avg_gm_reward = df['gm_reward_per_100'].mean() if len(
                df) > 0 else 0
            Logan.info(
                f"After avg trades per day filter (≥{TCNF.MIN_AVG_TRADES_PER_DAY}): {len(df)}/{initial_count} markets "
                f"(avg attractiveness: {avg_attractiveness:.2f}, avg GM reward: {avg_gm_reward:.2f})",
                namespace="poly_data.market_selection"
            )

        if 'unique_traders' in df.columns:
            condition = df['unique_traders'].fillna(
                0) >= TCNF.MIN_UNIQUE_TRADERS
            df = df[condition].copy()
            df = ensure_manual(df)
            avg_attractiveness = df['attractiveness_score'].mean() if len(
                df) > 0 else 0
            avg_gm_reward = df['gm_reward_per_100'].mean() if len(
                df) > 0 else 0
            Logan.info(
                f"After unique traders filter (≥{TCNF.MIN_UNIQUE_TRADERS}): {len(df)}/{initial_count} markets "
                f"(avg attractiveness: {avg_attractiveness:.2f}, avg GM reward: {avg_gm_reward:.2f})",
                namespace="poly_data.market_selection"
            )

        if 'order_arrival_rate_sensitivity' in df.columns:
            lower_condition = df['order_arrival_rate_sensitivity'].fillna(
                0) >= TCNF.MIN_ARRIVAL_RATE_SENSITIVITY
            upper_condition = df['order_arrival_rate_sensitivity'].fillna(
                0) <= TCNF.MAX_ARRIVAL_RATE_SENSITIVITY
            df = df[lower_condition & upper_condition].copy()
            df = ensure_manual(df)
            avg_attractiveness = df['attractiveness_score'].mean() if len(
                df) > 0 else 0
            avg_gm_reward = df['gm_reward_per_100'].mean() if len(
                df) > 0 else 0
            Logan.info(
                f"After arrival rate sensitivity filter ({TCNF.MIN_ARRIVAL_RATE_SENSITIVITY} ≤ k ≤ {TCNF.MAX_ARRIVAL_RATE_SENSITIVITY}): {len(df)}/{initial_count} markets "
                f"(avg attractiveness: {avg_attractiveness:.2f}, avg GM reward: {avg_gm_reward:.2f})",
                namespace="poly_data.market_selection"
            )

        avg_attractiveness = df['attractiveness_score'].mean() if len(
            df) > 0 else 0
        avg_gm_reward = df['gm_reward_per_100'].mean() if len(df) > 0 else 0
        Logan.info(
            f"Activity filters completed: {len(df)}/{initial_count} markets remaining "
            f"(avg attractiveness: {avg_attractiveness:.2f}, avg GM reward: {avg_gm_reward:.2f})",
            namespace="poly_data.market_selection"
        )
    else:
        Logan.warn(
            "Activity metrics not found in data, skipping activity-based filtering",
            namespace="poly_data.market_selection"
        )

    df = ensure_manual(df)

    # Prepare final selection dataframes
    df['condition_id'] = df['condition_id'].astype(str)
    df_unique = df.drop_duplicates(subset='condition_id', keep='first').copy()

    manual_selection = df_unique.iloc[0:0].copy()
    manual_order = manual_ids
    if manual_order:
        frames = []
        for cid in manual_order:
            match = df_unique[df_unique['condition_id'] == cid]
            if match.empty:
                fallback = manual_rows[manual_rows['condition_id'].astype(
                    str) == cid].copy()
                fallback = fallback.reindex(
                    columns=df_unique.columns, fill_value=pd.NA)
                match = fallback
            if not match.empty:
                frames.append(match.iloc[0:1])
        if frames:
            manual_selection = pd.concat(frames, ignore_index=True, sort=False)
        manual_selection['selection_source'] = 'manual'

    auto_pool = df_unique[~df_unique['condition_id'].isin(manual_order)].copy()
    sort_columns: list[str] = []
    if 'gm_reward_per_100' in auto_pool.columns:
        sort_columns.append('gm_reward_per_100')
    sort_columns.append('attractiveness_score')
    ascending_flags = [False] * len(sort_columns)
    auto_pool_sorted = auto_pool.sort_values(
        by=sort_columns,
        ascending=ascending_flags,
        na_position='last'
    )

    auto_slots = max(TCNF.MARKET_COUNT - len(manual_order), 0)
    auto_selected = auto_pool_sorted.head(auto_slots).copy()
    if not auto_selected.empty:
        auto_selected['selection_source'] = 'auto'

    result = pd.concat([manual_selection, auto_selected],
                       ignore_index=True, sort=False)
    result = result.drop_duplicates(subset='condition_id', keep='first')

    if manual_col and manual_col in result.columns:
        result[manual_col] = result[manual_col].fillna('')
    if 'selection_source' in result.columns:
        result['selection_source'] = result['selection_source'].fillna('auto')
    else:
        result['selection_source'] = pd.Series(
            ['auto'] * len(result), dtype=str)

    if manual_ids:
        present_ids = set(result['condition_id'].astype(str))
        missing_manual = [cid for cid in manual_ids if cid not in present_ids]
        if missing_manual:
            Logan.warn(
                f"Manual selections filtered out: {missing_manual}",
                namespace="poly_data.market_selection"
            )

    final_avg_attractiveness = result['attractiveness_score'].mean() if len(
        result) > 0 else 0
    final_avg_gm_reward = result['gm_reward_per_100'].mean() if len(
        result) > 0 else 0
    Logan.info(
        f"Final selection: {len(result)}/{initial_count} markets (manual: {len(manual_selection)}, auto: {len(auto_selected)}) "
        f"(avg attractiveness: {final_avg_attractiveness:.2f}, avg GM reward: {final_avg_gm_reward:.2f})",
        namespace="poly_data.market_selection"
    )

    try:
        manual_ids = result[result['selection_source']
                            == 'manual']['condition_id'].astype(str).tolist()
        auto_ids = result[result['selection_source']
                          != 'manual']['condition_id'].astype(str).tolist()
        global_state.set_selection_groups(manual_ids, auto_ids)
    except Exception as e:
        Logan.error(
            "Failed to update selection groups",
            namespace="poly_data.market_selection",
            exception=e
        )

    # Write selected markets back to the Selected Markets sheet
    try:
        write_selected_markets_to_sheet(result)
    except Exception as e:
        Logan.error(
            "Failed to write selected markets to sheet",
            namespace="poly_data.market_selection",
            exception=e
        )

    return result


def write_selected_markets_to_sheet(selected_df: pd.DataFrame):
    """Write the selected markets to the 'Selected Markets' sheet for visibility"""
    try:
        spreadsheet = get_spreadsheet()
        worksheet = spreadsheet.worksheet("Selected Markets")

        # Clear existing data and write new selection
        worksheet.clear()

        # Select key columns for the sheet including activity metrics if available
        base_output_cols = [
            'question',
            'answer1',
            'answer2',
            'attractiveness_score',
            'volatility_sum',
            'gm_reward_per_100',
            'rewards_daily_rate',
            'min_size',
            'max_spread',
            'best_bid',
            'best_ask',
            'reward_bid_floor',
            'reward_ask_ceiling',
            'token1',
            'token2',
            'condition_id'
        ]

        manual_col = next(
            (
                col
                for col in selected_df.columns
                if col and col.strip().lower().replace(' ', '_') == 'manual_select'
            ),
            None
        )

        if manual_col:
            base_output_cols = [manual_col] + base_output_cols

        if 'selection_source' in selected_df.columns:
            base_output_cols.append('selection_source')

        # Add activity columns if they exist
        activity_output_cols = [
            'total_volume', 'volume_usd', 'avg_trades_per_day', 'unique_traders']
        available_activity_cols = [
            col for col in activity_output_cols if col in selected_df.columns]

        output_cols = base_output_cols + available_activity_cols
        header_cols = [
            col for col in output_cols if col in selected_df.columns]
        output_df = selected_df[header_cols].copy()

        if output_df.empty:
            worksheet.clear()
            if header_cols:
                worksheet.update('A1', [header_cols])
            Logan.info(
                "Selected markets list is empty; wrote headers only",
                namespace="poly_data.market_selection"
            )
            return

        set_with_dataframe(worksheet, output_df, include_index=False,
                           include_column_header=True, resize=True)

        Logan.info(
            f"Successfully wrote {len(output_df)} selected markets to Selected Markets sheet",
            namespace="poly_data.market_selection"
        )
    except Exception as e:
        Logan.error(
            f"Error writing to Selected Markets sheet",
            namespace="poly_data.market_selection",
            exception=e
        )


def calculate_position_sizes():
    selected_df = global_state.selected_markets_df
    if selected_df is None or selected_df.empty:
        global_state.market_trade_sizes = {}
        return

    total_liquidity = float(global_state.available_liquidity or 0.0)
    budget = total_liquidity * TCNF.BUDGET_MULT
    total_sharpe = selected_df['attractiveness_score'].sum()
    market_count = len(selected_df)

    reward_cap_multiplier = max(TCNF.REWARD_TRADE_SIZE_MULTIPLIER, 1.0)
    reward_max_pos_multiplier = max(TCNF.REWARD_MAX_POSITION_MULTIPLIER, 1.0)

    proposed_sizes: dict[str, PositionSizeResult] = {}
    floors: dict[str, float] = {}
    ceilings: dict[str, float] = {}

    for _, row in selected_df.iterrows():
        condition_id = str(row['condition_id'])
        sharpe = max(float(row.get('attractiveness_score', 0) or 0.0), 0.0)
        weight = sharpe / \
            total_sharpe if total_sharpe > 0 else (1.0 / market_count)

        min_size = float(row.get('min_size', 0) or 0.0)
        floor_size = min_size if min_size > 0 else 0.0
        if min_size > 0:
            ceiling_size = min(TCNF.INVESTMENT_CEILING,
                               min_size * reward_cap_multiplier)
        else:
            ceiling_size = TCNF.INVESTMENT_CEILING

        raw_size = budget * weight if budget > 0 else 0.0
        if min_size > 0:
            size = max(floor_size, min(raw_size, ceiling_size))
        else:
            size = min(raw_size, ceiling_size)

        proposed_sizes[condition_id] = PositionSizeResult(
            trade_size=size,
            max_size=min(size * reward_max_pos_multiplier,
                         TCNF.INVESTMENT_CEILING)
        )
        floors[condition_id] = floor_size
        ceilings[condition_id] = ceiling_size if ceiling_size >= floor_size else floor_size

    try:
        redistributed = redistribute_for_bounds(
            proposed_sizes, floors, ceilings)
        adjusted: dict[str, PositionSizeResult] = {}
        for key, value in redistributed.items():
            trade_size = value.trade_size
            max_size = min(trade_size * reward_max_pos_multiplier,
                           TCNF.INVESTMENT_CEILING)
            adjusted[key] = PositionSizeResult(
                trade_size=trade_size, max_size=max_size)
        global_state.market_trade_sizes = adjusted
    except Exception as e:
        Logan.warn(
            f"Couldn't redistribute for bounds: {e}",
            namespace="poly_data.market_selection"
        )
        global_state.market_trade_sizes = fallback_position_sizes_for_low_liquidity(
            budget)


def fallback_position_sizes_for_low_liquidity(budget: float):
    """
    Select up to TCNF.MARKET_COUNT markets, sorted by attractiveness_score descending,
    and allocate their min_size as position size, until the total exceeds the budget.
    Returns a dict[condition_id, PositionSizeResult].
    """
    selected_df = global_state.selected_markets_df.copy()
    if selected_df is None or selected_df.empty:
        return {}

    reward_cap_multiplier = max(TCNF.REWARD_TRADE_SIZE_MULTIPLIER, 1.0)
    reward_max_pos_multiplier = max(TCNF.REWARD_MAX_POSITION_MULTIPLIER, 1.0)

    # Sort by attractiveness_score descending
    selected_df = selected_df.sort_values(
        "attractiveness_score", ascending=False)
    total = 0.0
    result = {}

    for i, row in selected_df.iterrows():
        condition_id = str(row["condition_id"])
        min_size = float(row.get("min_size", 0.0))
        if min_size <= 0:
            continue

        target_size = min(min_size * reward_cap_multiplier,
                          TCNF.INVESTMENT_CEILING)
        remaining_budget = budget - total

        if remaining_budget < min_size:
            break

        size = min(target_size, remaining_budget)
        result[condition_id] = PositionSizeResult(
            trade_size=size,
            max_size=min(size * reward_max_pos_multiplier,
                         TCNF.INVESTMENT_CEILING)
        )
        total += size

    return result


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
    w = np.asarray([float(position_sizes[k].trade_size)
                   for k in keys], dtype=float)

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

    return {k: PositionSizeResult(trade_size=float(x[i]), max_size=float(x[i]) * TCNF.MAX_POSITION_MULT) for i, k in enumerate(keys)}


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
    position_size_info = global_state.market_trade_sizes.get(condition_id)
    if position_size_info:
        # Override trade_size and max_size with calculated values
        market_row['trade_size'] = position_size_info.trade_size
        market_row['max_size'] = position_size_info.max_size

    return market_row
