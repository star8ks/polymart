import threading
import time
from dataclasses import dataclass
import pandas as pd
from typing import Dict, Optional, Tuple, Iterable
from logan import Logan
# ============ Market Data ============

# List of all tokens being tracked
all_tokens = []
all_tokens_lock = threading.Lock()
all_tokens_version = 0

# Mapping from condition_id -> (token_yes, token_no)
condition_token_map: Dict[str, Tuple[str, str]] = {}

# Manual / auto selection tracking
selection_lock = threading.Lock()
manual_condition_ids: set[str] = set()
auto_condition_ids: set[str] = set()
manual_ready_last_state: Optional[bool] = None
manual_ready_last_details: Tuple[str, ...] = tuple()
manual_unserviceable: Dict[str, str] = {}
manual_order_state: Dict[str, 'ManualOrderState'] = {}
manual_pending_exit: Dict[str, Dict[str, float]] = {}


@dataclass
class ManualOrderState:
    buy_price: float = 0.0
    buy_size: float = 0.0
    sell_price: float = 0.0
    sell_size: float = 0.0
    last_update: float = 0.0
    last_submission: float = 0.0


# Mapping between tokens in the same market (YES->NO, NO->YES)
REVERSE_TOKENS = {}

# Order book data for all markets
order_book_data = {}

# Market configuration data from Google Sheets
df = None

# Filtered markets after applying custom selection logic
selected_markets_df = None

markets_with_positions = None

# Position sizing information for each market
# Format: {condition_id: PositionSizeResult}
market_trade_sizes = {}

# Available cash liquidity for trading (USDC balance)
available_liquidity: Optional[float] = None

# ============ Client & Parameters ============

# Polymarket client instance
client = None

# Trading parameters from Google Sheets
params = {}

# Lock for thread-safe trading operations
lock = threading.Lock()

# ============ Trading State ============

# Tracks trades that have been matched but not yet mined
# Format: {"token_side": {trade_id1, trade_id2, ...}}
performing = {}

# Timestamps for when trades were added to performing
# Used to clear stale trades
performing_timestamps = {}

# Timestamps for when positions were last updated
last_trade_update = {}

# Current open orders for each token
# Format: {token_id: {'buy': {price, size}, 'sell': {price, size}}}
orders = {}

# Current positions for each token
# Format: {token_id: {'size': float, 'avgPrice': float}}
positions = {}


def _increment_token_version():
    global all_tokens_version
    all_tokens_version += 1


def register_market_tokens(token_yes: str, token_no: str, condition_id: Optional[str] = None) -> bool:
    """Register tokens for a market, tracking new entries and reverse mappings."""

    token_yes = str(token_yes)
    token_no = str(token_no)
    condition_id = str(condition_id) if condition_id is not None else None

    tokens_added = False

    with all_tokens_lock:
        if token_yes not in all_tokens:
            all_tokens.append(token_yes)
            tokens_added = True
            Logan.info(
                f"Tracking new token {token_yes} (condition_id={condition_id})",
                namespace="poly_data.global_state"
            )

        if token_yes not in REVERSE_TOKENS:
            REVERSE_TOKENS[token_yes] = token_no

        if token_no not in REVERSE_TOKENS:
            REVERSE_TOKENS[token_no] = token_yes

        order_keys = [
            f"{token_yes}_buy",
            f"{token_yes}_sell",
            f"{token_no}_buy",
            f"{token_no}_sell"
        ]

        for key in order_keys:
            if key not in performing:
                performing[key] = set()
            if key not in performing_timestamps:
                performing_timestamps[key] = {}

        if condition_id:
            condition_token_map[condition_id] = (token_yes, token_no)

        if tokens_added:
            _increment_token_version()

    return tokens_added


def set_selection_groups(manual_ids: Iterable[str], auto_ids: Iterable[str]):
    global manual_condition_ids, auto_condition_ids, manual_unserviceable, manual_order_state, manual_pending_exit
    manual_normalized = {str(cid) for cid in manual_ids if cid is not None}
    auto_normalized = {str(cid) for cid in auto_ids if cid is not None}

    with selection_lock:
        manual_condition_ids = manual_normalized
        auto_condition_ids = auto_normalized
        manual_unserviceable = {
            cid: reason
            for cid, reason in manual_unserviceable.items()
            if cid in manual_condition_ids
        }
        manual_order_state = {
            cid: state
            for cid, state in manual_order_state.items()
            if cid in manual_condition_ids
        }
        manual_pending_exit = {
            cid: data
            for cid, data in manual_pending_exit.items()
            if cid in manual_condition_ids
        }


def set_manual_unserviceable(condition_id: str, reason: str):
    condition_id = str(condition_id)
    if not condition_id:
        return

    with selection_lock:
        manual_unserviceable[condition_id] = reason

    clear_manual_order_state(condition_id)

    Logan.warn(
        f"Manual market {condition_id} marked unserviceable: {reason}",
        namespace="poly_data.global_state"
    )


def clear_manual_unserviceable(condition_id: str):
    condition_id = str(condition_id)
    if not condition_id:
        return

    removed = False
    with selection_lock:
        if condition_id in manual_unserviceable:
            manual_unserviceable.pop(condition_id, None)
            removed = True

    if removed:
        Logan.info(
            f"Manual market {condition_id} is now serviceable",
            namespace="poly_data.global_state"
        )


def manual_markets_ready(log_details: bool = False) -> bool:
    with selection_lock:
        manual_ids_snapshot = list(manual_condition_ids)
        unserviceable_snapshot = dict(manual_unserviceable)
        order_state_snapshot = {cid: manual_order_state.get(cid)
                                for cid in manual_ids_snapshot}

    if not manual_ids_snapshot:
        return True

    with all_tokens_lock:
        token_map_snapshot = {
            cid: condition_token_map.get(cid)
            for cid in manual_ids_snapshot
        }

    missing_details: list[str] = []
    skipped_details: list[str] = []
    for cid in manual_ids_snapshot:
        if cid in unserviceable_snapshot:
            skipped_details.append(f"{cid}: {unserviceable_snapshot[cid]}")
            continue

        tokens = token_map_snapshot.get(cid)
        if not tokens:
            missing_details.append(f"{cid}: tokens not registered")
            continue

        token_yes = tokens[0]
        order_entry = orders.get(
            token_yes, {}) if isinstance(orders, dict) else {}
        buy_size = 0.0
        sell_size = 0.0

        if isinstance(order_entry, dict):
            buy_order = order_entry.get('buy', {})
            sell_order = order_entry.get('sell', {})
            buy_size = float(buy_order.get('size', 0) or 0)
            sell_size = float(sell_order.get('size', 0) or 0)

        state = order_state_snapshot.get(cid)
        if state and isinstance(state, ManualOrderState):
            now = time.time()
            if now - state.last_submission <= 300:
                buy_size = max(buy_size, state.buy_size)
                sell_size = max(sell_size, state.sell_size)

        if buy_size <= 0 and sell_size <= 0:
            missing_details.append(
                f"{cid}: no resting orders (buy={buy_size:.2f}, sell={sell_size:.2f})"
            )

    ready = len(missing_details) == 0

    global manual_ready_last_state, manual_ready_last_details
    details_tuple = tuple(missing_details + skipped_details)
    if ready != manual_ready_last_state or (not ready and details_tuple != manual_ready_last_details) or log_details:
        if ready:
            Logan.info(
                "All manual markets have active orders",
                namespace="poly_data.global_state"
            )
        else:
            message_parts = []
            if missing_details:
                message_parts.append("pending orders: " +
                                     "; ".join(missing_details))
            if skipped_details:
                message_parts.append("skipped: " + "; ".join(skipped_details))
            if not message_parts:
                message_parts.append("no manual details available")
            Logan.info(
                "Manual market status - " + " | ".join(message_parts),
                namespace="poly_data.global_state"
            )
        manual_ready_last_state = ready
        manual_ready_last_details = details_tuple

    return ready


def get_manual_order_state(condition_id: str) -> Optional[ManualOrderState]:
    return manual_order_state.get(str(condition_id))


def record_manual_order_submission(condition_id: str, side: str, price: float, size: float):
    condition_id = str(condition_id)
    side = side.lower()
    if condition_id not in manual_condition_ids:
        return

    with selection_lock:
        state = manual_order_state.get(condition_id)
        if state is None:
            state = ManualOrderState()

        now = time.time()
        if side == 'buy':
            state.buy_price = float(price)
            state.buy_size = float(size)
        elif side == 'sell':
            state.sell_price = float(price)
            state.sell_size = float(size)
        state.last_update = now
        state.last_submission = now
        manual_order_state[condition_id] = state


def record_manual_order_cancel(condition_id: str):
    condition_id = str(condition_id)
    with selection_lock:
        state = manual_order_state.get(condition_id)
        if state is None:
            return
        now = time.time()
        state.buy_size = 0.0
        state.sell_size = 0.0
        state.buy_price = 0.0
        state.sell_price = 0.0
        state.last_update = now
        manual_order_state[condition_id] = state


def clear_manual_order_state(condition_id: str):
    condition_id = str(condition_id)
    with selection_lock:
        manual_order_state.pop(condition_id, None)


def manual_recent_submission(condition_id: str, window_sec: float = 3.0) -> bool:
    condition_id = str(condition_id)
    state = manual_order_state.get(condition_id)
    if state is None:
        return False
    return time.time() - state.last_submission <= window_sec


def set_pending_exit(condition_id: str, side: str, price: float, size: float):
    condition_id = str(condition_id)
    if condition_id not in manual_condition_ids:
        return
    manual_pending_exit[condition_id] = {
        'side': side.lower(),
        'price': float(price),
        'size': float(size),
        'timestamp': time.time()
    }


def peek_pending_exit(condition_id: str) -> Optional[Dict[str, float]]:
    return manual_pending_exit.get(str(condition_id))


def clear_pending_exit(condition_id: str):
    manual_pending_exit.pop(str(condition_id), None)


def get_token_snapshot() -> Tuple[list[str], int]:
    """Return a copy of tracked tokens and the current version."""
    with all_tokens_lock:
        return list(all_tokens), all_tokens_version


def get_token_version() -> int:
    """Return the current token version."""
    with all_tokens_lock:
        return all_tokens_version


def get_active_markets():
    """Return the union of selected markets and markets with positions.

    When we have open positions, ensure those markets are included even if they
    are not currently selected by the filter. Duplicates are removed by
    `condition_id` while keeping the first occurrence.
    """
    combined_markets = selected_markets_df

    # Treat None as empty for robustness
    has_markets_with_positions = (
        markets_with_positions is not None and len(markets_with_positions) > 0
    )

    if has_markets_with_positions:
        if combined_markets is not None:
            combined_markets = pd.concat([combined_markets, markets_with_positions]).drop_duplicates(
                subset=['condition_id'], keep='first'
            )
        else:
            combined_markets = markets_with_positions

    return combined_markets
