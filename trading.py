import gc
import asyncio
import math

from logan import Logan

import poly_data.global_state as global_state
from configuration import TCNF
from poly_data.orders_in_flight import get_orders_in_flight, set_order_in_flight
from poly_data.trading_utils import get_best_bid_ask_deets, round_down, round_up
from poly_data.data_utils import get_position, get_order
from poly_data.market_selection import get_enhanced_market_row, get_latest_market_snapshot


def _safe_float(value, default=0.0):
    try:
        result = float(value)
        if math.isnan(result):
            return default
        return result
    except (TypeError, ValueError):
        return default


def _determine_trade_size(row):
    trade_size = _safe_float(row.get('trade_size'))
    min_size = _safe_float(row.get('min_size'))

    if trade_size <= 0:
        trade_size = min_size

    if trade_size <= 0:
        return 0.0

    if min_size > 0 and trade_size < min_size:
        trade_size = min_size

    return trade_size


def _ensure_reward_defaults(row, best_bid, best_ask, tick):
    reward_floor = _safe_float(row.get('reward_bid_floor'))
    reward_ceiling = _safe_float(row.get('reward_ask_ceiling'))
    reward_half = _safe_float(row.get('reward_half_spread'))
    max_spread_pct = _safe_float(row.get('max_spread'))

    if max_spread_pct <= 0:
        return row

    max_spread_abs = max_spread_pct / 100.0
    midpoint = (best_bid + best_ask) / \
        2 if best_bid is not None and best_ask is not None else None

    adjustments = []

    if reward_half <= 0 and midpoint is not None:
        reward_half = max(max_spread_abs / 2, tick)
        row['reward_half_spread'] = reward_half
        adjustments.append('reward_half_spread')

    if reward_floor <= 0 and midpoint is not None:
        floor_candidate = max(best_bid - max_spread_abs, tick)
        if reward_half > 0:
            floor_candidate = min(floor_candidate, midpoint - reward_half)
        row['reward_bid_floor'] = max(floor_candidate, tick)
        adjustments.append('reward_bid_floor')

    if reward_ceiling <= 0 and midpoint is not None:
        ceiling_candidate = min(best_ask + max_spread_abs, 1 - tick)
        if reward_half > 0:
            ceiling_candidate = max(ceiling_candidate, midpoint + reward_half)
        row['reward_ask_ceiling'] = min(ceiling_candidate, 1 - tick)
        adjustments.append('reward_ask_ceiling')

    if adjustments:
        Logan.info(
            f"Applied default reward band values ({', '.join(adjustments)}) for market {row.get('condition_id')}",
            namespace="trading"
        )

    return row


def _compute_yes_quotes(midpoint, tick, round_length, row):
    reward_floor = _safe_float(row.get('reward_bid_floor'))
    reward_ceiling = _safe_float(row.get('reward_ask_ceiling'))
    reward_half = _safe_float(row.get('reward_half_spread'))
    max_spread_pct = _safe_float(row.get('max_spread'))
    max_spread_abs = max_spread_pct / 100.0 if max_spread_pct > 0 else 0.0

    floor_candidates = []
    ceiling_candidates = []

    if reward_floor > 0:
        floor_candidates.append(reward_floor)
    if reward_half > 0:
        floor_candidates.append(midpoint - reward_half)
        ceiling_candidates.append(midpoint + reward_half)
    if max_spread_abs > 0:
        floor_candidates.append(midpoint - max_spread_abs)
        ceiling_candidates.append(midpoint + max_spread_abs)
    if reward_ceiling > 0:
        ceiling_candidates.append(reward_ceiling)

    floor_candidates = [f for f in floor_candidates if not math.isnan(f)]
    ceiling_candidates = [c for c in ceiling_candidates if not math.isnan(c)]

    if not floor_candidates or not ceiling_candidates:
        Logan.warn(
            f"Reward band incomplete for market {row.get('condition_id')} (floor candidates={floor_candidates}, ceiling candidates={ceiling_candidates})",
            namespace="trading"
        )
        return None, None

    band_floor = max([tick] + floor_candidates)
    band_ceiling = min([1 - tick] + ceiling_candidates)

    band_floor = min(band_floor, midpoint)
    band_ceiling = max(band_ceiling, midpoint)

    if band_ceiling - band_floor <= 2 * tick:
        return None, None

    half_width = min(midpoint - band_floor, band_ceiling - midpoint)
    if half_width <= tick:
        return None, None

    edge_offset = tick * \
        max(int(getattr(TCNF, 'REWARD_EDGE_OFFSET_TICKS', 1)), 0)
    passive_offset = tick * \
        max(int(getattr(TCNF, 'REWARD_PASSIVE_OFFSET_TICKS', 0)), 0)

    edge_offset = min(edge_offset, max(half_width - tick, 0.0))
    passive_offset = min(passive_offset, max(
        half_width - edge_offset - tick, 0.0))

    base_bid = midpoint - half_width + edge_offset
    base_ask = midpoint + half_width - edge_offset

    base_bid = max(band_floor, min(base_bid, midpoint - tick))
    base_ask = min(band_ceiling, max(base_ask, midpoint + tick))

    bid_price = max(band_floor, min(
        base_bid - passive_offset, midpoint - tick))
    ask_price = min(band_ceiling, max(
        base_ask + passive_offset, midpoint + tick))

    bid_price = round_down(bid_price, round_length)
    ask_price = round_up(ask_price, round_length)

    bid_price = max(band_floor, min(bid_price, band_ceiling - tick))
    ask_price = min(band_ceiling, max(ask_price, band_floor + tick))

    if bid_price >= ask_price:
        midpoint_clamped = min(
            max(midpoint, band_floor + tick), band_ceiling - tick)
        bid_price = round_down(
            max(midpoint_clamped - tick, band_floor), round_length)
        ask_price = round_up(
            min(midpoint_clamped + tick, band_ceiling), round_length)

        if bid_price >= ask_price:
            return None, None

    bid_price = max(tick, min(bid_price, 1 - tick))
    ask_price = max(tick, min(ask_price, 1 - tick))

    return bid_price, ask_price


def _orders_need_update(existing_orders, desired_orders):
    for side in ('buy', 'sell'):
        existing = existing_orders.get(side, {'price': 0.0, 'size': 0.0})
        desired = desired_orders.get(side)

        if desired is None:
            if existing.get('size', 0.0) > 0:
                return True
            continue

        target_price, target_size = desired
        current_size = existing.get('size', 0.0)
        current_price = existing.get('price', 0.0)

        if current_size <= 0:
            return True

        price_diff = abs(current_price - target_price)
        threshold = TCNF.BUY_PRICE_DIFF_THRESHOLD if side == 'buy' else TCNF.SELL_PRICE_DIFF_THRESHOLD
        if price_diff > threshold:
            return True

        if target_size > 0:
            size_diff = abs(current_size - target_size)
            if size_diff > target_size * TCNF.SIZE_DIFF_PERCENTAGE:
                return True

    return False


def _manual_orders_match(state, desired_orders):
    if state is None or not desired_orders:
        return False

    existing = {
        'buy': {
            'price': getattr(state, 'buy_price', 0.0),
            'size': getattr(state, 'buy_size', 0.0)
        },
        'sell': {
            'price': getattr(state, 'sell_price', 0.0),
            'size': getattr(state, 'sell_size', 0.0)
        }
    }

    return not _orders_need_update(existing, desired_orders)


def _handle_create_order_response(resp, order, side):
    if isinstance(resp, dict) and resp.get('success') and resp.get('orderID'):
        set_order_in_flight(
            order['market'], resp['orderID'], side.lower(), order['price'], order['size'])
        Logan.info(
            f"Placed {side} order for {order['token']} at {order['price']:.3f} size {order['size']:.2f}",
            namespace="trading"
        )
    else:
        Logan.error(
            f"Failed to create {side} order for token {order['token']}: {resp}",
            namespace="trading"
        )


def _submit_order(order, side):
    try:
        resp = global_state.client.create_order(
            order['token'],
            side,
            order['price'],
            order['size'],
            order['neg_risk']
        )
    except Exception as exc:
        Logan.error(
            f"Error sending {side} order for token {order['token']}",
            namespace="trading",
            exception=exc
        )
        return

    _handle_create_order_response(resp, order, side)


async def _refresh_market_snapshot(condition_id: str):
    """Fetch the latest sheet snapshot for a market without blocking the event loop."""

    try:
        loop = asyncio.get_running_loop()
    except RuntimeError:
        return get_latest_market_snapshot(condition_id)

    return await loop.run_in_executor(None, get_latest_market_snapshot, condition_id)


market_locks = {}


async def perform_trade(market):
    if get_orders_in_flight(market):
        return

    if market not in market_locks:
        market_locks[market] = asyncio.Lock()

    async with market_locks[market]:
        try:
            row = get_enhanced_market_row(market)
            if row is None:
                Logan.warn(
                    f"Market row not found for {market}",
                    namespace="trading"
                )
                return

            row = row.copy()
            selection_source = str(
                row.get('selection_source', 'auto') or 'auto').lower()
            condition_id = str(market)

            if selection_source == 'manual':
                snapshot = await _refresh_market_snapshot(condition_id)
                if snapshot:
                    trade_size_override = row.get('trade_size')
                    max_size_override = row.get('max_size')
                    selection_source_override = row.get('selection_source')

                    tracked_fields = (
                        'max_spread',
                        'reward_bid_floor',
                        'reward_ask_ceiling',
                        'reward_half_spread'
                    )
                    delta_messages = []
                    for field in tracked_fields:
                        if field in snapshot:
                            previous = _safe_float(
                                row.get(field), default=None)
                            updated = _safe_float(
                                snapshot.get(field), default=None)
                            if previous is not None and updated is not None and abs(previous - updated) > 1e-6:
                                delta_messages.append(
                                    f"{field} {previous:.5f}->{updated:.5f}")

                    for key, value in snapshot.items():
                        if key in ('trade_size', 'max_size', 'selection_source'):
                            continue
                        row[key] = value

                    if selection_source_override is not None:
                        row['selection_source'] = selection_source_override
                    if trade_size_override is not None:
                        row['trade_size'] = trade_size_override
                    if max_size_override is not None:
                        row['max_size'] = max_size_override

                    if delta_messages:
                        Logan.info(
                            f"Refreshed sheet data for market {condition_id}: {'; '.join(delta_messages)}",
                            namespace="trading"
                        )

            token_yes = str(row['token1'])

            if selection_source != 'manual':
                existing = get_order(token_yes)
                has_orders = (
                    existing.get('buy', {}).get('size', 0) > 0 or
                    existing.get('sell', {}).get('size', 0) > 0
                )
                if has_orders:
                    try:
                        global_state.client.cancel_all_asset(token_yes)
                    except Exception as exc:
                        Logan.error(
                            f"Error cancelling auto market orders for token {token_yes}",
                            namespace="trading",
                            exception=exc
                        )
                return

            if token_yes not in global_state.order_book_data:
                Logan.warn(
                    f"No order book data yet for token {token_yes}",
                    namespace="trading"
                )
                return

            tick = _safe_float(row.get('tick_size'), 0.01)
            tick_decimals = 0
            tick_str = str(tick)
            if '.' in tick_str:
                tick_decimals = len(tick_str.split('.')[1])

            try:
                book = get_best_bid_ask_deets(token_yes, max(
                    int(_safe_float(row.get('min_size'), 1)), 1))
            except KeyError:
                Logan.warn(
                    f"Order book snapshot missing for token {token_yes}",
                    namespace="trading"
                )
                return

            best_bid = book.get('best_bid') if book.get(
                'best_bid') is not None else book.get('top_bid')
            best_ask = book.get('best_ask') if book.get(
                'best_ask') is not None else book.get('top_ask')

            if best_bid is None or best_ask is None:
                best_bid = _safe_float(row.get('best_bid'))
                best_ask = _safe_float(row.get('best_ask'))

            if best_bid is None or best_ask is None or best_bid <= 0 or best_ask >= 1:
                Logan.warn(
                    f"Invalid best bid/ask for {market}: bid={best_bid}, ask={best_ask}",
                    namespace="trading"
                )
                return

            row = _ensure_reward_defaults(row, best_bid, best_ask, tick)

            pending_exit = global_state.peek_pending_exit(condition_id)
            if pending_exit is None and global_state.manual_recent_submission(condition_id, window_sec=2.0):
                return

            trade_size = _determine_trade_size(row)
            min_size = _safe_float(row.get('min_size'))
            liquidity = _safe_float(global_state.available_liquidity)
            position_yes = _safe_float(get_position(token_yes)['size'])
            max_position = max(trade_size, _safe_float(
                row.get('max_size'), trade_size))
            remaining_capacity = max(max_position - position_yes, 0.0)

            desired_orders = {}
            manual_state = global_state.get_manual_order_state(condition_id)

            if pending_exit:
                exit_side = str(pending_exit.get('side', '')).lower()
                exit_price = float(pending_exit.get('price', 0.0))
                exit_size = max(float(pending_exit.get('size', 0.0)), 0.0)

                if exit_side == 'buy':
                    if liquidity <= 0:
                        global_state.set_manual_unserviceable(
                            condition_id, f"exit buy blocked (liquidity={liquidity:.2f})")
                        return
                    if min_size > 0 and exit_size < min_size:
                        exit_size = min_size
                    desired_orders['buy'] = (exit_price, exit_size)
                elif exit_side == 'sell':
                    inventory = max(position_yes, 0.0)
                    if inventory <= 0:
                        global_state.set_manual_unserviceable(
                            condition_id, "exit sell blocked (no inventory)")
                        return
                    desired_orders['sell'] = (
                        exit_price, min(exit_size, inventory))
                else:
                    global_state.clear_pending_exit(condition_id)
            else:
                midpoint = (best_bid + best_ask) / 2
                bid_yes, ask_yes = _compute_yes_quotes(
                    midpoint, tick, tick_decimals, row)
                if bid_yes is None or ask_yes is None:
                    Logan.warn(
                        f"Failed to compute quotes for {market} (midpoint={midpoint:.4f}, tick={tick})",
                        namespace="trading"
                    )
                    return

                if trade_size <= 0:
                    return

                if min_size > 0 and trade_size < min_size:
                    trade_size = min_size

                if liquidity > 0 and remaining_capacity > 0:
                    max_affordable = liquidity / max(bid_yes, tick)
                    buy_size = min(trade_size, max_affordable,
                                   remaining_capacity)
                    if min_size > 0 and buy_size < min_size:
                        buy_size = 0.0
                    if buy_size > 0:
                        desired_orders['buy'] = (bid_yes, buy_size)

                sell_inventory = max(position_yes, 0.0)
                sell_size = min(trade_size, sell_inventory)
                if min_size > 0 and sell_size < min_size:
                    sell_size = 0.0
                if sell_size > 0:
                    desired_orders['sell'] = (ask_yes, sell_size)

            if not desired_orders:
                existing_orders_yes = get_order(token_yes)
                has_existing = (
                    existing_orders_yes.get('buy', {}).get('size', 0) > 0 or
                    existing_orders_yes.get('sell', {}).get('size', 0) > 0
                )
                if has_existing:
                    try:
                        global_state.client.cancel_all_asset(token_yes)
                    except Exception as exc:
                        Logan.error(
                            f"Error cancelling existing orders for token {token_yes}",
                            namespace="trading",
                            exception=exc
                        )

                skip_reason = "no liquidity"
                if liquidity <= 0:
                    skip_reason = f"insufficient liquidity ({liquidity:.2f})"
                elif min_size > 0 and liquidity < min_size:
                    skip_reason = f"min_size {min_size:.2f} exceeds available liquidity {liquidity:.2f}"
                elif trade_size <= 0:
                    skip_reason = "trade size is zero"
                elif remaining_capacity <= 0:
                    skip_reason = f"no capacity (position={position_yes:.2f}, max={max_position:.2f})"
                elif position_yes <= 0:
                    skip_reason = "no inventory to sell"

                global_state.set_manual_unserviceable(
                    condition_id, skip_reason)
                global_state.record_manual_order_cancel(condition_id)
                Logan.warn(
                    f"Manual market {condition_id} skipped: {skip_reason}",
                    namespace="trading"
                )
                return

            if _manual_orders_match(manual_state, desired_orders):
                return

            global_state.clear_manual_unserviceable(condition_id)

            existing_orders_yes = get_order(token_yes)
            if _orders_need_update(existing_orders_yes, desired_orders):
                try:
                    global_state.client.cancel_all_asset(token_yes)
                except Exception as exc:
                    Logan.error(
                        f"Error cancelling existing orders for token {token_yes}",
                        namespace="trading",
                        exception=exc
                    )
                else:
                    Logan.info(
                        f"Resetting orders for {token_yes} (buy={'buy' in desired_orders}, sell={'sell' in desired_orders})",
                        namespace="trading"
                    )
                    global_state.record_manual_order_cancel(condition_id)

                is_neg_risk = str(row.get('neg_risk', '')).upper() == 'TRUE'
                for side, (price, size) in desired_orders.items():
                    order = {
                        'market': market,
                        'token': token_yes,
                        'price': price,
                        'size': size,
                        'neg_risk': is_neg_risk
                    }
                    _submit_order(order, side.upper())

                global_state.clear_pending_exit(condition_id)

            token_no = str(row['token2'])
            if token_no != token_yes:
                existing_orders_no = get_order(token_no)
                if existing_orders_no['buy']['size'] > 0 or existing_orders_no['sell']['size'] > 0:
                    try:
                        global_state.client.cancel_all_asset(token_no)
                    except Exception as exc:
                        Logan.error(
                            f"Error cancelling legacy orders for token {token_no}",
                            namespace="trading",
                            exception=exc
                        )

        except Exception as exc:
            Logan.error(
                f"Critical error in perform_trade for market {market}",
                namespace="trading",
                exception=exc
            )

        gc.collect()
