import gc
import asyncio
import math

from logan import Logan

import poly_data.global_state as global_state
from configuration import TCNF
from poly_data.orders_in_flight import get_orders_in_flight, set_order_in_flight
from poly_data.trading_utils import get_best_bid_ask_deets, round_down, round_up
from poly_data.data_utils import get_position, get_order
from poly_data.market_selection import get_enhanced_market_row


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


def _compute_yes_quotes(midpoint, tick, round_length, row):
    reward_floor = _safe_float(row.get('reward_bid_floor'))
    reward_ceiling = _safe_float(row.get('reward_ask_ceiling'))
    reward_half = _safe_float(row.get('reward_half_spread'))

    if reward_half <= 0:
        reward_half = _safe_float(row.get('max_spread')) / 100.0

    if reward_floor <= 0 or reward_ceiling <= 0 or reward_ceiling <= reward_floor:
        if reward_half <= 0:
            return None, None
        reward_floor = max(midpoint - reward_half, tick)
        reward_ceiling = min(midpoint + reward_half, 1 - tick)
    else:
        reward_floor = max(reward_floor, tick)
        reward_ceiling = min(reward_ceiling, 1 - tick)
        if reward_ceiling <= reward_floor:
            return None, None
        inferred_half = (reward_ceiling - reward_floor) / 2.0
        if inferred_half > 0:
            reward_half = inferred_half if reward_half <= 0 else min(
                reward_half, inferred_half)

    band_width = reward_ceiling - reward_floor
    if band_width <= 0 or band_width <= 2 * tick:
        return None, None

    edge_offset = tick * \
        max(int(getattr(TCNF, 'REWARD_EDGE_OFFSET_TICKS', 1)), 0)
    passive_offset = tick * \
        max(int(getattr(TCNF, 'REWARD_PASSIVE_OFFSET_TICKS', 0)), 0)

    max_edge = max(reward_half - tick, 0.0)
    if edge_offset > max_edge:
        edge_offset = max_edge

    max_passive = max(reward_half - edge_offset - tick, 0.0)
    if passive_offset > max_passive:
        passive_offset = max_passive

    base_bid = midpoint - reward_half + edge_offset
    base_ask = midpoint + reward_half - edge_offset

    base_bid = max(reward_floor, min(base_bid, reward_ceiling - tick))
    base_ask = min(reward_ceiling, max(base_ask, reward_floor + tick))

    bid_price = max(reward_floor, min(
        base_bid - passive_offset, reward_ceiling - tick))
    ask_price = min(reward_ceiling, max(
        base_ask + passive_offset, reward_floor + tick))

    bid_price = round_down(max(bid_price, reward_floor), round_length)
    ask_price = round_up(min(ask_price, reward_ceiling), round_length)

    bid_price = max(reward_floor, min(bid_price, reward_ceiling - tick))
    ask_price = min(reward_ceiling, max(ask_price, reward_floor + tick))

    if bid_price >= ask_price:
        midpoint_clamped = min(
            max(midpoint, reward_floor + tick), reward_ceiling - tick)
        bid_price = round_down(
            max(midpoint_clamped - tick, reward_floor), round_length)
        ask_price = round_up(
            min(midpoint_clamped + tick, reward_ceiling), round_length)

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
                return

            token_yes = str(row['token1'])
            if token_yes not in global_state.order_book_data:
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
                return

            best_bid = book.get('best_bid') if book.get(
                'best_bid') is not None else book.get('top_bid')
            best_ask = book.get('best_ask') if book.get(
                'best_ask') is not None else book.get('top_ask')

            if best_bid is None or best_ask is None:
                best_bid = _safe_float(row.get('best_bid'))
                best_ask = _safe_float(row.get('best_ask'))

            if best_bid is None or best_ask is None or best_bid <= 0 or best_ask >= 1:
                return

            midpoint = (best_bid + best_ask) / 2
            bid_yes, ask_yes = _compute_yes_quotes(
                midpoint, tick, tick_decimals, row)
            if bid_yes is None or ask_yes is None:
                return

            trade_size = _determine_trade_size(row)
            if trade_size <= 0:
                return

            min_size = _safe_float(row.get('min_size'))
            if min_size > 0 and trade_size < min_size:
                trade_size = min_size

            position_yes = _safe_float(get_position(token_yes)['size'])
            max_position = max(trade_size, _safe_float(
                row.get('max_size'), trade_size))

            liquidity = _safe_float(global_state.available_liquidity)

            remaining_capacity = max(max_position - position_yes, 0.0)

            desired_orders = {}

            buy_size = 0.0
            if liquidity > 0 and remaining_capacity > 0:
                max_affordable = liquidity / max(bid_yes, tick)
                buy_size = min(trade_size, max_affordable, remaining_capacity)
                if min_size > 0 and buy_size < min_size:
                    buy_size = 0.0

            allow_buy = buy_size > 0
            if allow_buy:
                desired_orders['buy'] = (bid_yes, buy_size)

            sell_inventory = max(position_yes, 0.0)
            sell_size = min(trade_size, sell_inventory)
            if min_size > 0 and sell_size < min_size:
                sell_size = 0.0

            allow_sell = sell_size > 0
            if allow_sell:
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
                return

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
                        f"Resetting orders for {token_yes} (buy={allow_buy}, sell={allow_sell})",
                        namespace="trading"
                    )

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
