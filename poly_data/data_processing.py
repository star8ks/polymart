import json
import math
from typing import Optional
from sortedcontainers import SortedDict
import poly_data.global_state as global_state
from poly_data.CONSTANTS import _POSITION_EPS

from poly_data.orders_in_flight import clear_order_in_flight
from poly_data.trading_utils import get_best_bid_ask_deets
from trading import perform_trade
import time
import asyncio
from poly_data.data_utils import set_position, set_order, update_positions, get_order
from configuration import TCNF
from logan import Logan


def _safe_float(value, default=0.0):
    try:
        result = float(value)
        if math.isnan(result):
            return default
        return result
    except (TypeError, ValueError):
        return default


def _get_market_row(condition_id: str):
    markets = global_state.get_active_markets()
    if markets is None or len(markets) == 0:
        return None

    try:
        matches = markets[markets['condition_id'].astype(
            str) == str(condition_id)]
    except Exception:
        return None

    if len(matches) == 0:
        return None

    return matches.iloc[0]


def _determine_exit_price(token: str, market_row) -> Optional[float]:
    token_str = str(token)
    tick = _safe_float(getattr(market_row, 'tick_size', None) if hasattr(
        market_row, 'tick_size') else market_row.get('tick_size', None), 0.01)

    order_data = global_state.order_book_data.get(token_str)
    best_bid = None
    if isinstance(order_data, dict):
        bids = order_data.get('bids')
        if bids:
            try:
                best_bid = next(reversed(bids))
            except StopIteration:
                best_bid = None

    if best_bid is not None:
        best_bid = _safe_float(best_bid, None)

    if best_bid is None:
        try:
            book = get_best_bid_ask_deets(token_str, 1)
            raw_bid = book.get('best_bid') if book.get(
                'best_bid') is not None else book.get('top_bid')
            best_bid = _safe_float(raw_bid, None)
        except Exception:
            best_bid = None

    if best_bid is None and market_row is not None:
        source = getattr(market_row, 'best_bid', None)
        if source is None and isinstance(market_row, dict):
            source = market_row.get('best_bid')
        best_bid = _safe_float(source, None)

    if best_bid is None or best_bid <= 0:
        return None

    return max(tick, min(best_bid, 1 - tick))


def _count_active_sides(order_entry):
    if not isinstance(order_entry, dict):
        return 1

    count = 0
    buy = order_entry.get('buy') if isinstance(
        order_entry.get('buy'), dict) else None
    sell = order_entry.get('sell') if isinstance(
        order_entry.get('sell'), dict) else None

    if buy and _safe_float(buy.get('size')) > 0:
        count += 1
    if sell and _safe_float(sell.get('size')) > 0:
        count += 1

    return count if count > 0 else 1


async def _force_exit_after_delay(condition_id: str, token: str):
    delay = max(getattr(TCNF, 'FORCED_EXIT_DELAY_SECONDS', 300), 0)
    if delay <= 0:
        global_state.complete_forced_exit_task(
            condition_id, asyncio.current_task())
        return

    try:
        await asyncio.sleep(delay)

        token_str = str(token)
        position_entry = global_state.positions.get(token_str, {})
        remaining = _safe_float(position_entry.get('size'))
        if remaining <= _POSITION_EPS:
            return

        market_row = _get_market_row(condition_id)
        price = _determine_exit_price(token_str, market_row)
        tick = _safe_float(getattr(market_row, 'tick_size', None) if market_row is not None and hasattr(
            market_row, 'tick_size') else (market_row.get('tick_size') if isinstance(market_row, dict) else None), 0.01)
        if price is None or price <= 0:
            price = max(tick, 0.01)

        existing_orders = get_order(token_str)
        global_state.mark_expected_cancellation(
            token_str, _count_active_sides(existing_orders))
        try:
            global_state.client.cancel_all_asset(token_str)
        except Exception as exc:
            global_state.clear_expected_cancellation(token_str)
            Logan.error(
                f"Error cancelling orders before forced exit for token {token_str}",
                namespace="poly_data.data_processing",
                exception=exc
            )

        is_neg_risk = False
        if market_row is not None:
            source = getattr(market_row, 'neg_risk', None) if hasattr(
                market_row, 'neg_risk') else market_row.get('neg_risk') if isinstance(market_row, dict) else None
            is_neg_risk = str(source).upper() == 'TRUE'

        try:
            global_state.client.create_order(
                token_str,
                'SELL',
                price,
                remaining,
                is_neg_risk
            )
            Logan.warn(
                f"Forced exit submitted for market {condition_id}: sell {remaining:.2f} @ {price:.3f}",
                namespace="poly_data.data_processing"
            )
        except Exception as exc:
            Logan.error(
                f"Failed to execute forced exit for market {condition_id}",
                namespace="poly_data.data_processing",
                exception=exc
            )
    except asyncio.CancelledError:
        raise
    finally:
        global_state.complete_forced_exit_task(
            condition_id, asyncio.current_task())


def _initiate_exit_orders(condition_id: str, token: str, new_size: float):
    condition_id = str(condition_id)
    token_str = str(token)
    size = _safe_float(new_size)
    if size <= _POSITION_EPS:
        return

    market_row = _get_market_row(condition_id)
    if market_row is None:
        Logan.warn(
            f"Unable to locate market row for forced exit scheduling {condition_id}",
            namespace="poly_data.data_processing"
        )
        return

    price = _determine_exit_price(token_str, market_row)
    tick = _safe_float(getattr(market_row, 'tick_size', None) if hasattr(
        market_row, 'tick_size') else market_row.get('tick_size', None), 0.01)
    if price is None or price <= 0:
        price = max(tick, 0.01)

    global_state.set_pending_exit(condition_id, 'sell', price, size)
    Logan.info(
        f"Scheduled exit order for market {condition_id}: sell {size:.2f} @ {price:.3f}",
        namespace="poly_data.data_processing"
    )

    cooldown = max(getattr(TCNF, 'FORCED_EXIT_DELAY_SECONDS', 300), 0)
    if cooldown <= 0:
        return

    global_state.cancel_forced_exit_task(condition_id)
    forced_task = asyncio.create_task(
        _force_exit_after_delay(condition_id, token_str))
    global_state.register_forced_exit_task(condition_id, forced_task)


def _handle_position_update(condition_id: str, token: str, previous_size: float, new_size: float):
    try:
        prev = _safe_float(previous_size)
        current = _safe_float(new_size)

        if current <= _POSITION_EPS:
            if prev > _POSITION_EPS:
                global_state.cancel_forced_exit_task(condition_id)
                global_state.clear_pending_exit(condition_id)
            return

        if current > prev + _POSITION_EPS:
            _initiate_exit_orders(condition_id, token, current)
    except Exception as exc:
        Logan.error(
            f"Error handling position update for market {condition_id}",
            namespace="poly_data.data_processing",
            exception=exc
        )


def sync_order_book_data_for_reverse_token(updated_token: str):
    reverse_token = global_state.REVERSE_TOKENS[updated_token]
    global_state.order_book_data[reverse_token] = {
        'bids': SortedDict(),
        'asks': SortedDict()
    }

    global_state.order_book_data[reverse_token]['asks'].update(
        {1 - price: size for price, size in global_state.order_book_data[updated_token]['bids'].items()})
    global_state.order_book_data[reverse_token]['bids'].update(
        {1 - price: size for price, size in global_state.order_book_data[updated_token]['asks'].items()})


def process_book_data(token: str, json_data):
    global_state.order_book_data[token] = {
        'bids': SortedDict(),
        'asks': SortedDict()
    }

    global_state.order_book_data[token]['bids'].update(
        {float(entry['price']): float(entry['size']) for entry in json_data['bids']})
    global_state.order_book_data[token]['asks'].update(
        {float(entry['price']): float(entry['size']) for entry in json_data['asks']})

    sync_order_book_data_for_reverse_token(token)


def process_price_change(token: str, side, price_level, new_size):
    if side == 'bids':
        book = global_state.order_book_data[token]['bids']
    else:
        book = global_state.order_book_data[token]['asks']

    if new_size == 0:
        if price_level in book:
            del book[price_level]
    else:
        book[price_level] = new_size

    sync_order_book_data_for_reverse_token(token)


def process_data(json_datas, trade=True):
    # Check if json_datas is a dict or a list of dicts
    if isinstance(json_datas, dict):
        json_datas = [json_datas]
    elif not isinstance(json_datas, list):
        Logan.error(
            f"Expected dict or list of dicts, got: {type(json_datas)}", namespace="poly_data.data_processing")
        return

    for json_data in json_datas:
        event_type = json_data['event_type']
        market = json_data['market']
        # token = str(json_data['asset_id'])

        if event_type == 'book':
            token = str(json_data['asset_id'])
            process_book_data(token, json_data)

            if trade:
                asyncio.create_task(perform_trade(market))

        elif event_type == 'price_change':
            for data in json_data['price_changes']:
                token = str(data['asset_id'])
                side = 'bids' if data['side'] == 'BUY' else 'asks'
                price_level = float(data['price'])
                new_size = float(data['size'])
                process_price_change(token, side, price_level, new_size)

                if trade:
                    asyncio.create_task(perform_trade(market))


def add_to_performing(col, id):
    if col not in global_state.performing:
        global_state.performing[col] = set()

    if col not in global_state.performing_timestamps:
        global_state.performing_timestamps[col] = {}

    # Add the trade ID and track its timestamp
    global_state.performing[col].add(id)
    global_state.performing_timestamps[col][id] = time.time()


def remove_from_performing(col, id):
    if col in global_state.performing:
        global_state.performing[col].discard(id)

    if col in global_state.performing_timestamps:
        global_state.performing_timestamps[col].pop(id, None)


def process_user_data(rows):
    if isinstance(rows, dict):
        rows = [rows]
    elif not isinstance(rows, list):
        Logan.error(
            f"Expected dict or list of dicts, got: {type(rows)}", namespace="poly_data.data_processing")
        return

    for row in rows:
        market = row['market']
        market_str = str(market)

        side = row['side'].lower()
        token = str(row['asset_id'])

        if token in global_state.REVERSE_TOKENS:
            col = token + "_" + side

            if row['event_type'] == 'trade':
                size = 0
                price = 0
                maker_outcome = ""
                taker_outcome = row['outcome']

                is_user_maker = False
                for maker_order in row['maker_orders']:
                    if maker_order['maker_address'].lower() == global_state.client.browser_wallet.lower():
                        Logan.info(
                            "User is maker",
                            namespace="poly_data.data_processing"
                        )
                        size = float(maker_order['matched_amount'])
                        price = float(maker_order['price'])

                        is_user_maker = True
                        # this is curious
                        maker_outcome = maker_order['outcome']

                        if maker_outcome == taker_outcome:
                            # need to reverse as we reverse token too
                            side = 'buy' if side == 'sell' else 'sell'
                        else:
                            token = str(global_state.REVERSE_TOKENS[token])

                col = f"{token}_{side}"

                if not is_user_maker:
                    size = float(row['size'])
                    price = float(row['price'])
                    Logan.info(
                        "User is taker",
                        namespace="poly_data.data_processing"
                    )

                prev_entry = global_state.positions.get(token, {})
                prev_size = _safe_float(prev_entry.get('size'))

                Logan.info(
                    f"TRADE EVENT FOR: {row['market']}, ID: {row['id']}, STATUS: {row['status']}, SIDE: {row['side']}, MAKER OUTCOME: {maker_outcome}, TAKER OUTCOME: {taker_outcome}, PROCESSED SIDE: {side}, SIZE: {size}",
                    namespace="poly_data.data_processing"
                )

                if row['status'] == 'FAILED':
                    Logan.error(
                        f"Trade failed for {token}, decreasing",
                        namespace="poly_data.data_processing"
                    )
                    asyncio.create_task(asyncio.sleep(2))
                    update_positions()
                elif row['status'] == 'CONFIRMED':
                    remove_from_performing(col, row['id'])
                    Logan.info(
                        f"Confirmed. Performing is {len(global_state.performing[col])}",
                        namespace="poly_data.data_processing"
                    )
                    asyncio.create_task(perform_trade(market))
                elif row['status'] == 'MATCHED':
                    add_to_performing(col, row['id'])

                    Logan.info(
                        f"Matched. Performing is {len(global_state.performing[col])}",
                        namespace="poly_data.data_processing"
                    )
                    set_position(token, side, size, price)
                    Logan.info(
                        f"Position after matching is {global_state.positions[str(token)]}",
                        namespace="poly_data.data_processing"
                    )
                    new_size = _safe_float(
                        global_state.positions.get(str(token), {}).get('size'))
                    _handle_position_update(
                        market_str, token, prev_size, new_size)
                    asyncio.create_task(perform_trade(market))
                elif row['status'] == 'MINED':
                    remove_from_performing(col, row['id'])

            elif row['event_type'] == 'order':
                Logan.info(
                    f"ORDER EVENT FOR: {row['market']}, STATUS: {row['status']}, TYPE: {row['type']}, SIDE: {side}, ORIGINAL SIZE: {row['original_size']}, SIZE MATCHED: {row['size_matched']}",
                    namespace="poly_data.data_processing"
                )

                expected_cancel = global_state.consume_expected_cancellation(
                    token)
                manual_market = market_str in global_state.manual_condition_ids
                schedule_trade = True
                reason_text = row.get('reason') or row.get(
                    'status_reason') or row.get('message') or row.get('error')

                try:
                    # size of existing orders
                    order_size = global_state.orders[token][side]['size']
                except Exception as e:
                    order_size = 0

                if row['type'] == 'PLACEMENT':
                    order_size += float(row['original_size'])
                elif row['type'] == 'UPDATE':
                    order_size -= float(row['size_matched'])
                elif row['type'] == 'CANCELLATION':
                    order_size -= float(row['original_size'])
                    if expected_cancel:
                        pass
                    elif manual_market:
                        schedule_trade = False
                        msg = reason_text or 'remote cancellation'
                        global_state.set_manual_unserviceable(
                            market_str, f"order cancelled upstream ({msg})")
                        global_state.record_manual_order_cancel(market_str)
                        Logan.warn(
                            f"Manual market {market_str} cancellation originated upstream ({msg}); skipping re-place",
                            namespace="poly_data.data_processing"
                        )
                    else:
                        # ensure we do not accumulate stale expectations
                        global_state.clear_expected_cancellation(token)

                set_order(token, side, order_size, row['price'])
                clear_order_in_flight(row['id'])

                if schedule_trade:
                    asyncio.create_task(perform_trade(market))
