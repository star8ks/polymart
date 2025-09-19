import json
from sortedcontainers import SortedDict
import poly_data.global_state as global_state
import poly_data.CONSTANTS as CONSTANTS

from trading import perform_trade
import time 
import asyncio
from poly_data.data_utils import set_position, set_order, update_positions
from logan import Logan

def process_book_data(market, json_data):
    global_state.all_data[market] = {
        'bids': SortedDict(),
        'asks': SortedDict()
    }

    global_state.all_data[market]['bids'].update({float(entry['price']): float(entry['size']) for entry in json_data['bids']})
    global_state.all_data[market]['asks'].update({float(entry['price']): float(entry['size']) for entry in json_data['asks']})

def process_price_change(asset, side, price_level, new_size):
    if side == 'bids':
        book = global_state.all_data[asset]['bids']
    else:
        book = global_state.all_data[asset]['asks']

    if new_size == 0:
        if price_level in book:
            del book[price_level]
    else:
        book[price_level] = new_size

def process_data(json_datas, trade=True):
    # Check if json_datas is a dict or a list of dicts
    if isinstance(json_datas, dict):
        json_datas = [json_datas]
    elif not isinstance(json_datas, list):
        Logan.error(f"Expected dict or list of dicts, got: {type(json_datas)}", namespace="poly_data.data_processing")
        return

    for json_data in json_datas:
        event_type = json_data['event_type']
        market = json_data['market']

        if event_type == 'book':
            process_book_data(market, json_data)

            if trade:
                asyncio.create_task(perform_trade(market))
                
        elif event_type == 'price_change':
            for data in json_data['price_changes']:
                side = 'bids' if data['side'] == 'BUY' else 'asks'
                price_level = float(data['price'])
                new_size = float(data['size'])
                process_price_change(market, side, price_level, new_size)

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
    Logan.debug(f"Processing user data, rows: {rows}", namespace="poly_data.data_processing")
    if isinstance(rows, dict):
        rows = [rows]
    elif not isinstance(rows, list):
        Logan.error(f"Expected dict or list of dicts, got: {type(rows)}", namespace="poly_data.data_processing")
        return

    for row in rows:
        market = row['market']

        side = row['side'].lower()
        token = row['asset_id']
            
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
                        maker_outcome = maker_order['outcome'] #this is curious

                        if maker_outcome == taker_outcome:
                            side = 'buy' if side == 'sell' else 'sell' #need to reverse as we reverse token too
                        else:
                            token = global_state.REVERSE_TOKENS[token]
                
                if not is_user_maker:
                    size = float(row['size'])
                    price = float(row['price'])
                    Logan.info(
                        "User is taker",
                        namespace="poly_data.data_processing"
                    )

                Logan.info(
                    f"TRADE EVENT FOR: {row['market']}, ID: {row['id']}, STATUS: {row['status']}, SIDE: {row['side']}, MAKER OUTCOME: {maker_outcome}, TAKER OUTCOME: {taker_outcome}, PROCESSED SIDE: {side}, SIZE: {size}",
                    namespace="poly_data.data_processing"
                ) 


                if row['status'] == 'CONFIRMED' or row['status'] == 'FAILED' :
                    if row['status'] == 'FAILED':
                        Logan.error(
                            f"Trade failed for {token}, decreasing",
                            namespace="poly_data.data_processing"
                        )
                        asyncio.create_task(asyncio.sleep(2))
                        update_positions()
                    else:
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
                    asyncio.create_task(perform_trade(market))
                elif row['status'] == 'MINED':
                    remove_from_performing(col, row['id'])

            elif row['event_type'] == 'order':
                Logan.info(
                    f"ORDER EVENT FOR: {row['market']}, STATUS: {row['status']}, TYPE: {row['type']}, SIDE: {side}, ORIGINAL SIZE: {row['original_size']}, SIZE MATCHED: {row['size_matched']}",
                    namespace="poly_data.data_processing"
                )
                
                set_order(token, side, float(row['original_size']) - float(row['size_matched']), row['price'])
                asyncio.create_task(perform_trade(market))

        else:
            Logan.warn(
                f"User data received for {market} but its not in reverse tokens",
                namespace="poly_data.data_processing"
            )