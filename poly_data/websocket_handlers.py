import asyncio                      # Asynchronous I/O
import gc
import json                        # JSON handling
import websockets                  # WebSocket client
from logan import Logan

from poly_data.data_processing import process_data, process_user_data
import poly_data.global_state as global_state
from configuration import MCNF


async def connect_market_websocket(chunk):
    """
    Connect to Polymarket's market WebSocket API and process market updates.

    This function:
    1. Establishes a WebSocket connection to the Polymarket API
    2. Subscribes to updates for a specified list of market tokens
    3. Processes incoming order book and price updates

    Args:
        chunk (list): List of token IDs to subscribe to

    Notes:
        If the connection is lost, the while loop will attempt to reconnect
    """
    base_tokens = []
    if chunk:
        base_tokens = [str(tok) for tok in chunk]

    uri = "wss://ws-subscriptions-clob.polymarket.com/ws/market"
    last_version = -1

    while True:
        tokens, version = global_state.get_token_snapshot()
        if base_tokens:
            combined = base_tokens + tokens
            # Preserve order while deduplicating
            seen = set()
            tokens = [t for t in combined if not (t in seen or seen.add(t))]

        if not tokens:
            await asyncio.sleep(1)
            continue

        sleep_time = 3

        try:
            async with websockets.connect(uri, ping_interval=MCNF.WEBSOCKET_PING_INTERVAL, ping_timeout=None) as websocket:
                subscription = {"assets_ids": tokens}
                await websocket.send(json.dumps(subscription))
                last_version = version

                Logan.info(
                    f"Subscribed to {len(tokens)} assets (version {last_version})",
                    namespace="websocket_handlers"
                )

                resubscribe = False

                try:
                    while True:
                        if global_state.get_token_version() != last_version:
                            resubscribe = True
                            Logan.info(
                                "Detected token set update; refreshing market subscriptions",
                                namespace="websocket_handlers"
                            )
                            break

                        try:
                            message = await asyncio.wait_for(websocket.recv(), timeout=5)
                        except asyncio.TimeoutError:
                            continue

                        json_data = json.loads(message)
                        # Process order book updates and trigger trading as needed
                        process_data(json_data)

                except websockets.ConnectionClosed as e:
                    Logan.error(
                        "Market websocket connection closed unexpectedly",
                        namespace="websocket_handlers",
                        exception=e
                    )
                except Exception as e:
                    Logan.error(
                        "Unexpected error in market websocket connection",
                        namespace="websocket_handlers",
                        exception=e
                    )

                if resubscribe:
                    sleep_time = 0

        except Exception as e:
            Logan.error(
                "Failed to establish market websocket connection",
                namespace="websocket_handlers",
                exception=e
            )

        await asyncio.sleep(sleep_time)
        gc.collect()


async def connect_user_websocket():
    """
    Connect to Polymarket's user WebSocket API and process order/trade updates.

    This function:
    1. Establishes a WebSocket connection to the Polymarket user API
    2. Authenticates using API credentials
    3. Processes incoming order and trade updates for the user

    Notes:
        If the connection is lost, the function will exit and the main loop will
        attempt to reconnect after a short delay.
    """
    uri = "wss://ws-subscriptions-clob.polymarket.com/ws/user"

    while True:
        async with websockets.connect(uri, ping_interval=MCNF.WEBSOCKET_PING_INTERVAL, ping_timeout=None) as websocket:
            # Prepare authentication message with API credentials
            message = {
                "type": "user",
                "auth": {
                    "apiKey": global_state.client.client.creds.api_key,
                    "secret": global_state.client.client.creds.api_secret,
                    "passphrase": global_state.client.client.creds.api_passphrase
                }
            }

            # Send authentication message
            await websocket.send(json.dumps(message))

            Logan.info(
                "Sent user subscription message",
                namespace="websocket_handlers"
            )

            try:
                # Process incoming user data indefinitely
                while True:
                    message = await websocket.recv()
                    json_data = json.loads(message)
                    # Process trade and order updates
                    process_user_data(json_data)
            except websockets.ConnectionClosed as e:
                Logan.error(
                    "User websocket connection closed unexpectedly",
                    namespace="websocket_handlers",
                    exception=e
                )
            except Exception as e:
                Logan.error(
                    f"Unexpected error in user websocket connection",
                    namespace="websocket_handlers",
                    exception=e
                )
            finally:
                # Brief delay before attempting to reconnect
                await asyncio.sleep(3)
                gc.collect()
