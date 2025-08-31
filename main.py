import gc                      # Garbage collection
import time                    # Time functions
import asyncio                 # Asynchronous I/O
import threading               # Thread management
from logan import Logan


from poly_data.polymarket_client import PolymarketClient
from poly_data.data_utils import update_markets, update_positions, update_orders, update_liquidity
from poly_data.websocket_handlers import connect_market_websocket, connect_user_websocket
import poly_data.global_state as global_state
from poly_data.data_processing import remove_from_performing
from dotenv import load_dotenv

load_dotenv()

def update_once():
    """
    Initialize the application state by fetching market data, positions, and orders.
    """
    update_positions()  # Get current positions from Polymarket
    update_orders()     # Get current orders from Polymarket
    update_markets()    # Get market information from Google Sheets

def remove_from_pending():
    """
    Clean up stale trades that have been pending for too long (>15 seconds).
    This prevents the system from getting stuck on trades that may have failed.
    """
    try:
        current_time = time.time()
            
        # Iterate through all performing trades
        for col in list(global_state.performing.keys()):
            for trade_id in list(global_state.performing[col]):
                
                try:
                    # If trade has been pending for more than 15 seconds, remove it
                    if current_time - global_state.performing_timestamps[col].get(trade_id, current_time) > 15:
                        Logan.log(f"Removing stale entry {trade_id} from {col} after 15 seconds", type="info", namespace="cleanup")
                        remove_from_performing(col, trade_id)
                except Exception as e:
                    Logan.log(f"Error removing stale trade {trade_id} from {col}: {e}", type="error", namespace="cleanup", exception=e)
    except Exception as e:
        Logan.log(f"Error in remove_from_pending function while cleaning stale trades: {e}", type="error", namespace="cleanup", exception=e)

def update_periodically():
    """
    Background thread function that periodically updates market data, positions and orders.
    - Positions and orders are updated every 5 seconds
    - Market data is updated every 30 seconds (every 6 cycles)
    - Stale pending trades are removed each cycle
    """
    i = 1
    while True:
        time.sleep(5)  # Update every 5 seconds
        
        try:
            # Clean up stale trades
            remove_from_pending()
            
            # Update positions and orders every cycle
            update_positions(avgOnly=True)  # Only update average price, not position size
            update_orders()

            # Update market data every 6th cycle (30 seconds)
            if i % 6 == 0:
                update_markets()
                i = 1
                    
            gc.collect()  # Force garbage collection to free memory
            i += 1
        except Exception as e:
            Logan.log(f"Error in update_periodically background thread (cycle {i}): {e}", type="error", namespace="updater", exception=e)
            
async def main():
    """
    Main application entry point. Initializes client, data, and manages websocket connections.
    """

    Logan.init()
    time.sleep(3)

    # Initialize client
    global_state.client = PolymarketClient()
    
    # Initialize state and fetch initial data
    global_state.all_tokens = []
    update_once()
    Logan.log(f"After initial updates: orders={global_state.orders}, positions={global_state.positions}", type="info", namespace="init")

    Logan.log(f'There are {len(global_state.df)} markets, {len(global_state.positions)} positions and {len(global_state.orders)} orders. Starting positions: {global_state.positions}', type="info", namespace="init")

    # Start background update thread
    update_thread = threading.Thread(target=update_periodically, daemon=True)
    update_thread.start()
    
    # Main loop - maintain websocket connections
    while True:
        try:
            # Connect to market and user websockets simultaneously
            await asyncio.gather(
                connect_market_websocket(global_state.all_tokens), 
                connect_user_websocket()
            )
            Logan.log("Reconnecting to the websocket", type="info", namespace="websocket")
        except Exception as e:
            Logan.log(f"Error in main websocket connection loop: {e}", type="error", namespace="websocket", exception=e)
            
        await asyncio.sleep(1)
        gc.collect()  # Clean up memory

if __name__ == "__main__":
    asyncio.run(main())