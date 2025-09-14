from poly_data.data_utils import get_position


class MarketStrategy:
    @classmethod
    def get_buy_sell_amount(cls, position, row, force_sell=False) -> tuple[float, float]:
        buy_amount = 0
        sell_amount = 0

        trade_size = row.get('trade_size', position) # on sell-only mode
        max_size = row.get('max_size', trade_size)
        
        # effective_position = max(position - other_token_position, 0)
        
        if position < max_size:
            remaining_to_max = max_size - position
            buy_amount = min(trade_size, remaining_to_max)

        if position >= trade_size or force_sell:
            sell_amount = position

        # Ensure minimum order size compliance
        if buy_amount > 0.7 * row['min_size'] and buy_amount < row['min_size']:
            buy_amount = row['min_size']
        if sell_amount > 0.7 * row['min_size'] and sell_amount < row['min_size']:
            sell_amount = row['min_size']

        # if we are selling more than we have;
        if sell_amount > position:
            if force_sell:
                sell_amount = position
            else:
                sell_amount = 0
        
        if force_sell:
            buy_amount = 0

        return buy_amount, sell_amount
    
    @classmethod
    def get_order_prices(cls, best_bid, best_bid_size, best_ask, best_ask_size, avgPrice, row, token, position) -> tuple[float, float]:
        tick = row['tick_size']
        trade_size = row.get('trade_size', position)

        if avgPrice == 0 or avgPrice is None:
            avgPrice = (best_bid + best_ask) / 2
        
        bid_price = best_bid + tick
        ask_price = best_ask - tick

        # If best bid and best ask is not large enough, we don't have to beat them.
        if bid_price > best_bid and best_bid_size < trade_size:
            bid_price = best_bid
        if ask_price < best_ask and best_ask_size < trade_size:
            ask_price = best_ask

        # if we already have position on this token, be lenient to buy more
        pos = get_position(token)
        if pos['size'] >= trade_size:
            bid_price -= 1
        
        # Safety guards
        bid_price = min(bid_price, avgPrice - tick)
        ask_price = max(ask_price, avgPrice + tick)

        if (bid_price + ask_price) >= 1:
            bid_price = best_bid
            ask_price = best_ask
        
        if bid_price < 0.1 or bid_price > 0.9:
            bid_price = best_bid
        if ask_price < 0.1 or ask_price > 0.9:
            ask_price = best_ask

        return bid_price, ask_price