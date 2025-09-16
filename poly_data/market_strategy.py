from math import log

from logan import Logan
from configuration import TCNF
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
    def get_order_prices(cls, best_bid, best_ask, avgPrice, row, token, tick, force_sell=False) -> tuple[float, float]:
        # We don't have valid data to calculate the prices
        if row['volatility_sum'] == 0 or row['order_arrival_rate_sensitivity'] <= 1:
            return best_bid, best_ask

        if avgPrice == 0 or avgPrice is None:
            avgPrice = (best_bid + best_ask) / 2

        reservation_price = cls.calculate_reservation_price(best_bid, best_ask, row, token)
        optimal_spread = cls.calculate_optimal_spread(row)
        bid_price = reservation_price - optimal_spread/2
        ask_price = reservation_price + optimal_spread/2

        Logan.debug(f"avgPrice: {avgPrice}, Reservation price: {reservation_price}, Optimal spread: {optimal_spread}, Bid price: {bid_price}, Ask price: {ask_price}", namespace="market_strategy")

        # Safety guards
        if avgPrice > 0.9 or avgPrice < 0.1 or bid_price >= ask_price or bid_price + ask_price >= 1:
            bid_price = best_bid
            ask_price = best_ask
        
        # I'm a pussy
        if bid_price >= avgPrice: 
            bid_price = avgPrice - tick
        if ask_price <= avgPrice:
            ask_price = avgPrice + tick
        
        if force_sell and ask_price > best_ask: 
            ask_price = best_ask
        if force_sell and bid_price >= best_bid: 
            bid_price = best_bid - tick

        Logan.debug(f"After safety guards: Bid price: {bid_price}, Ask price: {ask_price}", namespace="market_strategy")
        
        return bid_price, ask_price

    @classmethod
    def calculate_reservation_price(cls, best_bid, best_ask, row, token) -> float:
        pos = get_position(token)
        inventory = pos['size']
        mid_price = cls.calculate_weighted_mid_price(best_bid, best_ask, row['market_order_imbalance'])
        volatility = row['volatility_sum']
        risk_aversion = TCNF.RISK_AVERSION
        time_to_horizon = TCNF.TIME_TO_HORIZON_HOURS
        factor = 0.0000001 # Simply to scale the values to a reasonable range
        Logan.debug(f"Inventory: {inventory}, Volatility: {volatility}, Risk aversion: {risk_aversion}, Time to horizon: {time_to_horizon}, imbalance: {row['market_order_imbalance']}, Factor: {factor}", namespace="market_strategy")
        return mid_price - factor * inventory * risk_aversion * (volatility**2) * time_to_horizon
    
    @classmethod
    def calculate_weighted_mid_price(cls, best_bid, best_ask, imbalance) -> float:
        # Calculates fair price based on the order book imbalance
        return ((1 - imbalance) / 2) * best_bid + ((1 + imbalance) / 2) * best_ask


    @classmethod
    def calculate_optimal_spread(cls, row) -> float:
        risk_aversion = TCNF.RISK_AVERSION
        time_to_horizon = TCNF.TIME_TO_HORIZON_HOURS
        volatility = row['volatility_sum']
        arrival_sensitivity = max(row['order_arrival_rate_sensitivity'], 1)

        Logan.debug(f"Risk aversion: {risk_aversion}, Time to horizon: {time_to_horizon}, Volatility: {volatility}, Arrival sensitivity: {arrival_sensitivity}", namespace="market_strategy")
        factor = 0.00002 # Simply to scale the values to a reasonable range
        left = risk_aversion * (volatility**2) * time_to_horizon
        right = (2/risk_aversion) * log(1 + (risk_aversion / arrival_sensitivity))
        return factor * (left + right)
            

        
        