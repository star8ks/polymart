from abc import ABC, abstractmethod


class MarketStrategy(ABC):
    """Abstract base class defining the interface for market making strategies."""
    
    @classmethod
    @abstractmethod
    def get_buy_sell_amount(cls, position, row, force_sell=False) -> tuple[float, float]:
        """Calculate buy and sell amounts based on position and market data."""
        pass

    @classmethod
    @abstractmethod
    def get_order_prices(cls, best_bid, best_ask, avgPrice, row, token, tick, force_sell=False) -> tuple[float, float]:
        """Calculate optimal bid and ask prices."""
        pass

    @classmethod
    def apply_safety_guards(cls, bid_price, ask_price, avgPrice, tick, best_bid, best_ask, force_sell=False) -> tuple[float, float]:
        # Safety guards
        if avgPrice > 0.9 or avgPrice < 0.1 or bid_price >= ask_price:
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
        
        return bid_price, ask_price