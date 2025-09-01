"""
Configuration settings for the Poly-Maker trading system.

This module contains all constants used throughout the application,
organized into logical groups for different system components.
"""

class TradingConfig:
    """Configuration constants for trading logic and operations."""
    
    # Order pricing and execution thresholds
    SELL_ONLY_THRESHOLD = 0.6
    MIN_PRICE_LIMIT = 0.1
    MAX_PRICE_LIMIT = 0.9
    PRICE_PRECISION_LIMIT = 0.99  # Box sum guard threshold
    
    # Order cancellation thresholds
    BUY_PRICE_DIFF_THRESHOLD = 0.002  # Cancel if price diff > 0.2 cents
    SELL_PRICE_DIFF_THRESHOLD = 0.001  # Cancel if price diff > 0.1 cents
    SIZE_DIFF_PERCENTAGE = 0.1  # Cancel if size diff > 10%
    
    # Position merging and size limits
    MIN_MERGE_SIZE = 20  # From CONSTANTS.py
    DUST_POSITION_THRESHOLD = 1  # Positions smaller than this are ignored
    
    # Market selection and investment parameters
    INVESTMENT_CEILING = 2000
    MAX_POSITION_MULT = 3
    BUDGET_MULT = 0.5
    MARKET_COUNT = 10
    
    # Risk management thresholds
    MAX_VOLATILITY_SUM = 20.0
    MIN_ATTRACTIVENESS_SCORE = 0.0
    
    # Price calculation parameters
    VOLATILITY_THRESHOLD_COMMENT = 0.05  # Referenced in commented code
    SIGNATURE_TYPE = 1  # For Polymarket client
    


class MarketProcessConfig:
    """Configuration constants for market data processing and updates."""
    
    # Update intervals and timing
    POSITION_UPDATE_INTERVAL = 5  # seconds
    MARKET_UPDATE_INTERVAL = 30  # seconds
    STALE_TRADE_TIMEOUT = 15  # seconds to wait before removing stale trades
    
    # Calculated cycle count (how many position update cycles = 1 market update cycle)
    @property
    def MARKET_UPDATE_CYCLE_COUNT(self):
        import math
        return math.ceil(self.MARKET_UPDATE_INTERVAL / self.POSITION_UPDATE_INTERVAL)
    
    # WebSocket and API configuration
    WEBSOCKET_PING_INTERVAL = 5
    HTTP_TIMEOUT = 30  # seconds for HTTP requests
    
    # Data processing constants
    TICK_SIZE_CALCULATION_FACTOR = 100  # For price calculations
    
    # Market depth calculation
    MARKET_DEPTH_EPS = 1e-6  # Small value to avoid division by zero
    DEFAULT_IN_GAME_MULTIPLIER = 1.0
    DEFAULT_ALPHA = 0.1
    
    # Price range calculations
    TICK_SIZE_OFFSET = 1  # For rounding start point in generate_numbers
    PRICE_CALCULATION_PRECISION = 100  # For 1/price * 100 calculations


# Singleton instances for easy import
TCNF = TradingConfig()
MCNF = MarketProcessConfig()