# Update Markets Flow Analysis

## Current Architecture Overview

The market data flow in the poly-maker system involves two separate processes with distinct roles:

### 1. Data Collection Process (`update_markets.py`)

This standalone script continuously updates market data in Google Sheets:

**Flow:**
1. **Fetch All Markets**: Gets comprehensive market data from Polymarket API using `get_all_markets(client)`
2. **Process Market Data**: For each market, calls `get_all_results()` which:
   - Fetches order book data for each market
   - Calculates reward metrics (bid_reward_per_100, ask_reward_per_100, gm_reward_per_100)
   - Computes market depth and attractiveness scores
   - Processes ~40 markets at a time with rate limiting (10s between batches)
3. **Add Volatility Metrics**: Calls `add_volatility_to_df()` which:
   - Fetches price history for each market
   - Calculates volatility across multiple timeframes (1h, 3h, 6h, 12h, 24h, 7d, 14d, 30d)
   - Computes `volatility_sum` as sum of 24h + 7d + 14d volatility
4. **Write to Google Sheets**:
   - **"All Markets"**: Complete processed market data sorted by attractiveness_score
   - **"Volatility Markets"**: Filtered subset where `volatility_sum < 20`, sorted by attractiveness_score  
   - **"Full Markets"**: Raw market data (`m_data`) before volatility calculations
5. **Continuous Loop**: Repeats every hour

**Key Functions:**
- `get_all_markets()`: Fetches market list from Polymarket
- `get_all_results()`: Processes order books and calculates rewards
- `add_volatility_to_df()`: Adds historical volatility metrics
- `sort_df()`: Composite scoring (currently unused, replaced by attractiveness_score)

### 2. Trading Bot Process (`main.py` + `poly_data/`)

The main trading bot reads processed data and makes trading decisions:

**Data Reading Flow (`poly_data/utils.py:get_sheet_df()`):**
1. Reads "Volatility Markets" sheet as the primary filter (`vol` variable)
2. Reads "All Markets" sheet for complete data (`all` variable)  
3. **Inner joins** Volatility Markets with All Markets on 'question' field
4. Result: Only markets that passed volatility filter (`volatility_sum < 20`) with full data

**Market Selection (`poly_data/market_selection.py`):**
1. `filter_selected_markets()`: Further filters to top 10 markets by attractiveness_score
2. `calculate_position_sizes()`: Allocates budget proportionally based on attractiveness_score
3. Result stored in `global_state.selected_markets_df`

**Active Markets (`global_state.py`):**
- Combines `selected_markets_df` + `markets_with_positions` (existing positions)
- This is what the trading logic actually uses

## Key Issues with Current Architecture

1. **Rigid Volatility Filter**: Hard-coded `volatility_sum < 20` filter in update process
2. **Data Loss**: Markets filtered out in update process are unavailable to trading bot
3. **Separation of Concerns**: Update process makes filtering decisions that should be in trading logic
4. **Limited Flexibility**: Can't dynamically adjust volatility thresholds or use different filtering strategies

## Google Sheets Usage

- **"All Markets"**: Complete market data with all metrics
- **"Volatility Markets"**: Pre-filtered subset (volatility_sum < 20)
- **"Full Markets"**: Raw market data before volatility calculations (appears to be for debugging/analysis)
- **"Selected Markets"**: Manual market selection (currently unused in code)
- **"Hyperparameters"**: Trading parameters

## Rate Limiting & Performance

- **Order Book API**: 50 requests/10s → 40 markets per batch with 10s delays
- **Price History API**: 100 requests/10s → 40 markets per batch with 10s delays  
- **Total Processing Time**: ~2-3 minutes for 200+ markets
- **Update Frequency**: Every 60 minutes

## Data Dependencies

```
update_markets.py → Google Sheets → poly_data/utils.py → Trading Logic
     (writes)         (storage)         (reads)        (consumes)
```

The trading bot is completely dependent on the update process to provide filtered, processed data.