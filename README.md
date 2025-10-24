# Poly-Maker

Poly-Maker now runs as a **passive liquidity rewards engine** for Polymarket. Instead of chasing directional alpha, the process focuses on quoting inside each market's reward band to earn incentive payouts with minimal inventory churn. A summary of the original experience running the legacy trading version is available [here](https://x.com/defiance_cr/status/1906774862254800934).

## Overview

Poly-Maker is a focused solution for passive liquidity provision on Polymarket. It includes:

- Reward-band aware quoting that keeps resting orders inside the incentive window
- Real-time order book monitoring via WebSockets
- Position sizing derived from Google Sheets configuration
- Automated position merging functionality for on-chain efficiency

## Structure

The repository consists of several interconnected modules:

- `poly_data`: Core data management, reward-band quoting, and passive order maintenance
- `poly_merger`: Utility for merging positions (based on open-source Polymarket code)
- `poly_stats`: Account statistics tracking
- `poly_utils`: Shared utility functions, including Google Sheets helpers
- `data_updater`: Separate module for collecting market information and populating sheets

## Requirements

- Python 3.9 with latest setuptools
- Node.js (for `poly_merger`)
- Google Sheets API credentials
- Polymarket account and API credentials

## Installation

1. **Clone the repository**:
```
git clone https://github.com/yourusername/poly-maker.git
cd poly-maker
```

2. **Install Python dependencies**:
```
pip install -r requirements.txt
pip install git+https://github.com/merijjeyn/logan.git
```

3. **Install Node.js dependencies for the merger**:
```
cd poly_merger
npm install
cd ..
```

4. **Set up environment variables**:
```
cp .env.example .env
```

5. **Configure your credentials in `.env`**:
- `PK`: Your private key for Polymarket
- `BROWSER_ADDRESS`: Your wallet address

Make sure your wallet has done at least one trade thru the UI so that the permissions are proper.

6. **Set up Google Sheets integration**:
   - Create a Google Service Account and download json credential file as credentials.json to the main directory
   - Enable Google Sheets API for the google cloud project at [this page](https://console.cloud.google.com/apis/library/sheets.googleapis.com)
   - Copy the [sample Google Sheet](https://docs.google.com/spreadsheets/d/1Kt6yGY7CZpB75cLJJAdWo7LSp9Oz7pjqfuVWwgtn7Ns/edit?gid=1884499063#gid=1884499063)
   - Add your Google service account to the sheet with edit permissions
   - Update `SPREADSHEET_URL` in your `.env` file

7. **Update market data**:
   - Run `python update_markets.py` to fetch incentive information and refresh the Google Sheet tabs
   - Keep this running in the background (ideally on a different IP) so the sheet receives fresh reward-band metrics
   - Verify markets in the "Selected Markets" sheet, which now reflects the filtered reward candidates produced by `filter_selected_markets`
   - Tune liquidity limits and offsets in the "Hyperparameters" sheet if the defaults are not suitable

8. **Start the passive liquidity loop**:
```
python main.py
```
   The bot syncs with the sheet, calculates reward-band quotes, and maintains resting yes/no orders within the incentive window. There is no directional strategy module anymore—only passive quoting for rewards.

## Configuration

The bot is configured via a Google Spreadsheet with several worksheets:

- **Selected Markets**: Output of the reward-focused selection pipeline. `filter_selected_markets` writes back the latest filtered list so operators can audit what the process is quoting. Seeing the "Successfully wrote …" log simply confirms that the sheet reflects the newest passive selection.
- **All Markets**: Database of all markets on Polymarket (continuously refreshed by `update_markets.py`)
- **Hyperparameters**: Configuration parameters used by the reward-band quoting logic (spread offsets, minimum sizes, budget multipliers)


## Poly Merger

The `poly_merger` module is a utility that handles position merging on Polymarket. It's built on open-source Polymarket code and provides a smooth way to consolidate positions, reducing gas fees and improving capital efficiency.

## Important Notes

- This code interacts with real markets and can potentially lose real money
- Test thoroughly with small amounts before deploying with significant capital
- The `data_updater` is technically a separate repository but is included here for convenience

## License

MIT
