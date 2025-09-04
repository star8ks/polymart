import time
import pandas as pd
from data_updater.trading_utils import get_clob_client
from data_updater.google_utils import get_spreadsheet
from data_updater.find_markets import get_all_markets, get_all_results, get_markets, add_volatility_to_df, add_activity_metrics_to_df
from data_updater.activity_metrics import add_activity_metrics_to_market_data
from gspread_dataframe import set_with_dataframe
from logan import Logan
from configuration import TCNF

# Initialize global variables
spreadsheet = get_spreadsheet()
client = get_clob_client()

wk_all = spreadsheet.worksheet("All Markets")


def update_sheet(data, worksheet):
    all_values = worksheet.get_all_values()
    existing_num_rows = len(all_values)
    existing_num_cols = len(all_values[0]) if all_values else 0

    num_rows, num_cols = data.shape
    max_rows = max(num_rows, existing_num_rows)
    max_cols = max(num_cols, existing_num_cols)

    # Create a DataFrame with the maximum size and fill it with empty strings
    padded_data = pd.DataFrame('', index=range(max_rows), columns=range(max_cols))

    # Update the padded DataFrame with the original data and its columns
    padded_data.iloc[:num_rows, :num_cols] = data.values
    padded_data.columns = list(data.columns) + [''] * (max_cols - num_cols)

    # Update the sheet with the padded DataFrame, including column headers
    set_with_dataframe(worksheet, padded_data, include_index=False, include_column_header=True, resize=True)

def sort_df(df):
    # Calculate the mean and standard deviation for each column
    mean_gm = df['gm_reward_per_100'].mean()
    std_gm = df['gm_reward_per_100'].std()
    
    mean_volatility = df['volatility_sum'].mean()
    std_volatility = df['volatility_sum'].std()
    
    # Standardize the columns
    df['std_gm_reward_per_100'] = (df['gm_reward_per_100'] - mean_gm) / std_gm
    df['std_volatility_sum'] = (df['volatility_sum'] - mean_volatility) / std_volatility
    
    # Define a custom scoring function for best_bid and best_ask
    def proximity_score(value):
        if 0.1 <= value <= 0.25:
            return (0.25 - value) / 0.15
        elif 0.75 <= value <= 0.9:
            return (value - 0.75) / 0.15
        else:
            return 0
    
    df['bid_score'] = df['best_bid'].apply(proximity_score)
    df['ask_score'] = df['best_ask'].apply(proximity_score)
    
    # Create a composite score (higher is better for rewards, lower is better for volatility, with proximity scores)
    df['composite_score'] = (
        df['std_gm_reward_per_100'] - 
        df['std_volatility_sum'] + 
        df['bid_score'] + 
        df['ask_score']
    )
    
    # Sort by the composite score in descending order
    sorted_df = df.sort_values(by='composite_score', ascending=False)
    
    # Drop the intermediate columns used for calculation
    sorted_df = sorted_df.drop(columns=['std_gm_reward_per_100', 'std_volatility_sum', 'bid_score', 'ask_score', 'composite_score'])
    
    return sorted_df

def fetch_and_process_data():
    global spreadsheet, client, wk_all, wk_vol, sel_df
    
    spreadsheet = get_spreadsheet()
    client = get_clob_client()

    wk_all = spreadsheet.worksheet("All Markets")



    all_df = get_all_markets(client)
    Logan.info(
        "Got all Markets",
        namespace="update_markets"
    )
    all_results = get_all_results(all_df, client)
    Logan.info(
        "Got all Results",
        namespace="update_markets"
    )
    all_markets = get_markets(all_results)
    Logan.info(
        "Got all orderbook",
        namespace="update_markets"
    )

    Logan.info(
        f'{pd.to_datetime("now")}: Fetched all markets data of length {len(all_markets)}.',
        namespace="update_markets"
    )
    new_df = add_volatility_to_df(all_markets)
    new_df['volatility_sum'] =  new_df['24_hour'] + new_df['7_day'] + new_df['14_day']
    
    Logan.info(
        f'{pd.to_datetime("now")}: Adding activity metrics to market data.',
        namespace="update_markets"
    )
    
    # Add activity metrics to all markets in parallel
    enhanced_markets = add_activity_metrics_to_df(new_df)
    new_df = pd.DataFrame(enhanced_markets)
    
    new_df['volatilty/reward'] = ((new_df['gm_reward_per_100'] / new_df['volatility_sum']).round(2)).astype(str)

    # Define the base columns that should always exist
    base_columns = ['question', 'answer1', 'answer2', 'spread', 'rewards_daily_rate', 'gm_reward_per_100', 'sm_reward_per_100', 'bid_reward_per_100', 'ask_reward_per_100',  'volatility_sum', 'volatilty/reward', 'min_size', '1_hour', '3_hour', '6_hour', '12_hour', '24_hour', '7_day', '30_day',  
                    'best_bid', 'best_ask', 'volatility_price', 'max_spread', 'tick_size', 'depth_yes_in', 'depth_no_in', 'attractiveness_score', 'market_order_imbalance',
                    'neg_risk',  'market_slug', 'token1', 'token2', 'condition_id']
    
    # Add activity metrics columns (these may not exist if there were errors)
    activity_columns = ['total_volume', 'volume_usd', 'decay_weighted_volume', 'avg_daily_volume',
                       'total_trades', 'avg_trades_per_day', 'avg_trades_per_hour',
                       'unique_makers', 'unique_takers', 'unique_traders', 'unique_transactions']
    
    # Select only columns that actually exist in the DataFrame
    available_columns = [col for col in base_columns + activity_columns if col in new_df.columns]
    new_df = new_df[available_columns]

    # Sort all markets by attractiveness_score (no filtering)
    new_df = new_df.sort_values('attractiveness_score', ascending=False)
    

    Logan.info(
        f'{pd.to_datetime("now")}: Fetched select market of length {len(new_df)}.',
        namespace="update_markets"
    )

    if len(new_df) > 50:
        update_sheet(new_df, wk_all)
    else:
        Logan.warn(
            f'{pd.to_datetime("now")}: Not updating sheet because of length {len(new_df)}.',
            namespace="update_markets"
        )

if __name__ == "__main__":
    while True:
        try:
            fetch_and_process_data()
            time.sleep(60 * 60)  # Sleep for an hour
        except Exception as e:
            Logan.error(
                f"Error in market data update loop",
                namespace="update_markets",
                exception=e
            )
