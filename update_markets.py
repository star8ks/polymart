import time
from dotenv import load_dotenv
import pandas as pd
from data_updater.trading_utils import get_clob_client
from data_updater.google_utils import get_spreadsheet
from data_updater.find_markets import cleanup_all_markets, get_all_markets, get_all_markets_detailed
from gspread_dataframe import set_with_dataframe
from logan import Logan
from configuration import TCNF

# Initialize global variables
load_dotenv()
spreadsheet = get_spreadsheet()
client = get_clob_client()

wk_all = spreadsheet.worksheet("All Markets")


def update_sheet(data, worksheet):
    all_values = worksheet.get_all_values()
    existing_num_rows = len(all_values)
    existing_num_cols = len(all_values[0]) if all_values else 0

    manual_col = None
    manual_map = None
    manual_rows = None

    if existing_num_rows > 0:
        header = all_values[0]
        rows = all_values[1:]
        existing_df = pd.DataFrame(rows, columns=header)

        manual_col = next(
            (
                col
                for col in existing_df.columns
                if col and col.strip().lower().replace(' ', '_') == 'manual_select'
            ),
            None
        )

        if manual_col and 'condition_id' in existing_df.columns:
            existing_df['condition_id'] = existing_df['condition_id'].astype(
                str)

            manual_series = existing_df[manual_col].fillna(
                '').astype(str).str.strip()
            manual_series = manual_series.replace(
                {'nan': '', 'NaN': '', 'None': ''})

            manual_rows = existing_df[manual_series != ''].copy()
            if not manual_rows.empty:
                manual_rows = manual_rows.drop_duplicates(
                    subset='condition_id', keep='first'
                )

            manual_map = existing_df[['condition_id', manual_col]].copy()
            manual_map = manual_map[manual_map['condition_id'].str.strip(
            ) != '']
            manual_map = manual_map.drop_duplicates(
                subset='condition_id', keep='first')
            manual_map = manual_map.set_index('condition_id')

    data = data.copy()

    if 'condition_id' in data.columns:
        data['condition_id'] = data['condition_id'].astype(str)

    if manual_rows is not None and not manual_rows.empty and 'condition_id' in data.columns:
        current_ids = set(data['condition_id'].astype(str))
        missing_manual = manual_rows[~manual_rows['condition_id'].isin(
            current_ids)].copy()

        if not missing_manual.empty:
            combined_columns = list(dict.fromkeys(
                list(data.columns) + list(missing_manual.columns)))
            data = data.reindex(columns=combined_columns)
            missing_manual = missing_manual.reindex(
                columns=combined_columns, fill_value='')
            data = pd.concat([data, missing_manual],
                             ignore_index=True, sort=False)

    if manual_map is not None and 'condition_id' in data.columns:
        data = data.drop_duplicates(subset='condition_id', keep='first')
        data = data.set_index('condition_id')
        data[manual_col] = manual_map.reindex(data.index)[manual_col]
        data = data.reset_index()
        cols = [manual_col] + [c for c in data.columns if c != manual_col]
        data = data[cols]

    num_rows, num_cols = data.shape
    max_rows = max(num_rows, existing_num_rows)
    max_cols = max(num_cols, existing_num_cols)

    padded_data = pd.DataFrame('', index=range(
        max_rows), columns=range(max_cols))

    padded_data.iloc[:num_rows, :num_cols] = data.values
    padded_data.columns = list(data.columns) + [''] * (max_cols - num_cols)

    set_with_dataframe(
        worksheet,
        padded_data,
        include_index=False,
        include_column_header=True,
        resize=True
    )


def sort_df(df):
    # Calculate the mean and standard deviation for each column
    mean_gm = df['gm_reward_per_100'].mean()
    std_gm = df['gm_reward_per_100'].std()

    mean_volatility = df['volatility_sum'].mean()
    std_volatility = df['volatility_sum'].std()

    # Standardize the columns
    df['std_gm_reward_per_100'] = (df['gm_reward_per_100'] - mean_gm) / std_gm
    df['std_volatility_sum'] = (
        df['volatility_sum'] - mean_volatility) / std_volatility

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
    sorted_df = sorted_df.drop(columns=[
                               'std_gm_reward_per_100', 'std_volatility_sum', 'bid_score', 'ask_score', 'composite_score'])

    return sorted_df


def fetch_and_process_data():
    global spreadsheet, client, wk_all, wk_vol, sel_df

    spreadsheet = get_spreadsheet()
    client = get_clob_client()

    wk_all = spreadsheet.worksheet("All Markets")

    all_markets_df = get_all_markets(client)
    Logan.info(
        f"Fetching information for {len(all_markets_df)} markets.",
        namespace="update_markets"
    )

    all_markets_df = get_all_markets_detailed(all_markets_df, client)
    all_markets_df = cleanup_all_markets(all_markets_df)
    Logan.info(
        f'{pd.to_datetime("now")}: Fetched all markets data with volatility and activity metrics of length {len(all_markets_df)}.',
        namespace="update_markets"
    )

    if len(all_markets_df) > 0:
        update_sheet(all_markets_df, wk_all)
    else:
        Logan.warn(
            'Skipping sheet update because no reward markets were returned.',
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
