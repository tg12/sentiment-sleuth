"""THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS
OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF
MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE, TITLE AND
NON-INFRINGEMENT. IN NO EVENT SHALL THE COPYRIGHT HOLDERS OR ANYONE
DISTRIBUTING THE SOFTWARE BE LIABLE FOR ANY DAMAGES OR OTHER LIABILITY,
WHETHER IN CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN
CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
"""

# pylint: disable=C0116, W0621, W1203, C0103, C0301, W1201
# Author : James Sawyer
# Maintainer : James Sawyer
# Version : 3.8
# Status : Production
# Copyright : Copyright (c) 2024 James Sawyer

import json
import logging
import sys
from datetime import datetime

import numpy as np
import pandas as pd
import pytz
import requests
from fake_useragent import UserAgent
from matplotlib import pyplot as plt
from scipy.stats import linregress
from tabulate import tabulate

TIMEOUT_SECONDS = 5

MARKET_CONDITION_TOO_SMALL = 0
MARKET_FLAT_DIRECTIONLESS = 1
MARKET_TREND_UP_STRONG = 2
MARKET_TREND_UP_WEAK = 3
MARKET_TREND_DOWN_STRONG = 4
MARKET_TREND_DOWN_WEAK = 5

epic_id = "CC.D.NG.USS.IP" # Natural Gas

def assess_market_trend_with_codes(data, volatility_threshold=0.001):
    """
    Assess the market trend and return coded values indicating the market condition.
    This version applies weights to the log returns, giving more significance to earlier entries.

    Parameters:
    - data: DataFrame with 'close' column, representing close prices over time.

    Returns:
    - A tuple: (Return code, R-squared value)
      Return code indicates the market condition based on trend analysis.
      R-squared value quantifies the trend strength.
    """

    # for debugging create a graph with the log returns
    plt.figure()

    if len(data) < 4:
        return (MARKET_CONDITION_TOO_SMALL, None)

    log_returns = np.log(data["close"] / data["close"].shift(1))

    # Generate weights for log returns: Higher for earlier prices, decreasing linearly
    weights_log_returns = np.linspace(start=1, stop=0, num=len(log_returns))

    # Weighted standard deviation for volatility (considering only non-NaN values for correctness)
    valid_weights = weights_log_returns[~np.isnan(log_returns)]
    valid_log_returns = log_returns[~np.isnan(log_returns)]
    average_log_return = np.average(valid_log_returns, weights=valid_weights)
    variance = np.average((valid_log_returns - average_log_return)**2, weights=valid_weights)
    volatility = np.sqrt(variance)

    plt.plot(weights_log_returns, log_returns)
    
    plt.xlabel("Weights")
    plt.ylabel("Log Returns")
    plt.title("Log Returns with Weights")
    plt.tight_layout()
    # reduce the font size of the x-axis
    plt.xticks(fontsize=8)
    plt.show()

    if volatility < volatility_threshold:
        return (MARKET_FLAT_DIRECTIONLESS, None)

    log_close_prices = np.log(data["close"])
    time_index = np.arange(len(log_close_prices))

    # Perform linear regression without considering weights for log returns here
    slope, intercept, r_value, p_value, std_err = linregress(time_index, log_close_prices)
    
    trend_strength = r_value**2

    if p_value < 0.05:
        if slope > 0:
            if trend_strength > 0.5:
                return (MARKET_TREND_UP_STRONG, trend_strength)
            else:
                return (MARKET_TREND_UP_WEAK, trend_strength)
        else:
            if trend_strength > 0.5:
                return (MARKET_TREND_DOWN_STRONG, trend_strength)
            else:
                return (MARKET_TREND_DOWN_WEAK, trend_strength)
    else:
        return (MARKET_FLAT_DIRECTIONLESS, trend_strength)


class IGAPIClient:
    def __init__(self, epic_id, live=False):
        try:
            self.ua = UserAgent()
        except Exception as e:
            logger.error("Failed to instantiate UserAgent: %s", e)
            sys.exit(0)

        self.live = live
        self.epic_id = epic_id
        self.base_url = self.get_base_url()
        self.auth_headers = self.authenticate()

    def get_base_url(self):
        if self.live:
            # add warning
            logger.warning("Using live account. Please be careful!")
            sys.exit(1)
            return "https://api.ig.com/gateway/deal"
        else:
            return "https://demo-api.ig.com/gateway/deal"

    def authenticate(self):
        endpoint = f"{self.base_url}/session"
        headers = self._get_common_headers()
        credentials = self.get_credentials()
        headers["X-IG-API-KEY"] = credentials["api_key"]

        try:
            payload = {
                "identifier": credentials["username"],
                "password": credentials["password"],
            }
            response = requests.post(
                endpoint,
                json=payload,
                headers=headers,
                timeout=TIMEOUT_SECONDS,
            )
        except Exception as e:
            logger.error("Request failed: %s", e)
            sys.exit(0)

        if response.status_code != 200:
            logger.error("Authentication failed: %s", response.text)
            sys.exit(0)

        auth_headers = headers.copy()
        auth_headers.update(
            {
                "CST": response.headers.get("CST", ""),
                "X-SECURITY-TOKEN": response.headers.get("X-SECURITY-TOKEN", ""),
                "X-IG-API-KEY": credentials["api_key"],
            },
        )

        return auth_headers

    def get_credentials(self):
        # Add logic to fetch credentials from a secure location
        if self.live:
            return {
                "api_key": "",
                "username": "",
                "password": "",
                "account_id": "",
            }
        else:
            return {
                "api_key": "", # demo key
                "username": "",
                "password": "",
                "account_id": "",
            }

    @staticmethod
    def _get_common_headers():
        return {
            "Content-Type": "application/json; charset=UTF-8",
            "Accept": "application/json, application/json; charset=UTF-8, */*; q=0.01",
            "Accept-Encoding": "gzip, deflate, br",
            "Accept-Language": "en-US,en;q=0.9",
            "Version": "2",
            "User-Agent": str(UserAgent().random),
            "Cache-Control": "no-store",
            "Pragma": "no-cache",
            "Connection": "keep-alive",
            "Upgrade-Insecure-Requests": "1",
            "Sec-Fetch-Dest": "empty",
            "Sec-Fetch-Mode": "cors",
            "Sec-Fetch-Site": "same-origin",
        }

    def get_market_info(self):
        endpoint = f"{self.base_url}/markets/{self.epic_id}"
        try:
            response = requests.get(
                endpoint,
                headers=self.auth_headers,
                timeout=TIMEOUT_SECONDS,
            )
            response.raise_for_status()
            # print(json.dumps(json.loads(response.text), indent=4,
            # sort_keys=True)) #debug
            pretty_json = json.dumps(
                json.loads(response.text),
                indent=4,
                sort_keys=True,
            )
        except Exception as e:
            logger.error("Failed to get market info: %s", e)
            return None

        return pretty_json

    @staticmethod
    def midpoint(bid, ask):
        try:
            return (bid + ask) / 2
        except Exception as e:
            logger.error("Failed to calculate midpoint: %s", e)
            return None

    @staticmethod
    def humanize_time(secs):
        try:
            mins, secs = divmod(secs, 60)
            hours, mins = divmod(mins, 60)
            return f"{hours:02d}:{mins:02d}:{secs:02d}"
        except Exception as e:
            logger.error("Failed to humanize time: %s", e)
            return "00:00:00"

    def get_prices(self, resolution):
        all_close_prices = []
        all_open_prices = []
        all_high_prices = []
        all_low_prices = []
        all_snapshotTimes = []
        endpoint = f"{self.base_url}/prices/{self.epic_id}/{resolution}"

        try:
            response = requests.get(
                endpoint,
                headers=self.auth_headers,
                timeout=TIMEOUT_SECONDS,
            )
            response.raise_for_status()
            data = json.loads(response.text)
        except Exception as e:
            logger.error("Failed to get prices: %s", e)
            return pd.DataFrame()

        # Check if 'prices' and 'allowance' keys exist
        if "prices" not in data or "allowance" not in data:
            logger.error("Invalid response data: 'prices' or 'allowance' key missing.")
            return pd.DataFrame()

        remaining_allowance = data["allowance"].get("remainingAllowance", "UNKNOWN")
        reset_time = self.humanize_time(
            int(data["allowance"].get("allowanceExpiry", 0)),
        )

        logger.debug(f"Remaining API Calls left: {remaining_allowance}")
        logger.debug(f"Time to API Key reset: {reset_time}")

        for i in data["prices"]:

            # get open, high, low, close prices
            bid_price_open = np.float64(i["openPrice"]["bid"])
            ask_price_open = np.float64(i["openPrice"]["ask"])
            mid_price_open = self.midpoint(bid_price_open, ask_price_open)
            # # round to 2 decimal places
            # mid_price_open = round(mid_price_open, 2)
            # logger.debug(f"Open: {mid_price_open}")

            bid_price_high = np.float64(i["highPrice"]["bid"])
            ask_price_high = np.float64(i["highPrice"]["ask"])
            mid_price_high = self.midpoint(bid_price_high, ask_price_high)
            # # round to 2 decimal places
            # mid_price_high = round(mid_price_high, 2)
            # logger.debug(f"High: {mid_price_high}")

            bid_price_low = np.float64(i["lowPrice"]["bid"])
            ask_price_low = np.float64(i["lowPrice"]["ask"])
            mid_price_low = self.midpoint(bid_price_low, ask_price_low)
            # # round to 2 decimal places
            # mid_price_low = round(mid_price_low, 2)
            # logger.debug(f"Low: {mid_price_low}")

            bid_price_close = np.float64(i["closePrice"]["bid"])
            ask_price_close = np.float64(i["closePrice"]["ask"])
            mid_price_close = self.midpoint(bid_price_close, ask_price_close)
            # # round to 2 decimal places
            # mid_price_close = round(mid_price_close, 2)
            # logger.debug(f"Close: {mid_price_close}")

            # validate there are no NaN values and there is a valid price for each high, low, open, close
            if (
                np.isnan(mid_price_open)
                or np.isnan(mid_price_high)
                or np.isnan(mid_price_low)
                or np.isnan(mid_price_close)
            ):
                logger.error("NaN values detected. Skipping...")
                continue

            # get snapshot time
            snapshotTime = datetime.strptime(
                i["snapshotTime"],
                "%Y/%m/%d %H:%M:%S",
            ).strftime("%Y:%m:%d-%H:%M:%S")
            all_snapshotTimes.append(snapshotTime)

            # append prices
            all_open_prices.append(mid_price_open)
            all_high_prices.append(mid_price_high)
            all_low_prices.append(mid_price_low)
            all_close_prices.append(mid_price_close)

        # verify that "close" is the same length as resolution
        # no missing data
        if len(all_close_prices) != int(resolution.split("/")[1]):
            logger.error(
                "Close prices are not the same length as resolution. Skipping...",
            )
            return pd.DataFrame()
        else:
            logger.debug(
                "Close prices are the same length as resolution. Continuing...",
            )

        return pd.DataFrame(
            {
                "snapshotTime": all_snapshotTimes,
                "open": all_open_prices,
                "high": all_high_prices,
                "low": all_low_prices,
                "close": all_close_prices,
            },
        )

    def get_backtest_data(self, resolutions):
        try:
            dfs = [self.get_prices(resolution) for resolution in resolutions]
        except Exception as e:
            logger.error("Failed to get backtest data for resolutions: %s", e)
            sys.exit(0)

        try:
            backtest_data = (
                pd.concat(dfs)
                .reset_index(drop=True)
                .sort_values(by=["snapshotTime"], ascending=True)
            )
            # drop any NaN values
            backtest_data = backtest_data.dropna()
            backtest_data = backtest_data.reset_index(drop=True)
            # drop any duplicate times, keeping the first
            backtest_data = backtest_data.drop_duplicates(
                subset=["snapshotTime"],
                keep="first",
            )
        except Exception as e:
            logger.error("Failed to concatenate dataframes or write to CSV: %s", e)
            sys.exit(0)

        # make sure ALL the data is rounded to 2 decimal places
        backtest_data = backtest_data.round(2)

        print(tabulate(backtest_data, headers="keys", tablefmt="psql"))
        logger.info("Writing to CSV...")
        backtest_data.to_csv("backtest_prices.csv", index=False)

        return backtest_data


if __name__ == "__main__":
    # add time taken
    start_time = datetime.now()

    # Setup logging
    logger = logging.getLogger()
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        handlers=[logging.StreamHandler(sys.stdout)],
    )

    logger.info("Execution started.")

    if epic_id == "":
        logger.error("Epic ID is empty. Exiting cleanly...")
        sys.exit(0)

    client = IGAPIClient(epic_id)

    RESOLUTIONS = ["MINUTE_10"] # ["MINUTE", "MINUTE_10", "HOUR", "DAY"]
    RESOLUTION_POINTS = [24]

    try:
        fraction_strings = []
        for resolution, points in zip(RESOLUTIONS, RESOLUTION_POINTS, strict=False):
            fraction_string = f"{resolution}/{points}"
            fraction_strings.append(fraction_string)

        client.get_backtest_data(fraction_strings)
    except Exception as e:
        logger.error("Failed to get prices: %s", e)
        sys.exit(0)

    # read backtest_prices.csv
    try:
        data = pd.read_csv("backtest_prices.csv")
    except FileNotFoundError:
        logger.error("File not found: backtest_prices.csv")
        sys.exit(0)
    except Exception as e:
        logger.error("Failed to read from file: %s", e)
        sys.exit(0)

    # read in the data from the CSV file to a DataFrame
    data = pd.read_csv("backtest_prices.csv")

    # calculate the market condition
    market_condition, trend_strength = assess_market_trend_with_codes(data)

    if market_condition == MARKET_CONDITION_TOO_SMALL:
        logger.error("Not enough data points to assess market condition.")
    elif market_condition == MARKET_FLAT_DIRECTIONLESS:
        logger.info("Market is flat and directionless.")
    elif market_condition == MARKET_TREND_UP_STRONG:
        logger.info("Market is trending up strongly.")
    elif market_condition == MARKET_TREND_UP_WEAK:
        logger.info("Market is trending up weakly.")
    elif market_condition == MARKET_TREND_DOWN_STRONG:
        logger.info("Market is trending down strongly.")
    elif market_condition == MARKET_TREND_DOWN_WEAK:
        logger.info("Market is trending down weakly.")
    else:
        logger.error("Invalid market condition code.")

    logger.info("Execution completed.")
    end_time = datetime.now()
    time_taken = end_time - start_time
    logger.info(f"Time taken: {time_taken}")
    # everything went fine!
    sys.exit(0)
