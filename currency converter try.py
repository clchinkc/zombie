
import re
from datetime import datetime, timedelta
from pprint import pprint

import requests
import yahoo_fin.stock_info as si
from bs4 import BeautifulSoup as bs
from dateutil.parser import parse


def get_exchange_list_xrates(currency, amount=1):
    # make the request to x-rates.com to get current exchange rates for common currencies
    content = requests.get(f"https://www.x-rates.com/table/?from={currency}&amount={amount}").content
    # initialize beautifulsoup
    soup = bs(content, "html.parser")
    # get the last updated time
    price_datetime = parse(soup.find_all("span", attrs={"class": "ratesTimestamp"})[1].text)
    # get the exchange rates tables
    exchange_tables = soup.find_all("table")
    exchange_rates = {}
    for exchange_table in exchange_tables:
        for tr in exchange_table.find_all("tr"):
            # for each row in the table
            tds = tr.find_all("td")
            if tds:
                currency = tds[0].text
                # get the exchange rate
                exchange_rate = float(tds[1].text)
                exchange_rates[currency] = exchange_rate        
    return price_datetime, exchange_rates

def x_rates():
    source_currency = "EUR"
    amount = 1000
    target_currency = "GBP"
    price_datetime, exchange_rates = get_exchange_list_xrates(source_currency, amount)
    print("Last updated:", price_datetime)
    pprint(exchange_rates)
    



def convert_currency_xe(src, dst, amount):
    def get_digits(text):
        """Returns the digits and dots only from an input `text` as a float
        Args:
            text (str): Target text to parse
        """
        new_text = ""
        for c in text:
            if c.isdigit() or c == ".":
                new_text += c
        return float(new_text)

    url = f"https://www.xe.com/currencyconverter/convert/?Amount={amount}&From={src}&To={dst}"
    content = requests.get(url).content
    soup = bs(content, "html.parser")
    exchange_rate_html = soup.find_all("p")[2]
    last_updated_div = soup.find("div", class_="result__LiveSubText-sc-1bsijpp-2 iKYXwX")
    last_updated_text = re.search(r"Last updated (.+)", last_updated_div.text).group(1)
    return last_updated_text, get_digits(exchange_rate_html.text)

def xe():
    source_currency = "EUR"
    destination_currency = "USD"
    amount = 1000
    last_updated_text, exchange_rate = convert_currency_xe(source_currency, destination_currency, amount)
    print("Last updated:", last_updated_text)
    print(f"{amount} {source_currency} = {exchange_rate} {destination_currency}")




def convert_currency_yahoofin(src, dst, amount):
    # construct the currency pair symbol
    symbol = f"{src}{dst}=X"
    # extract minute data of the recent 2 days
    latest_data = si.get_data(symbol, interval="1m", start_date=datetime.now() - timedelta(days=2))
    # get the latest datetime
    last_updated_datetime = latest_data.index[-1].to_pydatetime()
    # get the latest price
    latest_price = latest_data.iloc[-1].close
    # return the latest datetime with the converted amount
    return last_updated_datetime, latest_price * amount

def yahoofin():
    source_currency = "EUR"
    destination_currency = "USD"
    amount = 1000
    last_updated_datetime, exchange_rate = convert_currency_yahoofin(source_currency, destination_currency, amount)
    print("Last updated datetime:", last_updated_datetime)
    print(f"{amount} {source_currency} = {exchange_rate} {destination_currency}")



def get_all_exchange_rates_erapi(src):
    url = f"https://open.er-api.com/v6/latest/{src}"
    # request the open ExchangeRate API and convert to Python dict using .json()
    data = requests.get(url).json()
    if data["result"] == "success":
        # request successful
        # get the last updated datetime
        last_updated_datetime = parse(data["time_last_update_utc"])
        # get the exchange rates
        exchange_rates = data["rates"]
    return last_updated_datetime, exchange_rates

def convert_currency_erapi(src, dst, amount):
    # get all the exchange rates
    last_updated_datetime, exchange_rates = get_all_exchange_rates_erapi(src)
    # convert by simply getting the target currency exchange rate and multiply by the amount
    return last_updated_datetime, exchange_rates[dst] * amount

if __name__ == "__main__":
    source_currency = "EUR"
    destination_currency = "USD"
    amount = 1000
    last_updated_datetime, exchange_rate = convert_currency_erapi(source_currency, destination_currency, amount)
    print("Last updated datetime:", last_updated_datetime)
    print(f"{amount} {source_currency} = {exchange_rate} {destination_currency}")
