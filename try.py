import pandas as pd
import requests
from bs4 import BeautifulSoup

# Dataframe to store the data
df = pd.DataFrame()

# Scraper
# Example : https://coinmarketcap.com/historical/20211128/
def scrape(date='20211128/', number=10):
    # StoringInfo variables
    name, marketCap, price, circulatingSupply, symbol = [], [], [], [], []
    # URL to scrape
    url = 'https://coinmarketcap.com/historical/'+date
    # Request a website
    webpage = requests.get(url)
    # parse the text
    soup = BeautifulSoup(webpage.text, 'html.parser')
    
    # Get table row element
    tr = soup.find_all('tr', attrs={'class':'cmc-table-row'})
    
    count = 0
    
    for row in tr:
        if count == number:
            break
        else:
            count += 1
            
            # Store name of the crypto currency            
            name_col = row.find('td', attrs={'class':'cmc-table__cell cmc-table__cell--sticky cmc-table__cell--sortable cmc-table__cell--left cmc-table__cell--sort-by__name'})
            cryptoname = name_col.find('a', attrs={'class':'cmc-table__column-name--name cmc-link'}).text.strip()
            
            # Market cap
            marketcap = row.find('td', attrs={'class':'cmc-table__cell cmc-table__cell--sortable cmc-table__cell--right cmc-table__cell--sort-by__market-cap'}).text.strip()
            
            # Price
            crytoprice = row.find('td', attrs={'class':'cmc-table__cell cmc-table__cell--sortable cmc-table__cell--right cmc-table__cell--sort-by__price'}).text.strip()
            
            # Circulating supply and symbol            
            circulatingSupplySymbol = row.find('td', attrs={'class':'cmc-table__cell cmc-table__cell--sortable cmc-table__cell--right cmc-table__cell--sort-by__circulating-supply'}).text.strip()
            supply = circulatingSupplySymbol.split(' ')[0]
            sym = circulatingSupplySymbol.split(' ')[1]
            # append the data
            name.append(cryptoname)
            marketCap.append(marketcap)
            price.append(crytoprice)
            circulatingSupply.append(supply)
            symbol.append(sym)  
    return name, marketCap, price, circulatingSupply, symbol

# Scraping
name, marketCap, price, circulatingSupply, symbol = scrape('20231205/', number=20)

for i in [name, marketCap, price, circulatingSupply, symbol]:
    print(i,'\n')
    
df['Name'] = name
df['Market Cap'] = marketCap
df['Price'] = price
df['Circulating Suppy'] = circulatingSupply
df['Symbol'] = symbol
print(df)


"""
def get_cryptocurrency_data(coin_id):
    url = f"https://api.coingecko.com/api/v3/coins/{coin_id}"
    response = requests.get(url)
    if response.status_code == 200:
        data = response.json()
        name = data['name']
        symbol = data['symbol']
        current_price = data['market_data']['current_price']['usd']
        market_cap = data['market_data']['market_cap']['usd']
        print(f"Name: {name}\nSymbol: {symbol}\nCurrent Price: ${current_price}\nMarket Cap: ${market_cap}")
    else:
        print("Failed to retrieve data")

# Example usage
get_cryptocurrency_data('bitcoin')
"""

"""
import time

def get_cryptocurrency_data(coin_ids):
    base_url = "https://api.coingecko.com/api/v3/coins/"
    for coin_id in coin_ids:
        try:
            response = requests.get(base_url + coin_id)
            if response.status_code == 200:
                data = response.json()
                name = data['name']
                symbol = data['symbol']
                current_price = data['market_data']['current_price']['usd']
                market_cap = data['market_data']['market_cap']['usd']
                print(f"Name: {name}\nSymbol: {symbol}\nCurrent Price: ${current_price}\nMarket Cap: ${market_cap}\n")
            else:
                print(f"Failed to retrieve data for {coin_id}. Status code: {response.status_code}")
        except requests.RequestException as e:
            print(f"Request failed: {e}")

        time.sleep(1)

# Example usage
coin_ids = ['bitcoin', 'ethereum', 'dogecoin', 'cardano', 'ripple']
get_cryptocurrency_data(coin_ids)
"""

"""
import random
from itertools import cycle

import requests

# Example list of proxies
proxies = ["103.155.217.1:41317", "47.91.56.120:8080", "103.141.143.102:41516", "167.114.96.13:9300", "103.83.232.122:80"]

proxy_pool = cycle(proxies)

def get_cryptocurrency_data(coin_id):
    proxy = next(proxy_pool)
    try:
        response = requests.get(f"https://api.coingecko.com/api/v3/coins/{coin_id}", proxies={"http": proxy, "https": proxy})
        if response.status_code == 200:
            # Process response here
            pass
        else:
            print(f"Failed to retrieve data for {coin_id}. Status code: {response.status_code}")
    except requests.RequestException as e:
        print(f"Request failed: {e}")

# Example usage
get_cryptocurrency_data('bitcoin')

# https://github.com/jhao104/proxy_pool
# https://github.com/topics/proxy-pool
# https://github.com/imWildCat/scylla
# https://github.com/lawrence-peng/proxy-pool
# crypto web scraping python
"""
