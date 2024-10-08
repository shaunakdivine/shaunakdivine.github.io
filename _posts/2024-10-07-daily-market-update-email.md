---
layout: default
title: Daily Market Update Email
date:   2024-10-07 11:21:21 -0500
categories: market research
permalink: /projects/project3/

---

# Automating Market Updates with Python, Yahoo Finance API, and Windows Task Scheduler

In this project, I created a Python script that fetches market data and sends it to me via email. I wanted to be able to wake up with all the quick market information that is important to me in one easy to find place. To automate this process, I used **Windows Task Scheduler** to ensure the script runs automatically every weekday morning at 8:30 AM, without requiring manual intervention. It provides me with information on the Dow Jones Industrial Average, S&P 500 Index, NASDAQ Composite Index, Volatility Index (VIX), Gold Futures, WTI Crude Oil Futures, Bitcoin, and the 10-Year Treasury Yield. This was a quick but interesting build as it allowed me to get more in depth with Windows Task Scheduler and Python Email libraries for the first time. It is also a very useful project as the returned information can be easily changed and customized as my portfolio interests adjust. 

## The Python Script

The script uses the `requests` library to fetch financial data from Yahoo Finance and the `smtplib` library to send email updates via Gmail. It pulls data for multiple symbols and emails the results every weekday morning.

### The Code

```python
import requests
from datetime import datetime
import os
import smtplib
from email.mime.text import MIMEText

API_KEY = os.getenv('RAPIDAPI_KEY')

def get_yahoo_finance_data(symbol):
    url = f"https://apidojo-yahoo-finance-v1.p.rapidapi.com/market/v2/get-quotes?symbols={symbol}&region=US"
    headers = {
        "X-RapidAPI-Key": API_KEY,
        "X-RapidAPI-Host": "apidojo-yahoo-finance-v1.p.rapidapi.com"
    }
    response = requests.get(url, headers=headers)
    data = response.json()
    
    if "quoteResponse" in data and len(data["quoteResponse"]["result"]) > 0:
        return data["quoteResponse"]["result"][0]
    else:
        return None

def print_yahoo_finance_data(symbol, data):
    if data:
        return (f"{symbol} Data:\n"
                f"Price: ${data['regularMarketPrice']}\n"
                f"Previous Close: ${data['regularMarketPreviousClose']}\n"
                f"Change: {data['regularMarketChange']} ({data['regularMarketChangePercent']}%)\n"
                f"Volume: {data['regularMarketVolume']}\n\n")
    else:
        return f"No data found for {symbol}.\n\n"

def fetch_and_print_data():
    result = f"Good morning, market update time! Fetching data at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} CT\n\n"
    
    symbols = ['^DJI', '^GSPC', '^IXIC', '^VIX', 'GC=F', 'CL=F', 'BTC-USD', '^TNX']
    result += """
    Symbol Definitions:
    -------------------
    Dow Jones Industrial Average:  ^DJI
    S&P 500 Index:                 ^GSPC
    NASDAQ Composite Index:        ^IXIC
    Volatility Index (VIX):        ^VIX
    Gold Futures:                  GC=F
    WTI Crude Oil Futures:         CL=F
    Bitcoin (BTC) to USD:          BTC-USD
    10-Year Treasury Yield:        ^TNX
    -------------------
    \n\n
    """
    
    for symbol in symbols:
        data = get_yahoo_finance_data(symbol)
        result += print_yahoo_finance_data(symbol, data)
    
    return result

def send_email(body):
    sender_email = "shaunakmarketupdate@gmail.com"
    receiver_email = ["shaunak.divine@gmail.com"]
    password = #insert own password

    recipients_str = ", ".join(receiver_email)

    
    msg = MIMEText(body)
    msg['Subject'] = "Daily Market Update"
    msg['From'] = sender_email
    msg['To'] = recipients_str

    
    try:
        with smtplib.SMTP_SSL('smtp.gmail.com', 465) as server:
            server.login(sender_email, password)
            server.sendmail(sender_email, receiver_email, msg.as_string())
        print("Email sent successfully!")
    except Exception as e:
        print(f"Error sending email: {e}")

if __name__ == "__main__":
    
    data = fetch_and_print_data()
    
    
    send_email(data)
```
## Sample Output

#### Fetching data at 2024-10-07 22:21:01 ET

#### ^DJI Data:
- **Price:** $41,954.24  
- **Previous Close:** $42,352.75  
- **Change:** -398.51172 (-0.9409347%)  
- **Volume:** 307,238,418  

#### ^GSPC Data:
- **Price:** $5,695.94  
- **Previous Close:** $5,751.07  
- **Change:** -55.129883 (-0.9586022%)  
- **Volume:** 2,384,804,000  

#### ^IXIC Data:
- **Price:** $17,923.904  
- **Previous Close:** $18,137.85  
- **Change:** -213.94531 (-1.1795517%)  
- **Volume:** 4,658,922,000  

#### ^VIX Data:
- **Price:** $22.64  
- **Previous Close:** $19.21  
- **Change:** 3.4300003 (17.855286%)  
- **Volume:** 0  

#### GC=F Data:
- **Price:** $2,658.8  
- **Previous Close:** $2,666.0  
- **Change:** -7.199951 (-0.2700657%)  
- **Volume:** 23,940  

#### CL=F Data:
- **Price:** $75.83  
- **Previous Close:** $77.14  
- **Change:** -1.3099976 (-1.6982079%)  
- **Volume:** 44,513  

#### BTC-USD Data:
- **Price:** $62,565.375  
- **Previous Close:** $62,227.664  
- **Change:** -1,107.9453 (-1.7400451%)  
- **Volume:** 32,866,029,568  

#### ^TNX Data:
- **Price:** 4.026  
- **Previous Close:** 3.9810002  
- **Change:** 0.044999838 (1.1303651%)  
- **Volume:** 0  
