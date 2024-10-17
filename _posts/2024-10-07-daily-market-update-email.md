---
layout: default
title: Daily Market Update Email
date:   2024-10-07 11:21:21 -0500
categories: market research
permalink: /projects/project3/

---

# Automating Market Updates with Python, Yahoo Finance API, and Windows Task Scheduler

In this project, I created a Python script that not only fetches key market data but also aggregates top financial news headlines from **CNBC** and **The Wall Street Journal (WSJ)**, sending it all to my friends and me via email. I wanted to automate the process of receiving the latest market data and top financial headlines in one place, ensuring that I start each weekday morning with relevant and up-to-date information.

The script fetches market data for the **Dow Jones Industrial Average**, **S&P 500 Index**, **NASDAQ Composite Index**, **Volatility Index (VIX)**, **Gold Futures**, **WTI Crude Oil Futures**, **Bitcoin (BTC to USD)**, and the **10-Year Treasury Yield**. In addition to market data, it scrapes the top six financial headlines from CNBC and WSJ via their RSS feeds, providing a comprehensive overview of the market and major news events.

To automate the process, I used **Windows Task Scheduler** to run the script every weekday morning at 8:30 AM. This ensures that the market update and financial news are delivered to my inbox without requiring any manual intervention.

This project was a fun and useful build as it allowed me to get hands-on experience with Python's **email automation**, **RSS feed parsing**, and **Windows Task Scheduler**. The flexibility of the script means that it can be easily customized to fetch additional market symbols or financial headlines as my portfolio and interests evolve.

## The Python Script

The script uses the following key libraries:
- **`requests`** to fetch financial data from Yahoo Finance.
- **`feedparser`** to retrieve the top financial news headlines from CNBC and WSJ RSS feeds.
- **`smtplib`** to send email updates via Gmail.
  
It pulls market data for multiple symbols and headlines from leading financial news sources, formatting everything into a well-structured email that's sent every weekday morning.

## The Code


```python
import requests
import feedparser
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

def format_finance_data(symbol, data):
    if data:
        return (f"{symbol} Data:\n"
                f"  Price: ${data['regularMarketPrice']:,.2f}\n"
                f"  Previous Close: ${data['regularMarketPreviousClose']:,.2f}\n"
                f"  Change: {data['regularMarketChange']:.2f} ({data['regularMarketChangePercent']:.2f}%)\n"
                f"  Volume: {data['regularMarketVolume']:,}\n")
    else:
        return f"{symbol} Data: No data available.\n"


        

def fetch_and_format_data():
    timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    result = f"Good afternoon, market update time! Fetching data at {timestamp} CT\n\n"
    
    symbols = ['^DJI', '^GSPC', '^IXIC', '^VIX', 'GC=F', 'CL=F', 'BTC-USD', '^TNX']
   
    result += (
        "****************************************\n"
        "*            Market Overview            *\n"
        "****************************************\n"
        "Symbol Definitions:\n"
        "-------------------\n"
        "  Dow Jones Industrial Average:  ^DJI\n"
        "  S&P 500 Index:                 ^GSPC\n"
        "  NASDAQ Composite Index:        ^IXIC\n"
        "  Volatility Index (VIX):        ^VIX\n"
        "  Gold Futures:                  GC=F\n"
        "  WTI Crude Oil Futures:         CL=F\n"
        "  Bitcoin (BTC) to USD:          BTC-USD\n"
        "  10-Year Treasury Yield:        ^TNX\n"
        "-------------------\n\n"
    )
    
    for symbol in symbols:
        data = get_yahoo_finance_data(symbol)
        result += format_finance_data(symbol, data) + "\n"
    
    return result

def get_financial_news_from_rss():
    rss_feed_url = "https://www.cnbc.com/id/100003114/device/rss/rss.html"  
    feed = feedparser.parse(rss_feed_url)

    if 'entries' in feed and len(feed.entries) > 0:
        headlines = "\n".join([f"- {entry.title}" for entry in feed.entries[:6]])  
        return (
            "****************************************\n"
            "*    Top Financial Headlines from CNBC   *\n"
            "****************************************\n"
            f"{headlines}\n\n\n"
        )
    else:
        return (
            "****************************************\n"
            "*    Top Financial Headlines from CNBC   *\n"
            "****************************************\n"
            "\nNo financial news available at the moment.\n"
        )
    

def get_wsj_headlines():
    wsj_rss_url = "https://feeds.a.dj.com/rss/RSSMarketsMain.xml"  
    feed = feedparser.parse(wsj_rss_url)

    if 'entries' in feed and len(feed.entries) > 0:
        headlines = "\n".join([f"- {entry.title}" for entry in feed.entries[:6]])  
        return (
            "****************************************\n"
            "*    Top Financial Headlines from WSJ    *\n"
            "****************************************\n"
            f"{headlines}\n\n\n"
        )
    else:
        return (
            "****************************************\n"
            "*    Top Financial Headlines from WSJ    *\n"
            "****************************************\n"
            "\nNo financial news available at the moment.\n"
        )

def send_email(body):
    
    sender_email = "shaunakmarketupdate@gmail.com"
    receiver_email = [
        #input emails
    ]
    

    password = #input password

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
    data = fetch_and_format_data()
    cnbc_news = get_financial_news_from_rss()
    wsj_news = get_wsj_headlines()
    
    full_body = data + cnbc_news + wsj_news
    
    send_email(full_body)

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

#### Top Financial Headlines from CNBC:
- The solid-state batteries hype is fading – prompting auto giants to consider alternatives
- Stanley Druckenmiller says he's 'licking my wounds' from selling Nvidia too soon
- Key change coming for 401(k) ‘max savers’ in 2025, expert says — here's what you need to know
- Amazon goes nuclear, to invest more than $500 million to develop small modular reactors
- Lucid shares tumble following public offering of nearly 262.5 million shares
- Morgan Stanley shares pop 7% after beating estimates for third-quarter profit and revenue

#### Top Financial Headlines from WSJ:
- The WSJ Dollar Index Rises 0.3% to 98.22
- Dow Edges Up; Morgan Stanley Gains on Strong Earnings
- Storms Be Damned, Florida Keeps Building in High-Risk Areas
- Winter Demand Concerns Weigh Down U.S. Natural Gas
- Gold Sets Fresh High as Consumers Continue to Buy
- A Librarian Hopes to Retire at 55 and Travel. Can She Afford It?