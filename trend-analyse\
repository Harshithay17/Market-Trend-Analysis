import yfinance as yf
import pandas as pd
from transformers import pipeline
import torch

# Use CPU or GPU
device = 0 if torch.cuda.is_available() else -1

# Load sentiment analysis pipeline
sentiment_analyzer = pipeline("sentiment-analysis", device=device)

def get_last_10_days_trend_with_sentiment_analysis(company_name_or_ticker):
    # Fetch the stock ticker
    ticker = company_name_or_ticker.upper()

    print(f"\n📊 Getting the last 10 days' trend for {ticker}...\n")

    # Get company name from Yahoo Finance info
    stock_info = yf.Ticker(ticker).info
    company_name = stock_info.get("longName", ticker)  # Default to ticker if company name not found

    df = yf.download(ticker, period="14d", interval="1d", auto_adjust=True)

    if df.empty:
        print("❌ No data found.")
        return

    df = df[['Close']].dropna()
    df.rename(columns={'Close': 'Price'}, inplace=True)
    df['Change (%)'] = df['Price'].pct_change() * 100
    df['Direction'] = df['Change (%)'].apply(lambda x: 'UP' if x > 0 else ('DOWN' if x < 0 else 'NO CHANGE'))
    df = df.dropna().tail(10)

    print(f"🏢 Company: {company_name}")
    print("📅 Last 10 Days Trend:\n")

    trend_texts = []
    sentiment_scores = []
    for date, row in df.iterrows():
        date_str = date.strftime('%Y-%m-%d')
        price = round(float(row['Price']), 2)
        change_val = round(float(row['Change (%)']), 2)
        direction = row['Direction']
        sign = "+" if change_val >= 0 else ""
        trend_line = f"{date_str}: ${price:.2f} ({sign}{change_val:.2f}%) [{direction}]"
        trend_texts.append(trend_line)

        # Perform sentiment analysis on each price change direction
        sentiment = sentiment_analyzer(trend_line)
        sentiment_scores.append(sentiment[0]['label'])

    for trend_line, sentiment in zip(trend_texts, sentiment_scores):
        print(f"{trend_line} -> Sentiment: {sentiment}")

    # Analyze overall trend sentiment
    positive_sentiment = sentiment_scores.count('POSITIVE')
    negative_sentiment = sentiment_scores.count('NEGATIVE')

    overall_trend = "Neutral"
    if positive_sentiment > negative_sentiment:
        overall_trend = "Bullish"
    elif negative_sentiment > positive_sentiment:
        overall_trend = "Bearish"

    print(f"\n📈 Overall Trend for {company_name}: {overall_trend}")

# ✅ Example Usage
company_name_or_ticker = input("Enter company name or stock ticker: ")
get_last_10_days_trend_with_sentiment_analysis(company_name_or_ticker)
