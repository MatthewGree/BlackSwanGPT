from argparse import ArgumentParser
from calendar import monthrange
from collections.abc import Generator
from datetime import date
import time
from typing import Any
from news import get_news
from gpt import get_company_sentiment
from gpt import cache as gpt_cache
from news import cache as news_cache
from pandas import DataFrame
from tqdm import tqdm
from fastapi import FastAPI
import uvicorn


app = FastAPI()


@app.get("/signal/latest/{tokenPair}")
async def get_latest_signal(tokenPair: str):
    today = date.today()
    news = get_news(tokenPair, today)
    sentiment = get_company_sentiment(tokenPair, news, today)
    match sentiment:
        case "positive":
            result = 1
        case "negative":
            result = -1
        case _:
            result = 0
    return {"tokenPair": tokenPair, "action": result, "timestamp": time.time()}


if __name__ == "__main__":
    config = uvicorn.Config("main:app", port=3005, log_level="info")
    server = uvicorn.Server(config)
    server.run()
