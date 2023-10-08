from datetime import date, timedelta
import time
from typing import Any
from requests import Response
import requests
from retry import retry

from gnews import GNews
from newspaper import Article
from tqdm import tqdm
import percache


cache = percache.Cache("news-cache")


class ParsedArticle:
    def __init__(self, url: str, title: str, text: str):
        self.url = url
        self.title = title
        self.text = text

    def __repr__(self):
        return f"url:{self.url},title={self.title}"

    @classmethod
    def from_article(cls, article: Article):
        return cls(article.url, article.title, article.text)

    @classmethod
    def from_dict(cls, dict: dict[str, str]):
        return cls(dict["url"], dict["title"], dict["text"])

    def to_json(self):
        return self.__dict__

    def to_dict(self):
        return self.__dict__


@retry(tries=5, delay=1, backoff=2)
def __get_full_article(url: str) -> ParsedArticle | None:
    article = GNews().get_full_article(url=url)
    if article is not None:
        return ParsedArticle.from_article(article)
    else:
        return None


@retry(tries=5, delay=2, backoff=3)
def __article_without_consent(article: dict[str, Any]) -> Response | None:
    try:
        return requests.get(article["url"])
    except requests.ConnectionError:
        print("Connection error, returning None")
        return None


@cache
def get_news(company_name: str, date: date) -> list[ParsedArticle]:
    nextDay = date + timedelta(days=1)
    news_getter = GNews(
        language="en",
        max_results=10,
        start_date=(date.year, date.month, date.day),
        end_date=(nextDay.year, nextDay.month, nextDay.day),
    )
    news_articles = news_getter.get_news(company_name)
    if news_articles is None:
        return []
    without_consent_with_none = map(
        lambda article: __article_without_consent(article), news_articles
    )
    without_consent = [item for item in without_consent_with_none if item is not None]
    urls = map(lambda article: article.url, without_consent)
    full_articles: list[ParsedArticle] = []
    for url in tqdm(list(urls), desc="Downloading and parsing articles..."):
        full_article = __get_full_article(url)
        if full_article is not None:
            full_articles.append(full_article)
        time.sleep(1)
    return full_articles
