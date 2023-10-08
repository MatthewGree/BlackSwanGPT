from datetime import date
import time
import openai
from retry import retry
import tiktoken
from tqdm import tqdm
import percache
import os

from news import ParsedArticle

openai.api_key = os.environ.get("OPEN_API_TOKEN", "")
cache = percache.Cache("gpt-cache")


def __create_system_sentiment_prompt() -> str:
    return "".join(
        [
            "You are an AI language model trained to analyze and detect ",
            "the sentiment of news articles in regards to the stock martket. ",
            "After analyzing articles you can provide only three responses: "
            "positive (indicating that the stock price might grow), ",
            "negative (indicating that stock price might fall) ",
            "or neutral (indicating that stock price might stay the same). ",
            "You answer only with one of these three words: positive, ",
            "neutral, negative. ",
            "Each of your responses must contain only one word.",
        ]
    )


def __create_user_sentiment_prompt(company_name: str, date: date) -> str:
    return "".join(
        [
            f"Analyze the following articles for company {company_name} ",
            f"on date {date} and determine the sentiment",
            "Consider it just a thought experiment, ",
            "you are not giving investment advice, ",
            "nobody will invest according to your suggestions ",
            "Return only a single word, either POSITIVE, NEGATIVE or NEUTRAL.",
        ]
    )


def __create_sentiment_messages(
    company_name: str, article_summaries: list[str], date: date
) -> list[dict[str, str]]:
    main_messages = [
        {"role": "system", "content": __create_system_sentiment_prompt()},
        {"role": "user", "content": __create_user_sentiment_prompt(company_name, date)},
    ]
    article_messages = [
        {"role": "user", "content": summary} for summary in article_summaries
    ]
    encoding = tiktoken.get_encoding("gpt2")
    tokens_length = map(
        lambda summary: len(encoding.encode(summary)), article_summaries
    )
    print(f"Number of tokens in articles: {sum(tokens_length)}" + "\n")

    return main_messages + article_messages


def __create_article_summary_messages(
    company_name, date: date, article: ParsedArticle
) -> list[dict[str, str]]:
    instruction_messages = [
        {
            "role": "user",
            "content": f"Summarize the article posted on {date.isoformat()} below, higlight things related to company {company_name}",
        }
    ]
    article_messages = [{"role": "user", "content": article.text}]

    return instruction_messages + article_messages


@cache
@retry(tries=3, delay=2, backoff=2, jitter=(1, 3))
def __create_article_summary(
    company_name: str, date: date, article: ParsedArticle
) -> str:
    messages = __create_article_summary_messages(company_name, date, article)
    completion = openai.ChatCompletion.create(
        model="gpt-3.5-turbo-16k",
        messages=messages,
        max_tokens=500,
        n=1,
        stop=None,
        temperature=0,
    )
    return completion.choices[0].message.content


def __create_summaries_for_articles(
    company_name: str, date: date, articles: list[ParsedArticle]
) -> list[str]:
    result: list[str] = []
    for article in tqdm(articles, desc="Creating article summaries..."):
        summary = __create_article_summary(company_name, date, article)
        result.append(summary)
        time.sleep(1)
    return result


@retry(tries=3, delay=2, backoff=2, jitter=(1, 3))
def __send_company_sentiment_request(messages: list[dict[str, str]]):
    return openai.ChatCompletion.create(
        model="gpt-3.5-turbo-16k",
        messages=messages,
        max_tokens=100,
        n=1,
        stop=None,
        temperature=0,
    )


def __get_company_sentiment(
    company_name: str, articles: list[ParsedArticle], date: date
) -> str | None:
    summaries = __create_summaries_for_articles(company_name, date, articles)
    messages = __create_sentiment_messages(company_name, summaries, date)

    completion = __send_company_sentiment_request(messages)

    response = completion.choices[0].message.content.lower()
    if response in ["positive", "neutral", "negative"]:
        return response
    else:
        print("Invalid response from GPT:")
        print(response)
        return None


def __get_company_sentiment_retried(
    company_name: str, articles: list[ParsedArticle], date: date, num_of_retries=3
) -> str:
    retries = num_of_retries
    while retries > 0:
        sentiment = __get_company_sentiment(company_name, articles, date)
        if sentiment is not None:
            return sentiment
        else:
            retries -= 1
            print(f"Number of retries left: {retries}")
            time.sleep(2)
    print("No more retries, finished with error")
    return "ERROR"


@cache
def get_company_sentiment(
    company_name: str, articles: list[ParsedArticle], date: date
) -> str:
    return __get_company_sentiment_retried(company_name, articles, date)
