from newspaper import Article


def get_text_from_url(news_article_url):
    article = Article(news_article_url)
    article.download()
    article.parse()
    article.nlp()
    return article.text
