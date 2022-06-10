from pprint import pprint
import requests
from bs4 import BeautifulSoup
from newspaper import Article
from tqdm import tqdm
import pandas as pandas
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import csv


def crawler():

    data = []
    url_list = []

    page = 77
    while True:
        page = page + 1

        page_url = f"https://www.eghtesadnews.com/%D8%A8%D8%AE%D8%B4-%D8%A7%D8%AE%D8%A8%D8%A7%D8%B1-%D8%B3%DB%8C%D8%A7%D8%B3%DB%8C-57?curp=1&categories=57&order=order_time&page={page}"

        html = requests.get(page_url).text

        soup = BeautifulSoup(html, features='lxml')

        # get all titles from page
        titles = soup.find_all('div', class_='title')

        if len(url_list) and f"https://www.eghtesadnews.com{titles[0].a['href']}" in url_list:
            break

        for link in tqdm(titles):
            article_url = f"https://www.eghtesadnews.com{link.a['href']}"

            url_list.append(article_url)

            article = Article(article_url)
            try:
                article.download()
                article.parse()
                data.append({
                    'page': page,
                    'url': article_url,
                    'title': article.title,
                    'text': article.text,
                })
            except:
                print(f"failed on page: {page}")

    file = pandas.DataFrame(data)
    file.to_csv(f'./export.csv')


# crawler()


csv_file = open('./export.csv', encoding='utf-8')
reader = csv.reader(csv_file, delimiter=',')

docs = []

for row in reader:
    docs.append(row[3])

vectorizer = TfidfVectorizer(lowercase=True)
vectorizer.fit(docs)

docs_tfidf = vectorizer.transform(docs)

query = 'ابراهیم رئیسی'

query_tfidf = vectorizer.transform([query])[0]

cosines = []
for d in docs_tfidf:
    cosines.append(float(cosine_similarity(d, query_tfidf)))

sorted_ids = np.argsort(cosines)

for i in range(2):
    cur_id = sorted_ids[-i-1]
    print(docs[cur_id], cosines[cur_id])
