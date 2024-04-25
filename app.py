import streamlit as st
import requests
from bs4 import BeautifulSoup
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
import nltk
from nltk.corpus import stopwords

# Ensure necessary NLTK downloads
nltk.download('stopwords')

# def fetch_news_articles(url):
#     response = requests.get(url)
#     soup = BeautifulSoup(response.text, 'html.parser')
#     articles = []
#     for article in soup.find_all("article"):  # Assuming articles are contained in <article> tags
#         title = article.find("h2").text if article.find("h2") else "No Title"
#         link = article.find("a")["href"]
#         summary = article.find("p").text if article.find("p") else "No Summary"
#         articles.append((title, link, summary))
#     return articles


import feedparser

# def fetch_news_articles(url):
#     feed = feedparser.parse(url)
#     articles = []
#     for entry in feed.entries:
#         title = entry.title
#         link = entry.link
#         summary = entry.summary
#         articles.append((title, link, summary))
#     return articles


def fetch_news_articles(url):
    feed = feedparser.parse(url)
    articles = []
    for entry in feed.entries:
        title = getattr(entry, 'title', 'No Title Available')
        link = getattr(entry, 'link', '#')

        # Handle variability in content descriptions
        content_attributes = ['summary', 'description', 'content']
        summary = 'No Summary Available'
        for attr in content_attributes:
            if hasattr(entry, attr):
                summary = getattr(entry, attr, 'No Summary Available')
                if isinstance(summary, list):  # In case it's a list of contents
                    summary = summary[0].value if summary else 'No Summary Available'
                break

        articles.append((title, link, summary))
    return articles



def preprocess_articles(articles):
    texts = [article[2] for article in articles]  # Using summaries for preprocessing
    vectorizer = TfidfVectorizer(stop_words=stopwords.words('english'))
    tfidf_matrix = vectorizer.fit_transform(texts)
    return tfidf_matrix

def cluster_articles(tfidf_matrix, num_clusters=5):
    km = KMeans(n_clusters=num_clusters)
    km.fit(tfidf_matrix)
    clusters = km.labels_.tolist()
    return clusters

def display_clusters(articles, clusters):
    clustered_articles = {i: [] for i in range(max(clusters) + 1)}
    for cluster, article in zip(clusters, articles):
        clustered_articles[cluster].append(article)

    st.title('Clustered News Articles')
    for cluster_id, articles in clustered_articles.items():
        st.subheader(f"Cluster {cluster_id}")
        for title, link, summary in articles:
            st.write(f"[{title}]({link}) - {summary}")

def main():
    st.sidebar.title('News Clustering')
    url = st.sidebar.text_input('Enter the URL of the news site:', 'https://example-news-website.com')
    if st.sidebar.button('Fetch and Cluster Articles'):
        articles = fetch_news_articles(url)
        if articles:
            tfidf_matrix = preprocess_articles(articles)
            clusters = cluster_articles(tfidf_matrix)
            display_clusters(articles, clusters)
        else:
            st.write("No articles found at the URL.")

if __name__ == "__main__":
    main()
