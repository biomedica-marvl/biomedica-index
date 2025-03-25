import biomedica_index as bi
import os
import time

load_start = time.time()
if 'BIOMEDICA_INDEX_PATH' in os.environ:
    INDEX_PATH = os.environ['BIOMEDICA_INDEX_PATH']
else:
    INDEX_PATH = input("Please enter your local path to the index > ")
index = bi.BiomedicaIndex(INDEX_PATH)
loader = bi.BiomedicaArticleLoader(INDEX_PATH)
index.query_articles(text="dummy initialization/loading query", top_k=1, subsets='all')
load_end = time.time()
print(f"TOTAL LOAD/INITIALIZATION TIME: {load_end - load_start:.2f}s")
while (text_query := input('QUERY ["q" to quit] > ')) != 'q':
    query_start = time.time()
    top_articles = index.query_articles(text=text_query, top_k=5, subsets='all')
    query_end = time.time()
    print(f"TOTAL QUERY RUNTIME: {query_end - query_start:.2f}s")
    article_output = ""
    for article_metadata, score in top_articles:
        title, full_text, date_str = loader.get_article(article_metadata)
        article_output += f"{'='*20}\nARTICLE TITLE: {title}, DATE_STR={date_str}, SCORE={score:.4f}"
        article_output += f", SUBSET={article_metadata['subset']}"
        article_output += f"\n\nSNIPPET: {full_text[1000:2000]}\n{'='*20}\n"
    retrieve_end = time.time()
    print(article_output)
    print(f"TOTAL RETRIEVE RUNTIME: {retrieve_end - query_end:.2f}s")