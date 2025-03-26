import biomedica_index as bi
from PIL import Image
import os
import time

print("WARNING: THIS SCRIPT CURRENTLY REQUIRES >150GB RAM TO RUN")

if 'BIOMEDICA_INDEX_PATH' in os.environ:
    INDEX_PATH = os.environ['BIOMEDICA_INDEX_PATH']
else:
    INDEX_PATH = input("Please enter your local path to the index > ")

rib_xray = Image.open('assets/rib_xray.jpg')
pseudovirus = Image.open('assets/pseudovirus_infection.jpg')
load_start = time.time()
index = bi.BiomedicaIndex(INDEX_PATH)
# index.query_figures(text="dummy initialization query", image=rib_xray, top_k=1, subsets='all')
index.query_figures(text="dummy initialization query", top_k=1, subsets='all')
# index.query_figures(image=rib_xray, top_k=1, subsets='all')
load_end = time.time()
print(f"INITIAL LOAD/INITIALIZATION TIME: {load_end - load_start:.2f}s")
prompts = [
    ('an x-ray of a rib', rib_xray),
    ('pseudovirus infectivity of SARS-CoV-S and VSV-G', pseudovirus)
]
for text_query, image_query in prompts:
    print('='*20)
    query_start = time.time()
    text_results = index.query_figures(text=text_query, top_k=5, subsets='all')
    query_end = time.time()
    print("< TEXT RESULTS >")
    for _, meta in text_results:
        print(meta)
    print(f"TEXT QUERY RUNTIME (hybrid): {query_end - query_start:.2f}s")
    query_start = time.time()
    image_results = index.query_figures(image=image_query, top_k=5, subsets='all')
    query_end = time.time()
    print("< IMAGE RESULTS >")
    for _, meta in image_results:
        print(meta)
    print(f"IMAGE QUERY RUNTIME: {query_end - query_start:.2f}s")
    print()
