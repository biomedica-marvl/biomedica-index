INDEX_PATH_VAR = "BIOMEDICA_INDEX_PATH"
SUBSETS = ["commercial", "noncommercial", "other"]
BIOMEDICA_ROOT = "https://huggingface.co/datasets/BIOMEDICA/biomedica_webdataset_24M/resolve/main"
FILEKEY_MAP_DTYPES = {f"caption-kw": 'S164', f"full_text-kw": 'S11'}
ARTICLE_CORPUS_SIZES = {
    "commercial": 4_255_781,
    "noncommercial": 1_690_426,
    "other": 414_857
}
CAPTION_CORPUS_SIZES = {
    "other": 1_109_948,
    "noncommercial": 5_332_781,
    "commercial": 17_607_694
}

S3_BUCKET = "test.bmca-pmc-data"
S3_REGION = "us-west-2"