import os
import json
from collections import defaultdict

import boto3
import botocore.exceptions

from biomedica_index.clients import BM25Client, VectorClient
from biomedica_index.utils.embed_utils import SentenceEmbedder, ImageEmbedder
from biomedica_index.utils.consts import SUBSETS, S3_BUCKET, S3_REGION

class BiomedicaArticleLoader:
    def __init__(self, index_path, local_article_path=None):
        self.local_path = local_article_path
        default_path = f"{index_path}/ARTICLES"
        if (local_article_path is None) and os.path.exists(default_path):
            self.local_path = default_path
        self.article_maps = {}
        article_map_path = f"{index_path}/full_text-kw/full/pmcid_map.json"
        with open(article_map_path, 'r') as fh:
            self.article_maps = json.load(fh)
        if self.local_path is not None:
            self.s3_client = boto3.client('s3', region_name=S3_REGION)

    def get_article_s3(self, subset, pmcid):
        batch_fname = self.article_maps[subset][pmcid]
        try:
            response = self.s3_client.get_object(Bucket=S3_BUCKET, Key=batch_fname)
            data_str = response['Body'].read().decode()
        except botocore.exceptions.ClientError as e:
            raise ConnectionError("Reading article data from s3 failed")
        article_batch = json.loads(data_str)
        for article in article_batch:
            if article["accession_id"] == pmcid:
                return article['title'], article['nxml'], article['date']
        return None, None, None
        
    def get_article_local(self, subset, pmcid):
        batch_path = f"{self.local_path}/{self.article_maps[subset][pmcid]}"
        with open(batch_path, 'r') as article_batch_fh:
            article_batch = json.load(article_batch_fh)
        for article in article_batch:
            if article["accession_id"] == pmcid:
                return article['title'], article['nxml'], article['date']
        return None, None, None

    def get_article(self, article_metadata):
        if not all(key in article_metadata for key in ('subset', 'pmcid')):
            raise KeyError("Local loading requires `subset` and `pmcid`")
        get = self.get_article_s3 if self.local_path is None else self.get_article_local
        return get(article_metadata['subset'], article_metadata['pmcid'])


class BiomedicaRetriever:
    VEC_QUERY_TYPES = ['image', 'caption']
    KEYWORD_QUERY_TYPES = ['full_text-kw', 'caption-kw']
    QUERY_TYPES = VEC_QUERY_TYPES + KEYWORD_QUERY_TYPES

    def __init__(self, index_path, embedder_device=None, search_multiplier=1, RRF_k=60):
        self.index_path = index_path
        self.clients = {}
        # lazy-load the vector clients
        for q_type in self.VEC_QUERY_TYPES:
            self.clients[q_type] = VectorClient(f"{index_path}/{q_type}", SUBSETS)
        # load embedders only if needed
        self.embedders = {}
        self.device = embedder_device
        self.search_multiplier = search_multiplier
        self.k = RRF_k

    def rrf_rank(self, meta_lists):
        fkey_scores = defaultdict(float)
        fkey_to_md = {}
        for ranked_metadatas in meta_lists:
            for i, md in enumerate(ranked_metadatas):
                fkey = md['filekey']
                fkey_to_md[fkey] = md
                fkey_scores[fkey] += 1.0/(self.k + i + 1)
        best_fkeys = sorted(fkey_scores.keys(), key=lambda k: fkey_scores[k], 
                            reverse=True) # higher RRF score is better
        res = [(fkey_scores[k], fkey_to_md[k]) for k in best_fkeys]
        return res

    def query_figures_by_type(self, subsets, query_type, prompt, n_results=5):
        if query_type in self.VEC_QUERY_TYPES:
            emb = self.embedders[query_type]
            prompt = emb.encode(emb.preprocess([prompt]))
        elif (query_type in self.KEYWORD_QUERY_TYPES) and (query_type not in self.clients):
            self.clients[query_type] = BM25Client(self.index_path, query_type)
        results = [] # list of (score, metadata_dict) tuples
        # get best results for each subset
        for subset in subsets:
            res = self.clients[query_type].query([prompt], subset=subset, n_results=n_results)
            if query_type == 'caption-kw':
                fkeys, scores = res
                scores = -scores # make lower = better
                meta_dicts = self.clients['caption'].get_metadata(fkeys, subset)
            else:
                meta_dicts, scores = res
            for md in meta_dicts:
                md['subset'] = subset
            # combine across subsets
            results += list(zip(scores, meta_dicts))
        # keep only the best results overall
        results = sorted(results, key=lambda x: x[0]) # sort by lowest score
        # return the list of the best n_results metadata_dicts, best-first
        return [md for (score, md) in results[:n_results]]

    def query_subset_articles(self, subset, prompt, top_k=5, normalize=False):
        assert subset in SUBSETS
        files_found = {}
        article_scores = []
        pmcids, scores = self.clients['full_text-kw'].query(
            [prompt], n_results=top_k, subset=subset
        )
        if normalize:
            scores /= scores.sum()
        files_found.update({_id: {'pmcid': _id} for _id in pmcids})
        article_scores = list(zip(pmcids, scores.tolist()))
        # sort by highest score
        top_scores = sorted(article_scores, key=lambda x: x[1], reverse=True)[:top_k]
        top_results = [(files_found[pmcid], score) for (pmcid, score) in top_scores]
        return top_results

    def query_figures(self, image=None, text=None, text_mode='all', top_k=5, subsets='all'):
        '''
        image: PIL image to query for
        text: caption string to query for
        text_mode: one of the following strings:
            'keyword' = use keyword-based search only
            'vector' = use vector-based search only
            'all' = use both bm25 and vector-based search
        top_k: int number of items to return
        subsets: either the string 'all' (to use all subsets)
            or some subset of {'commercial','noncommercial','other'}
        '''
        assert text_mode in ['keyword', 'vector', 'all'],\
            f"{text_mode} is not a recognized text query type"
        if subsets == 'all':
             subsets = SUBSETS 
        queries = []
        if image is not None:
            queries.append(('image', image))
            if not "image" in self.embedders:
                self.embedders["image"] = ImageEmbedder(device=self.device)
        if text is not None:
            if text_mode in ('keyword', 'all'):
                queries.append(('caption-kw', text))
            if text_mode in ('vector', 'all'):
                queries.append(('caption', text))
                if not "caption" in self.embedders:
                    self.embedders["caption"] = SentenceEmbedder(device=self.device)
        meta_dicts_per_type = []
        # optionally, collect more than just top-k for each query type to improve RRF quality
        N = top_k*self.search_multiplier
        for q_type, prompt in queries:
            best_metadata = self.query_figures_by_type(subsets, q_type, prompt, n_results=N)
            meta_dicts_per_type.append(best_metadata)
        # sort by highest score
        final_scored_meta = self.rrf_rank(meta_dicts_per_type)
        # return list of (rrf_score, metadata) tuples
        return final_scored_meta

    def query_articles(self, text, top_k=5, subsets='all',
                       normalize=False, collapse_subsets=True, subset_boosts={}):
        if subsets == 'all':
             subsets = SUBSETS
        query_type = 'full_text-kw'
        if query_type not in self.clients:
            self.clients[query_type] = BM25Client(self.index_path, query_type)
        all_scored_meta = []
        subset_scored_meta = {subset: [] for subset in subsets}
        for subset in subsets:
            top_results = self.query_subset_articles(subset, prompt=text,
                                                     top_k=top_k, normalize=normalize)
            boost = subset_boosts.get(subset, 0)
            for file_data, score in top_results:
                score += boost
                file_data['subset'] = subset
                all_scored_meta.append((file_data, score))
                subset_scored_meta[subset].append((file_data, score))
        if collapse_subsets: # re-sort by highest score between subsets
            final_scored_meta = sorted(all_scored_meta, key=lambda x: x[1], reverse=True)[:top_k]
        else:
            final_scored_meta = subset_scored_meta
        # return list of (score, metadata) tuples
        return final_scored_meta