import os
import json
from collections import defaultdict

import boto3
import botocore.exceptions

from biomedica_index.clients import BM25Client, VectorClient
from biomedica_index.utils.embed_utils import SentenceEmbedder, ImageEmbedder
from biomedica_index.utils.consts import SUBSETS, S3_BUCKET, S3_REGION, INDEX_PATH_VAR

class BiomedicaArticleLoader:
    """
    Class to load full-text article data given identifying metadata (subset and PMCID).
    Functions intended for internal use have a leading _underscore.
    """
    def __init__(self, index_path=None, pmcid_article_map=None, local_article_path=None):
        """
        Initializes the index.

        Parameters:
            index_path (str): path to the data needed for the index. used to find the mapping of
                PMCIDs to actual article text
            pmcid_article_map (str): (optional) path to an overriding JSON file that maps PMCIDs
                to the locations of article batch files, as they are stored on the user's system
            local_article_path (str): (optional) path to the local directory where the article
                full-text is stored. If not provided, the loader attempts to find it in the default
                place within the index data itself. Currently required, but will be fully optional
                in the future.
        """
        if index_path is None:
            if (index_path := os.getenv(INDEX_PATH_VAR)) is None:
                raise ValueError("Index path must be given as a parameter "
                                 f"or in the environment variable {INDEX_PATH_VAR}")

        default_root = index_path + "/ARTICLES"
        default_map_path = index_path + "/full_text-kw/full/pmcid_map.json"

        self.local = True # try to read locally first
        if pmcid_article_map_path is not None:
            # option 1: provide your own mapping of PMCIDs directly to article batch files
            with open(pmcid_article_map_path, 'r') as fh:
                self.article_maps = json.load(fh)
            self.local_root = None
        else:
            # use the default PMCID -> article map
            with open(default_map_path, 'r') as fh:
                self.article_maps = json.load(fh)
            # option 2: provide a path to where the article batch files are stored
            if local_article_path is not None:
                local_root = local_article_path
                if os.path.exists(local_root):
                    self.local_root = local_root
                else:
                    raise ValueError(f"No valid path to articles found at {local_root}")
            # option 3: read from the default path where article batch files should be
            elif os.path.exists(default_root):
                self.local_root = default_root
            # option 4: use S3 bucket
            else:
                self.local = False
                self.s3_client = boto3.client('s3', region_name=S3_REGION)

    def _get_article_s3(self, subset, pmcid):
        raise NotImplementedError("S3 Bucket is currently not available")
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
        
    def _get_article_local(self, subset, pmcid):
        batch_path = self.article_maps[subset][pmcid]
        if self.local_root is not None:
            batch_path = os.path.join(self.local_root, batch_path)
        with open(batch_path, 'r') as article_batch_fh:
            article_batch = json.load(article_batch_fh)
        for article in article_batch:
            if article["accession_id"] == pmcid:
                return article['title'], article['nxml'], article['date']
        return None, None, None

    def get_article(self, article_metadata):
        """
        Loads article full-text data based on the given metadata.

        Parameters:
            article_metadata (dict): dictionary containing metadata for the article. Must contain:
                'subset': the Biomedica subset where the article is located
                'pmcid': the PMCID of the article
        
        Returns:
            tuple of data for the full-text article, specifically: (
                title (str): the title of the article,
                nxml (str): the XML body of the article, saved as a string
                date (str): the date the article was published, represented as a Unix timestamp
            )
        
        Raises:
            KeyError: if the input does not include all necessary identifying metadata
            (for now) NotImplementedError: if the user attempts to load articles non-locally
        """
        if not all(key in article_metadata for key in ('subset', 'pmcid')):
            raise KeyError("Local loading requires `subset` and `pmcid`")
        get = self._get_article_local if self.local else self._get_article_s3
        return get(article_metadata['subset'], article_metadata['pmcid'])


class BiomedicaIndex:
    """
    Class to retrieve the most relevant items in the Biomedica dataset given a query.
    Can retrieve most-relevant articles by text and most-relevant figures by image or text.
    Functions intended for internal use have a leading _underscore.
    """
    VEC_QUERY_TYPES = ['image', 'caption']
    KEYWORD_QUERY_TYPES = ['full_text-kw', 'caption-kw']
    QUERY_TYPES = VEC_QUERY_TYPES + KEYWORD_QUERY_TYPES

    def __init__(self, index_path=None, embedder_device=None, search_multiplier=1, RRF_k=60):
        """
        Initializes the index.

        Parameters:
            index_path (str): path to the data needed for the index
            embedder_device (str OR torch.device): device used for query embedding models
            search_multiplier (int): expands the number of items used for hybrid search. specifically,
                the top (k*search_multiplier) results from each search type are considered during RRF
            RRF_k (float): k-value used in the RRF denominator (default is 60, which is commonly used)
        """
        if index_path is None:
            if (index_path := os.getenv(INDEX_PATH_VAR)) is None:
                raise ValueError("Index path must be given as a parameter "
                                 f"or in the environment variable {INDEX_PATH_VAR}")
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

    def _rrf_rank(self, meta_lists):
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

    def _query_figures_by_type(self, subsets, query_type, prompt, n_results=5):
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

    def _query_subset_articles(self, subset, prompt, top_k=5):
        files_found = {}
        article_scores = []
        pmcids, scores = self.clients['full_text-kw'].query(
            [prompt], n_results=top_k, subset=subset
        )
        files_found.update({_id: {'pmcid': _id} for _id in pmcids})
        article_scores = list(zip(pmcids, scores.tolist()))
        # sort by highest score
        top_scores = sorted(article_scores, key=lambda x: x[1], reverse=True)[:top_k]
        top_results = [(files_found[pmcid], score) for (pmcid, score) in top_scores]
        return top_results

    def query_figures(self, image=None, text=None, text_mode='all', top_k=5, subsets='all'):
        '''
        Retrieves the most relevant figures given an image or caption query.

        Parameters:
            image (PIL.Image): image to use as query for images (embedding similarity search)
            text (str): text to use as query for figure captions
            text_mode (str): specifies how the text query should be used. one of:
                'keyword' = use keyword-based (BM25) search only
                'vector' = use vector-based (embedding similarity) search only
                'all' = use both bm25 and vector-based search
            top_k (int): number of highest-ranking items to return
            subsets (str OR list[str]): either the string 'all' (to use all subsets)
                or a list containing some subset of {'commercial','noncommercial','other'}

        Returns:
            a list of (RRF score, metadata_dict) tuples, where each metadata_dict contains:
                pmcid (str): the PMCID of the article the figure comes from
                subset (str): the Biomedica subset the figure comes from
                shard (str): the ID for the Biomedica Webdataset shard the figure comes from
                filekey (str): unique identifier for the figure
                caption_text (str): the text of the original figure caption

        Raises:
            AssertionError: if text_mode or any of the subsets is not a valid option
        '''
        assert text_mode in ['keyword', 'vector', 'all'],\
            f"{text_mode} is not a recognized text query type"
        if subsets == 'all':
             subsets = SUBSETS
        else:
            for sub in subsets:
                assert sub in SUBSETS, f"{sub} is not a valid subset"
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
            best_metadata = self._query_figures_by_type(subsets, q_type, prompt, n_results=N)
            meta_dicts_per_type.append(best_metadata)
        # sort by highest score
        final_scored_meta = self._rrf_rank(meta_dicts_per_type)
        # return list of (rrf_score, metadata) tuples
        return final_scored_meta

    def query_articles(self, text, top_k=5, subsets='all'):
        '''
        Retrieves the most relevant figures given an image or caption query.

        Parameters:
            text (str): text to use as query for relevant articles via keyword-based (BM25) search
            top_k (int): number of highest-ranking items to return
            subsets (str OR list[str]): either the string 'all' (to use all subsets)
                or a list containing some subset of {'commercial','noncommercial','other'}

        Returns:
            a list of (BM25 score, metadata_dict) tuples, where each metadata_dict contains:
                pmcid (str): the PMCID of the article
                subset (str): the Biomedica subset the article comes from

        Raises:
            AssertionError: if any of the subsets is not a valid option
        '''
        if subsets == 'all':
             subsets = SUBSETS
        else:
            for sub in subsets:
                assert sub in SUBSETS, f"{sub} is not a valid subset"
        query_type = 'full_text-kw'
        if query_type not in self.clients:
            self.clients[query_type] = BM25Client(self.index_path, query_type)
        all_scored_meta = []
        subset_scored_meta = {subset: [] for subset in subsets}
        for subset in subsets:
            top_results = self._query_subset_articles(subset, prompt=text, top_k=top_k)
            for file_data, score in top_results:
                file_data['subset'] = subset
                all_scored_meta.append((file_data, score))
                subset_scored_meta[subset].append((file_data, score))
        # re-sort by highest score between subsets
        final_scored_meta = sorted(all_scored_meta, key=lambda x: x[1], reverse=True)[:top_k]
        # return list of (score, metadata) tuples
        return final_scored_meta