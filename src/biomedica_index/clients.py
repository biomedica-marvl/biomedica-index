import os

import chromadb
import bm25s
import numpy as np

from biomedica_index.utils.consts import ARTICLE_CORPUS_SIZES, CAPTION_CORPUS_SIZES, \
    FILEKEY_MAP_DTYPES, SUBSETS

class BM25Client:
    CORPUS_STARTS = {
        'full_text-kw': {
            'other': 0,
            'noncommercial': ARTICLE_CORPUS_SIZES['other'],
            'commercial': ARTICLE_CORPUS_SIZES['other']+ARTICLE_CORPUS_SIZES['noncommercial']
        },
        'caption-kw': {
            'other': 0,
            'noncommercial': CAPTION_CORPUS_SIZES['other'],
            'commercial': CAPTION_CORPUS_SIZES['other']+CAPTION_CORPUS_SIZES['noncommercial']
        }
    }
    CORPUS_SIZES = {
        'full_text-kw': ARTICLE_CORPUS_SIZES,
        'caption-kw': CAPTION_CORPUS_SIZES
    }

    def __init__(self, path, query_type, mmap=True):
        self.root = f"{path}/{query_type}/full"
        self.mmap = mmap
        self.retrievers = {}
        self.filekeys = np.memmap(self.root+"/filekeys.dat", mode='r',
                                  dtype=FILEKEY_MAP_DTYPES[query_type])
        self.tokenizer = bm25s.tokenization.Tokenizer()
        self.retriever = bm25s.BM25.load(self.root, mmap=mmap, load_corpus=False)
        self.retriever.backend = 'auto'
        self.tokenizer.word_to_id = self.retriever.vocab_dict
        self.NULL_TOKEN_ID = len(self.retriever.vocab_dict)-1
        # make masks for bm25 calculations
        sizes = self.CORPUS_SIZES[query_type]
        starts = self.CORPUS_STARTS[query_type]
        self.subset_masks = {}
        for subset in self.CORPUS_SIZES[query_type]:
            mask = np.zeros(sum(sizes.values()))
            mask[starts[subset]:starts[subset] + sizes[subset]] = 1
            self.subset_masks[subset] = mask

    def query(self, prompts, subset, n_results=5):
        mask = self.subset_masks[subset]
        tokenized = self.tokenizer.tokenize(prompts, return_as="ids")
        if tokenized[0] == [self.NULL_TOKEN_ID]:
            # this means no valid tokens were found!!
            return [], np.array([])
        ixs, scores = self.retriever.retrieve(tokenized, k=n_results, \
            weight_mask=mask, n_threads=-1)
        fkeys = [s.decode() for s in self.filekeys[ixs[0]]]
        return fkeys, scores[0]


class VectorSubsetClient:
    def __init__(self, root_path, subset):
        chunk_folders = sorted([ f.name for f in os.scandir(root_path)
                                if f.is_dir() and f.name.startswith(subset)])
        self.chunk_clients = [chromadb.PersistentClient(path=f"{root_path}/{folder}")
                              for folder in chunk_folders]
        self.collections = [c.get_collection(subset) for c in self.chunk_clients]
        self.subset = subset
    
    def query(self, embeds, n_results=3):
        all_res = []
        for coll in self.collections:
            res = coll.query(
                query_embeddings=embeds,
                n_results=n_results,
                include=["distances", "metadatas"]
            )
            meta_dicts = res['metadatas'][0]
            dists = res['distances'][0]
            all_res += zip(dists, meta_dicts)
        all_res = sorted(all_res, key=lambda x: x[0]) # sort by distances
        # convert [(score1, {dict1}), (score2, {dict2})] -> (score1,score2), ({dict1}, {dict2})
        best_dists, best_meta_dicts = zip(*all_res[:n_results])
        return best_meta_dicts, best_dists
    
    def get_metadata(self, ids):
        found_metadata = {}
        for coll in self.collections:
            coll_items = coll.get(ids=ids, include=['metadatas'])
            for md in coll_items['metadatas']:
                found_metadata[md['filekey']] = md
        return [found_metadata[_id] for _id in ids]


class VectorClient:
    def __init__(self, root_path, subsets):
        self.root_path = root_path
        self.subset_clients = {}
    
    def get_client(self, subset):
        if subset not in subset in SUBSETS:
            raise ValueError(f"{subset} is not a valid subset")
        elif subset not in self.subset_clients:
            self.subset_clients[subset] = VectorSubsetClient(self.root_path, subset)
        return self.subset_clients[subset]

    def query(self, embeds, subset, n_results=3):
        return self.get_client(subset).query(embeds, n_results=n_results)
    
    def get_metadata(self, ids, subset):
        return self.get_client(subset).get_metadata(ids)