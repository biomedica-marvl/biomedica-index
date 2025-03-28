# Changelog

## v0.2.1 (2025/03/28)
- add more options for infrastructure supporting the ArticleLoader and the Index

## v0.2.0 (2025/03/24)
- add docstrings for user-facing functions
- eliminate args used for dev purposes and refactor function and class names
    - BREAKING CHANGE: BiomedicaRetriever renamed to BiomedicaIndex for clarity

## v0.1.2 (2025/03/24)
- enable full-text article loading capability via BiomedicaArticleLoader

## v0.1.1 (2025/03/24)
- acceleration of keyword/BM25 retrieval by using numba and JAX for BM25 operations

## v0.1.0 (2025/03/24)
- baseline index retrieval systems implemented: tested functioning retrieval for articles and captions