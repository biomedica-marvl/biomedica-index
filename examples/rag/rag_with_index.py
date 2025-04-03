import os 
import re
import sys
import time
import json
from datetime import datetime, UTC

import numpy as np
from datasets import load_dataset
from langchain.chains.summarize import load_summarize_chain
from langchain_text_splitters   import CharacterTextSplitter
from langchain_core.prompts import PromptTemplate
from langchain.chains.summarize import load_summarize_chain
from biomedica_index       import BiomedicaIndex, BiomedicaArticleLoader

### Local Libraries ##
from llms.queryllms        import QueryLLM
from llms.querychainllms   import QuerySummaryChainLLM
from utils.citation_tools  import PubMedCitation
from utils.text_utils      import TextSplitter, Page
from utils.json_writers    import pretty_print_store,update_json_file,read_jsonl,save_jsonl

## Load prompts:
from utils.prompts         import PROMPT_SETS


#import warnings
#warnings.filterwarnings("ignore", category=FutureWarning)  # Suppress FutureWarnings
#warnings.filterwarnings("ignore", category=UserWarning)    # Suppress UserWarnings
from langchain_core.globals import set_verbose, set_debug

set_verbose(False) # Disable langchain verbose logging
set_debug(False)   # Disable  langchain_core debug logging

def print_message(message:str,rep:int):
    print("*"*rep)
    print(message)
    print("*"*rep)
    print()

def print_sep(rep:int=2):
    print("*"*rep)
    print("----"*15)

class BiomedicaRAG:
    KNOWN_MAX_TOKENS = {"gpt-4o":128000,"deepseek-ai/DeepSeek-R1-Distill-Llama-70B-free":4096}
    def __init__(self, 
             ### LLM arguments: ###
             provider: str, 
             model:    str, 
             api_key:  str      = None, 
             parameters:dict    = {"temperature":0,
                                   "timeout":None,
                                   "max_retries":10,
                                   "max_tokens":2000},
             delay:int          = None,
             cache:bool         = True,
             use_cache:bool     = True,
             enable_logger:bool = False,
             host_vllm_manually:bool = True,
             prompt_type:str       = 'basic',

            ### Index arguments: 
            query_subsets:str   = "all",
            index_kwargs:dict   = {},
            loader_kwargs:dict  = {}
            ):
        ### LLM arguments: ###
        inference_server_url = "http://localhost:8000/v1"
        
        self.provider   = provider
        self.api_key    = api_key
        self.model      = model
        self.model_name = model.replace("/","-")
        self.delay      = delay
        self.parameters = parameters
        self.cache      = cache
        self.use_cache  = use_cache
        self.enable_logger = enable_logger
        self.inference_server_url  = inference_server_url
        self.host_vllm_manually    = host_vllm_manually
        # option 1: explicitly set max_tokens > option 2: max_tokens known for model type > 4096 default
        self.max_input_token = parameters.get("max_tokens", self.KNOWN_MAX_TOKENS.get(self.model, 4096))

        self.prompts = PROMPT_SETS[prompt_type]
        ### Index arguments:
        self.query_subsets = query_subsets
      
        self.init_llms()      # Initalize LLMs
        self.init_splitter()  # Initalize text splitter (used for llm chains when artilces are longet than LLM's context) 
        self.init_index(index_kwargs=index_kwargs, loader_kwargs=loader_kwargs)     # Init index
        
    def init_llms(self):
        """Initialize LLMs for querying and summarization.
    
        This method initializes two types of language models:
        1. A standard LLM (`self.llm`) for handling queries within the model's context length.
        2. An LLM chain (`self.get_info_chain`) for processing longer articles using a multi-step refinement approach.

        The LLM chain requires an additional instruction prompt (`get_info`) to guide the summarization process.
        These two models have different parameter configurations to accommodate their distinct processing strategies.
        """
        self.llm = QueryLLM(
            provider=self.provider,
            api_key =self.api_key,
            model   =self.model,
            parameters=self.parameters,
            host_vllm_manually=self.host_vllm_manually)

        self.get_info_chain = QuerySummaryChainLLM(
            llm_client=self.llm.client,
            instruction_prompt=self.prompts['get_info'])

        self.compile_source_chain = QuerySummaryChainLLM(
            llm_client=self.llm.client,
            instruction_prompt=self.prompts['final_answer'])

            
    def init_splitter(self) -> None:
        """Initialize the text splitter for handling long articles.
    
        The text splitter is used to divide articles that exceed the LLM's context length.
        It ensures that long texts are processed efficiently, particularly for multi-step refinement in LangChain.
    
        - `self.splitter.split_text(str)` returns an instance of `Page`, which is required for structured refinement.
        - The maximum input token limit is determined based on the model's capacity, subtracting the space needed 
          for the instruction prompt.
        """
        # arbitrary choice, but used to account for context provided in iterative refinement
        prompt_size_buffer = 1000
        self.splitter   = TextSplitter(max_num_tokens=self.max_input_token - prompt_size_buffer)


    def init_index(self, index_kwargs={}, loader_kwargs={}, verbose=True) -> None:
        """
        Initialize the index and article loader for biomedical retrieval.
    
        Parameters
        ----------
        index_kwargs : dict, optional
            Keyword args to pass to the index
        loader_kwargs : dict, optional
            Keyword args to pass to the article loader
        verbose : bool, optional
            If True, prints the total initialization time (default is True).
    
        Returns
        -------
        None
        """
        load_start = time.time()
        self.index = BiomedicaIndex(**index_kwargs)
        self.loader = BiomedicaArticleLoader(**loader_kwargs)
        load_end   = time.time()
        if verbose:
            print(f"TOTAL LOAD/INITIALIZATION TIME: {load_end - load_start:.2f}s")


    def forward(self,question,n_articles:int=14,verbose:bool=True,save:bool=True):
        """
        Process a given question by retrieving and summarizing relevant articles, then generating an answer.
    
        Parameters
        ----------
        question : str
            The input question to be answered.
        n_articles : int, optional
            The number of articles to retrieve and summarize (default is 14).
        verbose : bool, optional
            If True, enables detailed logging of the process (default is True).
        save : bool, optional
            If True, saves the generated answer (default is True).
    
        Returns
        -------
        dict[str, str]
            A dictionary containing the generated answer.
        """
        query = self.create_query(question,verbose=verbose)
        summaries,articles  = self.summarize_evidence(query,question,n_articles=n_articles,verbose=verbose)
        answer:dict[str,str]= self.answer_question(question,summaries,articles,verbose=verbose,save=save)
                           
        return answer


    def create_query(self,question,verbose:bool=True) -> str:
        """Generate a query using the LLM for retrieval-augmented generation (RAG).
    
        This method converts a natural language question into a structured query using `self.llm.simple_query`.
        The generated query is used to retrieve relevant information for downstream processing.
    
        Args:
            verbose (bool, optional): If True, prints detailed information about the query generation process. 
                                      Defaults to True.
    
        Returns:
            str: The generated query response.
        """

        query = self.llm.simple_query(self.prompts['question2query'],question)
        
        if verbose:
            print_sep(rep=2)
            print_message(message="RAG GENERATION:",rep=15)
            print_message(message="Question to Query Conversion",rep=27)
            print(f"question: {question}\nquery: {query}\n")
            
        return query

    def summarize_evidence(self,
                           query:str,
                           question:str,
                           n_articles:int=14,
                           verbose:bool=True)->dict[str,str]:

        """
        Summarizes the evidence from the top retrieved articles based on the query and question.
    
        Args:
            query (str): The search query to retrieve articles.
            question (str): The question to be answered based on the retrieved articles.
            n_articles (int, optional): The number of top articles to retrieve and summarize. Default is 14.
            verbose (bool, optional): Whether to print verbose messages during processing. Default is True.
    
        Returns:
            dict[str, str]: A dictionary where the keys are PMCID of the articles and the values are the corresponding article summaries.
            
        Process:
            1. Retrieve the top `n_articles` from the index based on the query.
            2. For each article:
                - Fetch metadata and full text.
                - Clean the title and extract the date and subset information.
                - Split the article into chunks if necessary.
                - Use the language model to generate summaries, either for the whole text or chunked content.
            3. Return the summaries for each article as well as the metadata of the retrieved articles.
        """
        
        top_articles   = self.index.query_articles(
            text      = query, 
            top_k     = n_articles,
            subsets   = self.query_subsets, 
        )
        
        evidence_summaries:dict[str,str] = {}
        retrieved_articles:dict[str,str]  = {}
    
        if verbose:
            print_message(message="Reading Articles",rep=18)
            
        for i,(article_metadata, score) in enumerate(top_articles):
            ###########################################
            ##  step 4) Get article and its metadata ##
            ###########################################
            title, full_text,date= self.loader.get_article(article_metadata)
            title:str = re.sub(r"\{'.*?'\}", '', title)
            if not date:
                date_str = "Date not provided"
            elif date.isdigit(): # date is unix time
                date_str  = datetime.fromtimestamp(int(date), UTC).date().strftime("%Y-%m-%d")
            else:
                date_str = date
            pmcid     = article_metadata['pmcid']
            subset    = article_metadata['subset']
            
            retrieved_articles[article_metadata['pmcid']] = {
                "title":title,
                "date":date_str,
                "pmcid": pmcid,
                "subset": subset}
           
            ###########################
            ##  step 5) Apply prompt ##
            ###########################
            chunks         = self.splitter.split(full_text)
            use_chain:bool = len(chunks) >  1 
            query_msg:str = f"[{i+1}/{n_articles}] Getting summary for:\n\"{title}\""
            
            if use_chain == False:
                prompt = self.prompts['get_info'].replace("{text}","") + self.prompts['get_info_values'].replace("{question}",question).replace("{title}",title).replace("{context}",chunks[0].page_content)
                if verbose:
                    print(query_msg)
                summary:str = self.llm.simple_query("follow instruction", prompt)
                
            else:
                query_msg += f"\nArticle has been splitted into {len(chunks)} chunk, thus use_llm_chain={use_chain}"
                [c.update_content(self.prompts['get_info_values'].replace("{question}",question).replace("{title}",title).replace("{context}",c.page_content)) for c in chunks]
                if verbose:
                    print(query_msg)
                summary_log:str = self.get_info_chain.querychain(documents=chunks)
                summary:str = summary_log["output_clean"]
          
            evidence_summaries[article_metadata['pmcid']] = summary
    
        return evidence_summaries,retrieved_articles



    def format_article_summaries(self,
                                  evidence_summaries,
                                  retrieved_articles,
                                  th_filter:int=20) -> list[str]:
        """
        Packages article summaries into a formatted string.
    
        Parameters
        ----------
        evidence_summaries : dict
            A dictionary where keys are PMCID strings and values are lists of relevant summaries.
        retrieved_articles : dict
            A dictionary containing article metadata, where keys are PMCIDs and values are dictionaries
            with 'title', 'date', and 'pmcid' fields.
        th_filter : int, optional
            The minimum number of summaries required for an article to be included (default is 20).
    
        Returns
        -------
        str
            A formatted string containing all included article summaries.
    
        Notes
        -----
        - If an article has fewer summaries than `th_filter`, it will be skipped.
        - Assumes `retrieved_articles` contains valid metadata for all PMCIDs in `evidence_summaries`.
        - Prints "Time to read!" when executed.
    
        Raises
        ------
        AssertionError
            If the PMCID in `retrieved_articles` does not match the key in `evidence_summaries`.
        """

        # print("Time to read!")
        article_summaries:list[str] = []
        for i,(PMCID,summaries) in enumerate(evidence_summaries.items()):
            if len(summaries) > th_filter:
                title = retrieved_articles[PMCID]['title']
                date  = retrieved_articles[PMCID]['date']
                pmcid = retrieved_articles[PMCID]['pmcid']
                assert pmcid == PMCID
                article = f"----\n\nArticle number {i+1}:\nTITLE: {title}\nPublished Date: {date}\nPMCID: {pmcid} \nRelevant Information:\n{summaries}\n\n----"
                article_summaries.append(article)

        return article_summaries

    def merge_article_summaries(self,
                                question:str,
                                article_summaries:list[str],
                                verbose:bool=True) -> list[Page]:
        max_len = self.splitter.max_num_tokens
        summaries_header = self.prompts['final_answer_values'].replace('{question}', question).replace('{context}',"")
        init_token_count = self.splitter.get_num_tokens(summaries_header)
        summaries_str = summaries_header
        running_token_count = init_token_count
        chunks = []
        def add_page(summaries): # process a chunk of summaries as a Page and reset for the new article
            summaries_pages = self.splitter.split(summaries)
            if len(summaries_pages) > 1 and verbose:
                print("WARNING: summaries for a single article exceeds token limit, ignoring past first chunk")
            chunks.append(summaries_pages[0])

        for article in article_summaries:
            article_num_tokens = self.splitter.get_num_tokens(article)
            if running_token_count + article_num_tokens > max_len:
                add_page(summaries_str)
                # reset the count and string
                running_token_count = init_token_count
                summaries_str = summaries_header
            running_token_count += article_num_tokens
            summaries_str += article
        # make the last page
        if running_token_count == init_token_count: # EDGE CASE: no articles
            summaries_str += "No articles available"
        add_page(summaries_str)
        return chunks

    def format_final_answer(self,STORE):
        # Define a pattern to extract the key-value pairs
        pattern = r"(-\s*)?\*\*(.*?)\*\*:\s*((.|\n)*?)(?=\n+(-\s*)?\*\*|\Z)"
        matches = re.findall(pattern, STORE["answer"] , re.DOTALL) # Extract matches
        if matches:
            # Convert to a dictionary
            parsed_data = {}
            for (_,key,value,_,_) in matches:
                formatted_key = key.strip().lower().replace(" ","_")
                parsed_data[formatted_key] = value.strip()
            STORE["parsed_answer"]          = parsed_data
        else:
            STORE["parsed_answer"]          = "Could not parsed"

    def answer_question(self,
                        question:str,
                        evidence_summaries:dict,
                        retrieved_articles:dict,
                        metadata:dict = None,
                        verbose:bool=True,
                        save:bool=True) -> dict[str,str]:
        """
        Generate an answer to a given question based on retrieved evidence summaries.
    
        Parameters
        ----------
        question : str
            The input question to be answered.
        evidence_summaries : dict
            A dictionary containing summarized evidence from retrieved articles.
        retrieved_articles : dict
            A dictionary containing the retrieved articles used for answering.
        verbose : bool, optional
            If True, enables detailed logging of the process (default is True).
        save : bool, optional
            If True, saves the generated answer and metadata (default is True).
    
        Returns
        -------
        dict[str, str]
            A dictionary containing the generated answer along with relevant metadata.
        """
        
        log_msg = "Answer with RAG"
        article_summaries:list[str] =  self.format_article_summaries(evidence_summaries,retrieved_articles)

        prompt_chunks:list[Page] = self.merge_article_summaries(question, article_summaries, verbose=verbose)
        use_chain:bool = len(prompt_chunks) > 1
        if use_chain:
            log_msg += f" - long overall summary split into {len(prompt_chunks)} chunk and using llmchain"
            answer_log:str = self.compile_source_chain.querychain(documents=prompt_chunks)
            final_answer:str = answer_log["output_clean"]
        else:
            # add the header prompt
            summaries_text = prompt_chunks[0].page_content
            prompt_chunks[0].update_content(self.prompts['final_answer'].replace("{text}",summaries_text))
            final_answer:str = self.llm.simple_query("follow instruction",prompt_chunks[0].page_content)
        
        STORE = {} 
        if metadata:
            STORE.update(metadata)
        STORE["queried_subsets"] = self.query_subsets 
        STORE["n_articles"]      = len(retrieved_articles)
        STORE["question"]        = question
        STORE["articles"]        = retrieved_articles
        STORE["summaries"]       = evidence_summaries
        STORE["answer"]          = final_answer
        self.format_final_answer(STORE)
        
        if save:
            update_json_file(f"{self.model_name}_index.json", key=question, value=STORE)
            
        if verbose:
            print_message(message="",rep=20)
            print_message(message=log_msg,rep=20)
            
            pretty_print_store(STORE)
            print_sep(rep=2)

        return STORE

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--provider", type=str, default="together", \
        help="LLM API provider. vllm = locally-hosted (default: together)",
        choices=["openai", "google", "together", "anthropic", "vllm"])
    parser.add_argument("--model", type=str,\
        default="deepseek-ai/DeepSeek-R1-Distill-Llama-70B-free",
        help="path to the exact model for the given provider "
        "(default: deepseek-ai/DeepSeek-R1-Distill-Llama-70B-free)"
    )
    parser.add_argument("--index-path", type=str, \
        help="Path to the index (if not provided in the environment)")
    parser.add_argument("--article-map", type=str, \
        help="Override default mapping used by BiomedicaArticleLoader for PMCIDs -> articles")
    parser.add_argument("--article-path", type=str, \
        help="Override default location where articles are stored")
    args = parser.parse_args()
    api_key_var = args.provider.upper() + "_API_KEY"
    if (args.provider != 'vllm') and (api_key_var not in os.environ):
        api_key = input(f"Please provide your API key for {args.provider}: ")
    else:
        api_key = None
    n_articles:int=14
    biomedica_rag = BiomedicaRAG(
        args.provider, args.model, api_key, query_subsets="all",
        index_kwargs=dict(index_path=args.index_path),
        loader_kwargs=dict(
            index_path=args.index_path,
            pmcid_article_map=args.article_map,
            local_article_path=args.article_path,
        ),
    )
    # question:str  = "What are the most common genetic mutations in activated B cell-like (ABC) diffuse large B-Cell lymphoma (DLBCL)?"
    while (question := input('Question ["q" to quit] > ')) != 'q':
        query_start = time.time()
        output = biomedica_rag.forward(question,n_articles=n_articles)
        query_end = time.time()
        print(output['answer'])
        print(f">>> total runtime: {query_end - query_start}s <<<")
