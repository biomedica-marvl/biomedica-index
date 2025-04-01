from langchain.prompts import PromptTemplate
from langchain.chains  import load_summarize_chain
from langchain_community.chat_models import ChatOpenAI

from llms.queryllms import QueryLLM


class QuerySummaryChainLLM(QueryLLM):
    """
    Class to interface with various LLM providers for query and summarization chains.

    Args:
        provider (str): The provider to use (e.g., "google", "openai").
        model (str): The model to query.
        api_key (str): API key for accessing the provider's service.
        parameters (dict): Optional parameters for API queries (e.g., temperature, max_tokens).
        delay (int): Optional delay (in seconds) before sending the query.
        cache (bool): Whether to use caching for repeated queries.
        use_cache (bool): Whether to check cache for responses.
        enable_logger (bool): Whether to enable logging for the class operations.
    """
    
    def __init__(self, 
                 provider: str, 
                 model: str, 
                 instruction_prompt:str,
                 summary_template:str = None,
                 api_key: str         = None, 
                 parameters: dict     = {"temperature": 0, "max_tokens": None, "timeout": None, "max_retries": 2}, 
                 delay: int           = None, 
                 cache: bool          = True, 
                 use_cache: bool      = True, 
                 enable_logger: bool  = False,
                 host_vllm_manually:bool = True):
        super().__init__(provider, model, api_key, parameters, delay, cache, use_cache, enable_logger, host_vllm_manually)

        # Define the prompt templates for summarization
        self.prompt_template = instruction_prompt

        if summary_template:
            self.refine_template = summary_template
        else:
            self.refine_template = (
                "Your job is to refine a summary that is compliant with the instruction provided\n"
                "We have provided an existing result up to a certain point: {existing_answer}\n"
                "We have the opportunity to refine the existing summary"
                "(only if needed) with some more context below.\n"
                "------------\n"
                "{text}\n"
                "------------\n"
                "Given the new context, refine the original summary"
                "If the context isn't useful, return the original summary."
            )

        self.prompt        = PromptTemplate.from_template(self.prompt_template)
        self.refine_prompt = PromptTemplate.from_template(self.refine_template)

        # Load the summarization chain
        self.chain = load_summarize_chain(
            llm= self.client,
            chain_type= "refine",
            question_prompt= self.prompt,
            refine_prompt  = self.refine_prompt,
            return_intermediate_steps=True,
            input_key  = "input_documents",
            output_key = "output_text",
        )

    def querychain(self, documents: list):
        """
        Processes a list of documents and generates a concise summary using the chain.

        Args:
            documents (list): The input documents to be summarized.

        Returns:
            str: The final summary after processing the documents through the chain.
        """
        # Process the documents using the summarization chain
        result                 = self.chain.invoke({"input_documents": documents}, return_only_outputs=True)
        result["output_clean"] = self.strip_thinking_tokens(result["output_text"])
        return result