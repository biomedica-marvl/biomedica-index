from langchain.prompts import PromptTemplate
from langchain.chains  import load_summarize_chain
from langchain_community.chat_models import ChatOpenAI

from llms.queryllms import QueryLLM
import utils.text_utils as t_utils

class QuerySummaryChainLLM:
    def __init__(
            self,
            llm_client:QueryLLM,
            instruction_prompt:str,
            summary_template:str=None
    ):
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

        self.client = llm_client
        # Load the summarization chain
        self.chain = load_summarize_chain(
            llm= self.client,
            chain_type= "refine",
            question_prompt= self.prompt,
            refine_prompt  = self.refine_prompt,
            return_intermediate_steps=True,
            input_key  = "input_documents",
            output_key = "output_text"
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
        result["output_clean"] = t_utils.strip_thinking_tokens(result["output_text"])
        return result