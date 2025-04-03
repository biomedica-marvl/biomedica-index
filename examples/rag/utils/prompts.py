###########################################
# FEEL FREE TO ADD YOUR OWN PROMPTS       #
# WE INCLUDE THESE JUST AS BASIC EXAMPLES #
###########################################

BASIC_PROMPTS = dict(
    ### Prompt to answer question w/o RAG ###
    question = "Answer the question to the best of your capabilities, be concise and precise",

    ### Prompt to convert question to query ###
    question2query = "Convert the question into optimized search terms for BM25 by removing stop words and keeping only key terms. Key terms include important nouns, meaningful verbs, proper nouns, and technical terms. Return only the refined search terms.",

    ### Prompt to summarize evidence        ###
    get_info = """As a subject expert, (1) summarize the evidence provided by a given ARTICLE as it pertains to a given QUESTION and (2) provide a possible answer. To achieve this, if the provided article contains relevant information, you must return a list including the following items:

    - **Summary**: A concise but comprehensive summary based on the previously specified information, with a focus on the main findings.  
    - **Possible Answer **: A concise feasible answer given the evidence.
    
    Think step by step.
    {text}
    """,

    ### Str add values to summarize evidence prompt ###
    get_info_values = "\n\n**QUESTION**: '{question}'\n **ARTICLE TITLE**: '{title}'\n **ARTICLE CONTENT**: '{context}'", 

    ### Prompt to generate  final answer (given all evidence) ###
    final_answer = """Given the ARTICLE SUMMARIES. Provide a concise and precise answer to the provided QUESTION.

    After you think, return your answer with the following format:
    - **Rationale**: Your rationale
    - **Full Answer**:  A precise answer, citing each fact with the Article PMCID in brackets (e.g. [PMCID]).
    - **Answer**: The Precise and concise final answer (no citations)

    Think step by step.
    {text}
    """,

    ### Str add values to final answer prompt ###
    final_answer_values = "\n\n**QUESTION**: '{question}'\n **ARTICLE SUMMARIES**: {context}"
)

PROMPT_SETS = {
    'basic': BASIC_PROMPTS
}