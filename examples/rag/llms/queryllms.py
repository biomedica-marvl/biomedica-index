import os
import re
import time
import logging
import hashlib
import subprocess


# from langchain.chat_models    import ChatAnthropic
from langchain_openai         import ChatOpenAI
from openai import OpenAI
# from langchain_google_genai   import ChatGoogleGenerativeAI
from langchain_together       import ChatTogether
from langchain.prompts        import PromptTemplate
from langchain.chains         import load_summarize_chain


# Set up logging configuration
def setup_logger():
    """
    Sets up a logger to capture and display log messages for QueryLLM.

    Returns:
        logging.Logger: Configured logger object for QueryLLM operations.
    """
    logger = logging.getLogger("QueryLLM")
    logger.setLevel(logging.DEBUG)
    handler = logging.StreamHandler()
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    return logger
    

class QueryLLM:
    """
    Class to interface with various LLM providers (OpenAI, Google, TogetherAI, Anthropic, vLLM).

    Args:
        provider (str): The provider to use (e.g., "google", "openai", "togetherai").
        api_key (str): The API key for accessing the provider's service.
        model (str): The specific model to query from the provider.
        parameters (dict): Optional parameters for API queries (e.g., temperature, max_tokens).
        delay (int): Optional delay (in seconds) before sending the query.
        cache (bool): Whether to use caching for repeated queries.
        use_cache (bool): Whether to check cache for responses.
        enable_logger (bool): Whether to enable logging for the class operations.

    Raises:
        ValueError: If the provider is not valid.
    """
    
    
    def __init__(self, 
                 provider: str, 
                 model:    str, 
                 api_key:  str=None, 
                 parameters:dict={"temperature":0,"max_tokens":None,"timeout":None,"max_retries":2},
                 delay:int  = None,
                 cache:bool = True,
                 use_cache:bool     = True,
                 enable_logger:bool = False,
                 host_vllm_manually:bool = True):
        """
        Initializes the QueryLLM instance, validating provider and setting up the necessary configurations.

        Args:
            provider (str): The LLM provider.
            api_key (str): API key for the chosen provider.
            model (str): Model name for querying.
            parameters (dict): Additional parameters like temperature, max_tokens.
            delay (int): Optional delay before making a request.
            cache (bool): Whether to cache responses.
            use_cache (bool): Whether to use cache during requests.
            enable_logger (bool): Enable logging for tracking operations.
        """
        valid_providers = {"google", "openai", "togetherai", "anthropic", "vllm"}
        if provider not in valid_providers:
            raise ValueError(f"Invalid provider '{provider}'. Must be one of {valid_providers}")

        inference_server_url = "http://localhost:8000/v1"
        
        self.provider   = provider
        self.api_key    = api_key
        self.model      = model
        self.delay      = delay
        self.parameters = parameters
        self.cache      = cache
        self.use_cache  = use_cache
        self.enable_logger = enable_logger
        self.inference_server_url = inference_server_url
        self.host_vllm_manually    = host_vllm_manually
        if self.cache:
            self.CACHE = {}

        if self.enable_logger:
            self.logger = setup_logger()
            self.logger.info(f"Initializing QueryLLM with provider: {provider}, model: {model}")

        self.init_api()

    def init_api(self):
        """
        Initializes the API client based on the provider, setting the appropriate API key and model.

        Raises:
            ValueError: If the provider is unsupported.
        """
        if self.enable_logger:
            self.logger.info("Initializing API client...")

        if self.provider == "togetherai":
            self.set_key("TOGETHER_API_KEY")
            self.client = ChatTogether(model=self.model, **self.parameters)
            self.system_tag = "system"
            self.user_tag   = "human"
        elif self.provider == "openai":
            self.set_key("OPENAI_API_KEY")
            self.system_tag = "system"
            self.user_tag   = "human"
            self.client = ChatOpenAI(model=self.model, **self.parameters)
        elif self.provider == "google":
            self.set_key("GOOGLE_API_KEY")
            self.system_tag = "system"
            self.user_tag   = "human"
            self.client     = ChatGoogleGenerativeAI(model=self.model, **self.parameters)
        elif self.provider == "anthropic":
            self.set_key("ANTHROPIC_API_KEY")
            self.system_tag = "system"
            self.user_tag   = "human"
            self.client     = ChatGoogleGenerativeAI(model=self.model, **self.parameters)

        elif self.provider == "vllm":
            if self.host_vllm_manually == False:
                print(f"initalizing vLLM server")
                self.init_vllm_server()
            self.client = ChatOpenAI(model=self.model, openai_api_key = "EMPTY", openai_api_base= self.inference_server_url, **self.parameters)
            self.system_tag = "system"
            self.user_tag   = "human"

    def init_vllm_server(self,delay:int=120)-> None:
        command = ["vllm", "serve", self.model]
        self.process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        self.pid     = self.process.pid
        time.sleep(delay)
        print(f"initalized vLLM server  for model {self.model} with PID: {self.pid}")
        
    def kill_vllm_server(self,delay:int=1)-> None:
        self.process = self.process.terminate()
        time.sleep(delay)
        print(f"finalized vLLM server  for model {self.model} with PID: {self.pid}")           

    def set_key(self, key_name) -> None:
        """
        Sets the API key as an environment variable or uses the provided API key.

        Args:
            key_name (str): The environment variable name for the API key.

        Raises:
            ValueError: If the API key is not provided or found.
        """
        if self.api_key != None:
            os.environ[key_name] = self.api_key
        else:
            self.api_key = os.getenv(key_name)

        if self.api_key == None:
            raise ValueError("API key should be provided as an environment variable or as a parameter.")

        if self.enable_logger:
            self.logger.info(f"API key for {key_name} set successfully.")

    def wrap_message(self, role: str, message: str):
        """
        Wraps a message with the specified role.

        Args:
            role (str): The role of the message sender (e.g., 'system', 'human').
            message (str): The content of the message.

        Returns:
            tuple: A tuple containing the role and the message.
        """
        return (role, message)

    def default_chat_wrap(self, messages: list[str]) -> list[str,str]:
        """
        Default wrapper for messages in a chat format (system and human).

        Args:
            messages (list): List of messages where the first is the system message and the second is the user message.

        Returns:
            list: Wrapped messages for system and user roles.
        """
        wrapped_messages = [
            self.wrap_message(role=self.system_tag, message=messages[0]),
            self.wrap_message(role=self.user_tag , message=messages[1])]
        return wrapped_messages

    def query(self, messages: list) -> str:
        """
        Queries the LLM provider with the given messages and caches the response if applicable.

        Args:
            messages (list): List of messages to send to the provider.

        Returns:
            str: The generated response from the LLM provider.
        """
        response = None

        if self.delay:
            if self.enable_logger:
                self.logger.info(f"Delaying for {self.delay} seconds...")
            time.sleep(self.delay)

        message_hash: str = self.generate_unique_hash(" ".join([f"{role}: {message}" for role, message in messages]))

        if self.cache and self.use_cache:
            if self.enable_logger:
                self.logger.info(f"Checking cache for message hash: {message_hash}")
            response = self.CACHE.get(message_hash, None)
            if response:
                response = response.get("response", None)

        if self.cache == False  or self.use_cache == False or response == None:
            if self.enable_logger:
                self.logger.info(f"Cache miss, querying the provider...")
            if self.provider in ["togetherai", "openai", "google", "anthropic"]:
                response = dict(self.client.invoke(messages))

            if self.provider == "vllm":
                response = dict(self.client.invoke(messages,model=self.model))


            if self.cache:
                self.CACHE[message_hash] = {"query": messages, "response": response}

            if self.enable_logger:
                self.logger.info(f"Response generated and cached.")

        return response

    def simple_query(self,system_prompt:str,human_message:str,return_dict:bool=False):
        messages  = self.default_chat_wrap([system_prompt,human_message])
        response  = self.query(messages)
        if return_dict:
            return response
        else:
            r_content = self.strip_thinking_tokens(response.get("content"))
            return r_content

    ### Helper functions
    @staticmethod
    def strip_thinking_tokens(text: str) -> str:
        """
        Removes any <think>...</think> tokens from the provided text.

        Args:
            text (str): The text to be processed.

        Returns:
            str: The text with <think>...</think> tokens removed.
        """
        return re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL).strip()

    @staticmethod
    def generate_unique_hash(input_string: str) -> str:
        """
        Generates a unique SHA-256 hash based on the input string.

        Args:
            input_string (str): The string to hash.

        Returns:
            str: The hexadecimal hash of the input string.
        """
        hash_object = hashlib.sha256()
        hash_object.update(input_string.encode('utf-8'))
        return hash_object.hexdigest()