from transformers import GPT2Tokenizer
import re
import hashlib


def strip_thinking_tokens(text: str) -> str:
    """
    Removes any <think>...</think> tokens from the provided text.

    Args:
        text (str): The text to be processed.

    Returns:
        str: The text with <think>...</think> tokens removed.
    """
    return re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL).strip()


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


class Page:
    """
    Represents a page containing text content and metadata about its length in characters and tokens.

    Attributes
    ----------
    tokenizer : GPT2Tokenizer
        Tokenizer for encoding text using GPT-2.
    page_content : str
        The textual content of the page.
    metadata : dict
        Dictionary containing string length ('str_len') and token length ('token_len').
    """
    def __init__(self, content: str, tokenized_content: [int]):
        """
        Initializes a Page instance.

        Parameters
        ----------
        content : str
            The text content of the page.
        tokenized_content : list of int
            List of tokenized integers representing the content.
        """
        self.tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
        self.page_content = content
        self.metadata = {"str_len": len(content), "token_len": len(tokenized_content)}

    def get_num_tokens(self, text: str) -> int:
        """
        Computes the number of tokens in a given text.

        Parameters
        ----------
        text : str
            The text to be tokenized.

        Returns
        -------
        int
            The number of tokens in the text.
        """
        tokens = self.tokenizer.encode(text, add_special_tokens=False, verbose=False)
        return len(tokens)
    
    def update_content(self, new_content: str):
        """
        Updates the page content and its metadata.

        Parameters
        ----------
        new_content : str
            The new content to be updated.
        """
        self.page_content = new_content
        self.metadata = {"str_len": len(new_content), "token_len": self.get_num_tokens(new_content)}
        
    def __str__(self) -> str:
        """
        Returns a string representation of the Page object.

        Returns
        -------
        str
            A string describing the page's content length and token length.
        """
        return f"Page with str_len: {self.metadata['str_len']} and token_len: {self.metadata['token_len']}"

    def __repr__(self) -> str:
        """
        Returns a formal string representation of the Page object.

        Returns
        -------
        str
            A string representation of the Page object.
        """
        return self.__str__()
        
class TextSplitter:
    """
    Splits text into chunks based on a maximum token length using the GPT-2 tokenizer.

    Attributes
    ----------
    max_num_tokens : int
        The maximum number of tokens allowed per chunk.
    tokenizer : GPT2Tokenizer
        Tokenizer for encoding text using GPT-2.
    """
    def __init__(self, max_num_tokens=None):
        """
        Initializes a TextSplitter instance.
        """
        self.tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
        self.max_num_tokens = max_num_tokens

    def get_num_tokens(self, text: str) -> int:
        """
        Computes the number of tokens in a given text.

        Parameters
        ----------
        text : str
            The text to be tokenized.

        Returns
        -------
        int
            The number of tokens in the text.
        """
        tokens = self.tokenizer.encode(text, add_special_tokens=False, verbose=False)
        return len(tokens)
    
    def set_max_tokens(self, n_tokens):
        self.max_num_tokens = n_tokens
    
    def split(self, text: str, max_num_tokens=None) -> list:
        """
        Splits a given text into chunks based on the maximum number of tokens allowed.

        Parameters
        ----------
        text : str
            The text to be split.

        Returns
        -------
        list of Page
            A list of Page objects containing text chunks and their tokenized representations.
        """
        max_tok = max_num_tokens or self.max_num_tokens
        assert max_tok is not None
        tokens = self.tokenizer.encode(text, add_special_tokens=False, verbose=False)
        
        if len(tokens) <= max_tok:
            return [Page(text, tokens)]
        
        token_chunks = [tokens[i:i + max_tok] for i in range(0, len(tokens), max_tok)]
        text_chunks = [Page(self.tokenizer.decode(chunk), chunk) for chunk in token_chunks]
    
        return text_chunks