from pymed import PubMed

class PubMedCitation:
    """
    A class to handle PubMed article citation formatting.

    Attributes
    ----------
    pubmed : PubMed
        The PubMed tool instance for querying articles.
    """

    def __init__(self, email:str="your_email@example.com"):
        """
        Parameters
        ----------
        email : str, optional
            The email address to use with the PubMed tool (default is "your_email@example.com").
        """
        self.pubmed = PubMed(tool="MyTool", email=email)

    @staticmethod
    def _remove_trailing_period(text):
        """
        Removes any trailing period from the text.

        Parameters
        ----------
        text : str
            The input text to remove the period from.

        Returns
        -------
        str
            The text without the trailing period.
        """
        return text.rstrip('.') if text else text

    def _parse_authors(self, authors):
        """
        Parses the author information into a formatted string.

        Parameters
        ----------
        authors : list of dict
            A list of authors, each represented by a dictionary with 'initials' and 'lastname' keys.

        Returns
        -------
        str
            A string of authors in the format "Initials. Lastname".
        """
        parsed_authors = []
        for author in authors:
            initials = author.get('initials') or author.get('firstname', "No authors")[0]
            last_name = author['lastname']
            parsed_authors.append(f"{initials}. {last_name}")
        return ", ".join(parsed_authors)

    def _parse_citation(self, article):
        """
        Generates a citation string for the given article.

        Parameters
        ----------
        article : Article
            The article object containing title, authors, journal, publication date, and DOI.

        Returns
        -------
        str
            A formatted citation string for the article.
        """
        title:str    = self._remove_trailing_period(article.title)
        authors:str  = self._parse_authors(article.authors)
        journal:str  = self._remove_trailing_period(article.journal)
        pubdate:str  = article.publication_date.strftime("%Y") if article.publication_date else "Unknown Date"
        doi:str      = article.doi.split("\n")[0] if hasattr(article, "doi") else ""
        
        citation = f"{authors}. {title}. {journal}. {pubdate}."
        if len(doi) > 2:
            citation += f" doi: {doi}"

        return citation

    def get_pubmed_reference(self, pmid, parse_citation=True):
        """
        Fetches and returns a PubMed article's details and citation.

        Parameters
        ----------
        pmid : str
            The PubMed ID (PMID) of the article to fetch.
        parse_citation : bool, optional
            Whether to parse and include the citation in the output (default is True).

        Returns
        -------
        dict or None
            A dictionary containing the article details and optional citation, or None if no article is found.
        """
        article = next(self.pubmed.query(pmid, max_results=1), None)
        
        if not article:
            return None
        
        article_dict = article.toDict()
        if parse_citation:
            article_dict["citation"] = self._parse_citation(article)
        
        return article_dict

if __name__ == "__main__":
    # Example usage
    pmid = '33792783'  # Replace with an actual PMID
    citation_tool = PubMedCitation(email="your_email@example.com")
    article = citation_tool.get_pubmed_reference(pmid)
    print(article)
