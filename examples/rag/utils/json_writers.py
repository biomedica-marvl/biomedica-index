import os 
import json

def pretty_print_store(store,verbose:bool=False):
    """Pretty prints the STORE dictionary."""
    formatted_store = {
        "Queried Subsets":    store.get("queried_subsets", None),
        "Number of Articles": store.get("n_articles", None),
        "Question": store.get("question", None),
        "Answer":   store.get("parsed_answer", None),
     
        
    }
    print(json.dumps(formatted_store, indent=4, ensure_ascii=False))


def update_json_file(file_path, key, value):
    """
    Updates a JSON file with the given key-value pair.
    If the file does not exist, it creates a new one.
    If the key does not exist, it adds it.
    
    :param file_path: Path to the JSON file.
    :param key: The key to update or add.
    :param value: The value associated with the key.
    """

    # Extract the directory path from the filename
    directory = os.path.dirname(file_path)
    
    # Create the directory if it doesn't exist
    if directory and not os.path.exists(directory):
        os.makedirs(directory, exist_ok=True)
        
    # Check if the file exists, if not, create an empty dictionary
    if os.path.exists(file_path):
        with open(file_path, 'r', encoding='utf-8') as file:
            try:
                data = json.load(file)  # Load existing JSON
            except json.JSONDecodeError:
                data = {}  # If file is empty or corrupt, start fresh
    else:
        data = {}

    # Add or update the key-value pair
    if key not in data:
        data[key] = value
        with open(file_path, 'w', encoding='utf-8') as file:
            json.dump(data, file, indent=4, ensure_ascii=False)
        print(f"Added '{key}' to {file_path}")
    else:
        print(f"'{key}' already exists in {file_path}, no changes made.")


def read_jsonl(filename: str) -> list[dict[str,str]]:
    """Reads a JSONL file and returns a list of dictionaries."""
    with open(filename, 'r', encoding='utf-8') as f:
        return [json.loads(line) for line in f]
        
def save_jsonl(filename: str, data: list[dict[str,str]]) -> None:
    """
    Saves a list of dictionaries to a JSONL file, creating any necessary subdirectories.

    Args:
        filename (str): The path to the JSONL file where data will be saved.
        data (List[Dict]): A list of dictionaries to be written to the JSONL file.
    """
    # Extract the directory path from the filename
    directory = os.path.dirname(filename)
    
    # Create the directory if it doesn't exist
    if directory and not os.path.exists(directory):
        os.makedirs(directory, exist_ok=True)
    
    # Write data to the JSONL file
    with open(filename, 'w', encoding='utf-8') as f:
        for entry in data:
            json.dump(entry, f)
            f.write('\n')


