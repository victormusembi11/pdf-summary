import tiktoken


def num_tokens_from_string(string: str, encoding_name: str) -> int:
    """Returns the number of tokens in a string."""
    encoding = tiktoken.get_encoding(encoding_name)
    print(encoding.encode(string))
    num_tokens = len(encoding.encode(string))
    return num_tokens


num_tokens_from_string("Hello, world!", "cl100k_base")
