from cached_path import set_cache_dir

from topical.common import get_cache_dir

# Set the cache directory globally
set_cache_dir(get_cache_dir())

# Check if the required NLTK resources are downloaded, and if not, download them
try:
    import nltk

    nltk.data.find("tokenizers/punkt")
    nltk.data.find("stopwords")
except LookupError:
    nltk.download("punkt", quiet=True)
    nltk.download("stopwords", quiet=True)
