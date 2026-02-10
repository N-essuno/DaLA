import ssl
import nltk

# Bypass SSL verification for NLTK downloads
try:
    _create_unverified_https_context = ssl._create_unverified_context
except AttributeError:
    pass
else:
    ssl._create_default_https_context = _create_unverified_https_context

# Download required NLTK data
nltk.download('punkt_tab')
nltk.download('wordnet')
nltk.download('omw-1.4')
