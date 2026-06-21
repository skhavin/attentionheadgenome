# Helper to load WikiText-103 and group rows into proper Wikipedia articles.
# WikiText stores one line per row — articles are separated by "= Title =" headings.

from datasets import load_dataset
from config import DATASET_NAME, DATASET_CONFIG

def load_articles(split="train", max_articles=None):
    """Load WikiText-103 and group rows into full articles.
    Returns a list of strings, each string is one full Wikipedia article."""
    ds = load_dataset(DATASET_NAME, DATASET_CONFIG, split=split)

    articles = []
    current = []

    for row in ds:
        text = row["text"].strip()
        # Top-level heading (exactly one '=' on each side) = new article
        if text.startswith("= ") and text.endswith(" =") and text.count("=") == 2:
            if current:
                articles.append(" ".join(current))
            current = [text]
        else:
            if text:
                current.append(text)

    # Don't forget the last article
    if current:
        articles.append(" ".join(current))

    # Filter out very short articles (less than 100 chars)
    articles = [a for a in articles if len(a) > 100]

    if max_articles:
        articles = articles[:max_articles]

    print(f"Loaded {len(articles)} articles from {split} split")
    return articles


def load_concatenated_articles(split="validation", articles_per_doc=10, max_docs=None):
    """Concatenate multiple articles into longer documents (~4600 tokens each).
    This creates documents long enough that budget 512 still means real pruning."""
    articles = load_articles(split=split)

    long_docs = []
    for i in range(0, len(articles) - articles_per_doc + 1, articles_per_doc):
        chunk = articles[i:i + articles_per_doc]
        long_docs.append("\n\n".join(chunk))

    if max_docs:
        long_docs = long_docs[:max_docs]

    print(f"Created {len(long_docs)} concatenated docs ({articles_per_doc} articles each)")
    return long_docs

