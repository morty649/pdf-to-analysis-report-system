import re


def clean_text(text: str) -> str:
    """
    Basic preprocessing for regulatory PDFs
    """

    # removes multiple spaces
    text = re.sub(r"\s+", " ", text)

    # removes broken line breaks
    text = text.replace("\n", " ")

    # removes extra punctuation spacing
    text = re.sub(r"\s([?.!,](?:\s|$))", r"\1", text)

    return text.strip()