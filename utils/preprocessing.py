import re


def clean_text(text: str) -> str:
    """
    Basic preprocessing for regulatory PDFs
    """

    # remove multiple spaces
    text = re.sub(r"\s+", " ", text)

    # remove broken line breaks
    text = text.replace("\n", " ")

    # remove extra punctuation spacing
    text = re.sub(r"\s([?.!,](?:\s|$))", r"\1", text)

    return text.strip()