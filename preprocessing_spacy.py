import os
import spacy


class spaCyModel:
    def __init__(self, model_name = 'en_core_web_sm') -> None:
        self.model = spacy.load(model_name)

    def process(self, doc):
        doc = self.model(doc)
        tokens = [token.lemma_ for token in doc if not token.is_stop and not token.is_digit and not token.is_quote and not token.is_space and not token.is_bracket and not token.is_currency and not token.is_punct and not token.is_left_punct and not token.is_right_punct]
        # print(tokens)
        str = ""
        for index, token in enumerate(tokens):
            if index == 0:
                str = token
            else:
                str = str + " " + token
        return str


# Preprocess the documents using spaCy
def preprocess(nlp, doc):
    # Tokenize and preprocess each document
    doc = nlp(doc)
    # Lemmatize and remove stop words
    tokens = [token.lemma_ for token in doc if not token.is_stop and not token.is_digit and not token.is_quote and not token.is_space and not token.is_bracket and not token.is_currency and not token.is_punct and not token.is_left_punct and not token.is_right_punct]
    # print(tokens)
    str = ""
    for index, token in enumerate(tokens):
        if index == 0:
            str = token
        else:
            str = str + " " + token
    return str


if __name__ == "__main__":
    # Load spaCy model
    # nlp = spacy.load("en_core_web_sm")
    # nlp = spacy.load("en_core_web_md")
    # nlp = spacy.load("en_core_web_lg")
    nlp = spacy.load("de_core_news_sm")
    # nlp = spacy.load("de_core_news_md")
    # nlp = spacy.load("de_core_news_lg")

    # Sample documents
    documents = [
        "Tambora Vulkanausbruch 1815: Ein Jahr ohne Sommer - Indojunkie",
        "Coconut-Talk: Bali Serie - Traditionelles Familienleben mit Katha aus Bali - Indojunkie",
        "Echte Homestays auf Bali mit HSH Stay finden - Indojunkie",
        "Togian Inseln Sulawesi: Auf zu neuen Ufern",
        "Into the Wild – Top 10 Bali Dschungel-Unterkünfte - Indojunkie"
    ]

    # Apply preprocessing to each document
    processed_docs = [preprocess(nlp, doc) for doc in documents]
    print(processed_docs)


