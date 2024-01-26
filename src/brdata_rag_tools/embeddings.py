import os
from aenum import Enum, extend_enum
from typing import List, Optional, Type

import numpy as np
import requests
import logging

from .datastructures import BaseClass

embedding_data = {
    "sentence_transformers": {
        "dimension": 1024
    },
}

user_models = {}


def register(model, name=None, dimension: int = None):
    extend_enum(EmbeddingConfig, name.upper(), name)
    user_models[name] = model

    embedding_data[name] = {
        "dimension": dimension
    }

    logging.info(f"Successfully registered model {name}.")


class EmbeddingConfig(Enum):
    """
    The EmbeddingConfig class is an enumeration that represents different types of embeddings.
    It is used to define the type of embedding model to be used in a text processing task.

    ### Attributes:

    - SENTENCE_TRANSFORMERS (EmbeddingConfig): Represents the Sentence Transformers embedding model.
    - TF_IDF (EmbeddingConfig): Represents the TF-IDF embedding model.

    Example Usage::

        import test

        embedding_type = EmbeddingConfig.SENTENCE_TRANSFORMERS
        dimension = embedding_type.dimension
        model = embedding_type.model

    """
    SENTENCE_TRANSFORMERS = "sentence_transformers"
    TF_IDF = "tfidf"

    @property
    def dimension(self):
        """
        Get the dimension of the input.

        :return: The dimension of the input.
        :rtype: int
        """
        return embedding_data[self.value]["dimension"]

    @property
    def model(self):
        """
        :return: An instance of the used embedding model.
        """
        if self == self.SENTENCE_TRANSFORMERS:
            return SentenceTransformer()
        elif self == self.TF_IDF:
            raise NotImplementedError()
        else:
            try:
                return user_models[self.value]()
            except KeyError:
                raise KeyError(f"No embedding model with name {self.value} found.")


class Embedder:
    """
    The Embedder class is responsible for creating embeddings for a list of objects.

    :param endpoint: The endpoint for the embedding service.
    :type endpoint: str
    :param dimension: The dimension of the embeddings to be created.
    :type dimension: int
    :param token: The access auth_token for the embedding service (optional).
    :type token: str, optional

    :raises: NotImplementedError

    :return: None
    """

    def __init__(self, endpoint: str, auth_token: Optional[str] = None):
        self.endpoint = endpoint
        self.auth_token = auth_token

    def create_embedding_bulk(self, rows: List[Type[BaseClass]]) -> List[
        Type[BaseClass]]:
        raise NotImplementedError()

    def create_embedding(self, text: str) -> np.array:
        raise NotImplementedError()


class SentenceTransformer(Embedder):
    """
    Class representing a sentence transformer for creating sentence embeddings.

    Methods:
        - create_embedding(text: str) -> np.array:
            Takes a string of text and returns its corresponding sentence embedding.

        - create_embedding_bulk(rows: List[Type[BaseClass]]) -> List[Type[BaseClass]]:
            Takes a list of rows as input and creates sentence embeddings for each row. It updates
            the 'embedding' attribute of each row in the list and returns the updated list.

        Raises:
            - NotImplementedError: Raised when more than 250 rows are passed to the create_embedding_bulk method.
            - ValueError: Raised when there is an ID mismatch during the bulk embedding creation process.
    """

    def __init__(self):
        super().__init__(endpoint=os.environ.get("SENTENCE_TRANSFORMER_ENDPOINT", "https://gbert-cosine-embeddings.brdata-dev.de"),
                         auth_token=os.environ.get("SENTENCE_TRANSFORMER_TOKEN"))

        if self.auth_token is None:
            logging.info("No auth token for Sentence Transformer was provided.")

    def create_embedding(self, text: str) -> np.array:
        """
        :param text: The text to be embedded.
        :return: A numpy array representing the embedding of the text.
        """
        headers = {
            'accept': 'application/json',
            'Content-Type': 'application/json',
            'Authorization': f'Bearer {self.auth_token}'
        }

        json_data = {
            'content': [{
                'id': '1',
                'text': text,
            }]
        }

        response = requests.post(f'{self.endpoint}/create_embeddings',
                                 headers=headers, json=json_data).json()["content"]

        return np.array(response[0]["embedding"])

    def create_embedding_bulk(self, rows: List[Type[BaseClass]]) -> List[
        Type[BaseClass]]:
        """
        Create embeddings for a list of rows.

        :param rows: A list of rows to create embeddings for.
        :type rows: List[Type[BaseClass]]
        :return: A list of rows with embeddings added.
        :rtype: List[Type[BaseClass]]
        :raises NotImplementedError: If the number of rows is greater than 250.
        :raises ValueError: If there is a mismatch in ID between the rows and the response.
        """
        if len(rows) > 250:
            raise NotImplementedError(
                "Sorry, only 250 rows may be processed at one time.")

        if self.auth_token is None:
            raise ValueError(
                "No Access auth_token specified. Please set `SENTENCE_TRANSFORMER_TOKEN` environment variable.")

        headers = {
            'accept': 'application/json',
            'Content-Type': 'application/json',
            'Authorization': f'Bearer {self.auth_token}'
        }

        content = []
        for row in rows:

            if row.id is None or row.embedding_source is None:
                raise ValueError(
                    "ID or embedding source is missing. Please provide an ID or embedding source in your table"
                    "definition as described here with the declaration of podcast1: "
                    "https://br-data.github.io/rag-tools-library/#augmenting-your-prompt.")

            content.append({
                'id': row.id,
                'text': row.embedding_source,
            })

        json_data = {
            'content': content
        }

        response = requests.post(f'{self.endpoint}/create_embeddings',
                                 headers=headers, json=json_data)

        if response.status_code != 200:
            raise ConnectionError(f"Connection to the Embedder API returned status code {response.status_code}.")

        try:
            response = response.json()["content"]
        except KeyError:
            raise ValueError(
                "The Embedding service did not return any results. Please make sure your auth_token is correct"
                " and embedding_source is of a meaningful format.")

        for i, text_element in enumerate(response):
            if rows[i].id == text_element["id"]:
                rows[i].embedding = np.array(text_element["embedding"])
            else:
                raise ValueError("ID mismatch.")

        return rows


