import os
from enum import Enum
from typing import List, Optional, Type

import numpy as np
import requests

from .datastructures import BaseClass


class EmbeddingType(Enum):
    """
    The EmbeddingType class is an enumeration that represents different types of embeddings.
    It is used to define the type of embedding model to be used in a text processing task.

    ### Attributes:

    - SENTENCE_TRANSFORMERS (EmbeddingType): Represents the Sentence Transformers embedding model.
    - TF_IDF (EmbeddingType): Represents the TF-IDF embedding model.
    - TEST (EmbeddingType): Represents a test embedding model.

    Example Usage::

        import test

        embedding_type = EmbeddingType.SENTENCE_TRANSFORMERS
        dimension = embedding_type.dimension
        model = embedding_type.model

    """
    SENTENCE_TRANSFORMERS = "sentence_transformers"
    TF_IDF = "tfidf"
    TEST = "test"

    @property
    def dimension(self):
        """
        Get the dimension of the input.

        :return: The dimension of the input.
        :rtype: int
        """
        if self == self.SENTENCE_TRANSFORMERS:
            return SentenceTransformer().dimension
        if self == self.TF_IDF:
            return 768
        if self == self.TEST:
            return Test().dimension

    @property
    def model(self):
        """
        :return: An instance of the used language model.
        """
        if self == self.SENTENCE_TRANSFORMERS:
            return SentenceTransformer()
        if self == self.TF_IDF:
            raise NotImplementedError()
        if self == self.TEST:
            return Test()


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

    def __init__(self, endpoint: str, dimension: int, token: Optional[str] = None,
                 max_prompt_tokens: Optional[int] = None):
        self.endpoint = endpoint
        self.dimension = dimension
        self.token = token
        self.max_prompt_tokens = max_prompt_tokens

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
        super().__init__(endpoint="https://gbert-cosine-embeddings.brdata-dev.de", dimension=1024,
                         token=os.environ.get("SENTENCE_TRANSFORMER_TOKEN"))

    def create_embedding(self, text: str) -> np.array:
        """
        :param text: The text to be embedded.
        :return: A numpy array representing the embedding of the text.
        """
        headers = {
            'accept': 'application/json',
            'Content-Type': 'application/json',
            'Authorization': f'Bearer {self.token}'
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

        if self.token is None:
            raise ValueError("No Access token specified. Please set `SENTENCE_TRANSFORMER_TOKEN` env var.")

        headers = {
            'accept': 'application/json',
            'Content-Type': 'application/json',
            'Authorization': f'Bearer {self.token}'
        }

        content = []
        for row in rows:
            content.append({
                'id': row.id,
                'text': row.embedding_source,
            })

        json_data = {
            'content': content
        }

        response = requests.post(f'{self.endpoint}/create_embeddings',
                                 headers=headers, json=json_data).json()["content"]

        for i, text_element in enumerate(response):
            if rows[i].id == text_element["id"]:
                rows[i].embedding = np.array(text_element["embedding"])
            else:
                raise ValueError("ID mismatch.")

        return rows


class Test(Embedder):
    """
    This class represents a Test class that inherits from Embedder class.

    Args:
        None

    Attributes:
        None

    """

    def __init__(self, dimension: int = 3):
        super().__init__("example.com", dimension, None)

    def create_embedding_bulk(self, rows: List[Type[BaseClass]]) -> List[
        Type[BaseClass]]:
        """
        Create embeddings for a list of rows.

        :param rows: A list of rows, each row should be an instance of the BaseClass.
        :return: A list of rows with updated embeddings.
        :raises ValueError: If the endpoint of any row does not match the current endpoint.

        """
        rows_wo_embedding = [x for x in rows if x.embedding is None]
        rows_w_embedding = [x for x in rows if x.embedding is not None]

        for row in rows_wo_embedding:
            if row.embedding_type.model.endpoint != self.endpoint:
                raise ValueError(
                    f"Can't create an embedding with type {row.embedding_type.value} from endpoint {self.endpoint}.")
            row.embedding = np.array([1, 2, 3])

        return rows_w_embedding + rows_wo_embedding

    def create_embedding(self, text: str) -> np.array:
        """
        Create an embedding for the given text.

        :param text: The text to create an embedding for.
        :type text: str
        :return: A 3-dimensional numpy array with the values 1, 2 and 3.
        :rtype: numpy.array
        """
        return np.array([1, 2, 3])
        # return np.random.random(self.dimension).astype("float32")
