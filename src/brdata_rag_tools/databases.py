import os
from typing import List, Type

import numpy as np
from pgvector.sqlalchemy import Vector
from sqlalchemy import String, text
from sqlalchemy import create_engine, select
from sqlalchemy.orm import Session, Mapped, mapped_column

from datastructures import Base, BaseClass
from embeddings import EmbeddingType


class PGVector:
    """
    This class represents a connection to a PostgreSQL database with the pgvector addon.

    :param user: The username to connect to the database. Default is "postgres".
    :type user: str
    :param database: The name of the database to connect to. Default is "postgres".
    :type database: str
    :param password: The password to connect to the database. If not provided, it will use the value of the "DATABASE_PASSWORD" environment variable.
    :type password: str
    :param host: The host address of the database. Default is "localhost".
    :type host: str
    :param port: The port number of the database. Default is 5432.
    :type port: int
    :param verbose: Whether to enable verbose output. Default is False.
    :type verbose: bool
    """

    def __init__(self, user: str = "postgres", database: str = "postgres",
                 password: str = None, host: str = "localhost", port: int = 5432,
                 verbose: bool = False):
        self.user = user
        self.database = database
        self.password = os.environ.get(
            "DATABASE_PASSWORD") if password is None else password
        self.host = host
        self.port = port
        self.verbose = verbose
        self.engine = self._create_engine()
        self.embedding_table = None
        self.Base = Base
        self.metadata = Base.metadata

    def create_abstract_embedding_table(self, embed_type: EmbeddingType):
        """
        Create an abstract embedding table. Use it to inherit your own database tables to use with the PGVector class.

        :param embed_type: The type of embedding to be used.
        :return: The abstract embedding table class.
        """

        class EmbeddingTable(self.Base):
            embedding_type = embed_type

            __abstract__ = True
            id: Mapped[str] = mapped_column(primary_key=True, unique=True)
            embedding_source: Mapped[str] = mapped_column(String)
            embedding: Mapped[np.array] = mapped_column(
                Vector(embedding_type.dimension))

            def __repr__(self):
                return self.id + ": " + self.embedding_source

        return EmbeddingTable

    def create_tables(self):
        """
        Create tables in the database based on the defined metadata.

        :return: None
        """
        self.metadata.create_all(self.engine)

    def _create_engine(self):
        """
        Create a SQLAlchemy engine instance for connecting to a PostgreSQL database.

        :return: SQLAlchemy engine instance
        """
        return create_engine(
            f"postgresql+psycopg2://{self.user}:{self.password}@{self.host}/{self.database}",
            echo=self.verbose)

    def session(self):
        """
        Initiates a new session with the database.

        :return: Session object
        """
        session = Session(self.engine)
        session.execute(text('CREATE EXTENSION IF NOT EXISTS vector;'))
        return session

    def drop_table(self, name: str):
        """
        Drop a table from the database.

        :param name: The name of the table to be dropped.
        :type name: str
        :return: None
        """
        with self.session() as session:
            session.execute(text(f"DROP TABLE {name};"))
            session.commit()

    def retrieve_similar_content(self, prompt, table: Type[BaseClass],
                                 embedding_type: EmbeddingType, limit: int = 50):
        """
        Retrieve similar content based on a prompt. The function creates an embedding with the specified embedding type
        and queries the associated database for the most similar matches.

        :param prompt: The prompt for which similar content needs to be found.
        :param table: The table in which the content is stored.
        :param embedding_type: The type of embedding to be used.
        :param limit: The maximum number of similar content to be retrieved (default: 50).
        :return: A list of results containing similar content.
        """
        embedder = embedding_type.model
        prompt_embedding = embedder.create_embedding(prompt)

        with self.session() as session:
            results = session.execute(select(table, table.embedding.cosine_distance(
                prompt_embedding).label("cosine_dist")).limit(limit)).all()

        return results

    def get_ids_with_embedding(self, table: BaseClass):
        """
        Get the IDs from the given table using the provided embedding.

        :param table: The table to query the IDs from.
        :type table: BaseClass

        :return: A list of IDs from the provided table.
        :rtype: List[int]
        """
        with self.session() as session:
            ids = session.query(table.id).all()

        return [x.id for x in ids]

    def retrieve_embedding(self, row_id: str, table: BaseClass) -> np.array:
        """
        Retrieve embedding from the database for a given row ID and table.

        :param row_id: The ID of the row to retrieve the embedding from.
        :param table: The table containing the embedding column.
        :return: The retrieved embedding as a numpy array.

        """
        with self.session() as session:
            embedding = session.query(table.emebedding).where(
                table.id == row_id).first()

        return embedding

    def write_rows(self, rows: List[Type[BaseClass]], create_embeddings: bool = True):
        """
        Write rows to the database and optionally create embeddings for the rows.

        :param rows: A list of rows to be written to the database. Rows must be instances of BaseClass or its subclasses.
        :param create_embeddings: A boolean value indicating whether embeddings should be created for the rows.
                                  Default value is True.
        :return: None
        """
        table = type(rows[0])
        embedder = rows[0].embedding_type.model

        rows_with_embedding = self.get_ids_with_embedding(table)
        rows_wo_embedding = [x for x in rows if x.id not in rows_with_embedding]

        rows_wo_embedding = embedder.create_embedding_bulk(rows_wo_embedding)

        with self.session() as session:
            session.add_all(rows_wo_embedding)
            session.flush()
            session.commit()
