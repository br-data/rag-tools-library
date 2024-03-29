import os

from src.brdata_rag_tools.databases import PGVector, FAISS
from src.brdata_rag_tools.embeddings import EmbeddingConfig, Embedder, register

from sqlalchemy.orm import Mapped, mapped_column
from sqlalchemy import String, text
import numpy as np

from pytest import fixture


@fixture
def remove_table():
    database = PGVector()
    database.drop_table("test")
    yield
    database.drop_table("test")


class Test(Embedder):
    def __init__(self):
        super().__init__(endpoint="example.com", auth_token=None)

    def create_embedding_bulk(self, rows):
        """
        Takes an  list of SQLAlchemy Table classes as input and returns them with embeddings assigned.
        """
        for row in rows:
            row.embedding = self.create_embedding(row.embedding_source)

        return rows

    def create_embedding(self, text: str) -> np.array:
        if text == "test":
            return np.array([1, 2, 3])
        else:
            return np.array([4, 5, 6])


register(Test, name="test", dimension=3)


def test_faiss():
    database = FAISS()
    assert type(database) == FAISS

    abstract_table = database.create_abstract_embedding_table(EmbeddingConfig.TEST)
    assert len(set(abstract_table.__annotations__.keys()) & set(["id", "embedding_source", "embedding"])) == 3

    class Podcast(abstract_table):
        __tablename__ = "test"
        title: Mapped[str] = mapped_column(String)
        url: Mapped[str] = mapped_column(String)

    database.create_tables()
    assert "test" in list(database.metadata.tables.keys())

    podcasts = []

    for i in range(3):
        podcasts.append(Podcast(title="TRUE CRIME - Unter Verdacht",
                                id=i,
                                url="example.com",
                                embedding_source="test")
                        )

    podcasts.append(Podcast(title="TRUE CRIME - Unter Verdacht",
                            id=4,
                            url="example.com",
                            embedding_source="Different Vector")
                    )

    database.write_rows(podcasts, create_embeddings=True)

    with database.session() as s:
        response = s.execute(text("SELECT count(1) as count FROM test;")).first()

    assert response.count == 4

    simcont = database.retrieve_similar_content(prompt="different vector.", embedding_type=EmbeddingConfig.TEST,
                                                table=Podcast, max_dist=.05)

    assert len(simcont) == 1  # filters out 3 results
    assert isinstance(simcont[0], dict)
    assert simcont[0]["cosine_dist"] == 0

def test_faiss_persistence():
    db_path = "test/data/faiss.db"
    database = FAISS(db_path)
    assert type(database) == FAISS

    abstract_table = database.create_abstract_embedding_table(EmbeddingConfig.TEST)
    assert len(set(abstract_table.__annotations__.keys()) & set(["id", "embedding_source", "embedding"])) == 3

    class Podcast(abstract_table):
        __tablename__ = "test"
        title: Mapped[str] = mapped_column(String)
        url: Mapped[str] = mapped_column(String)

    database.create_tables()
    assert "test" in list(database.metadata.tables.keys())

    podcasts = []

    for i in range(3):
        podcasts.append(Podcast(title="TRUE CRIME - Unter Verdacht",
                                id=i,
                                url="example.com",
                                embedding_source="test")
                        )
    print("bla")
    database.write_index()

    assert "Podcast.index" in os.listdir(db_path) and "Podcast.mapping" in os.listdir(db_path)

    database2 = FAISS(db_path)

    assert database2 == database



def test_pgvector(remove_table):
    database = PGVector()
    assert type(database) == PGVector

    abstract_table = database.create_abstract_embedding_table(EmbeddingConfig.TEST)
    assert len(set(abstract_table.__annotations__.keys()) & set(["id", "embedding_source", "embedding"])) == 3

    class Podcast(abstract_table):
        __tablename__ = "test"
        title: Mapped[str] = mapped_column(String)
        url: Mapped[str] = mapped_column(String)

    database.create_tables()
    assert "test" in list(database.metadata.tables.keys())
    podcasts = []

    for i in range(3):
        podcasts.append(Podcast(title="TRUE CRIME - Unter Verdacht",
                                id=i,
                                url="example.com",
                                embedding_source="test")
                        )

    podcasts.append(Podcast(title="TRUE CRIME - Unter Verdacht",
                            id=4,
                            url="example.com",
                            embedding_source="Different Vector")
                    )

    database.write_rows(podcasts, create_embeddings=True)

    with database.session() as s:
        response = s.execute(text("SELECT count(1) FROM test;")).first()

    assert response.count == 4

    simcont = database.retrieve_similar_content(prompt="Hallo Test.", embedding_type=EmbeddingConfig.TEST,
                                                table=Podcast, max_dist=.02)

    assert len(simcont) == 1
    assert isinstance(simcont[0], dict)
    assert simcont[0]["cosine_dist"] == 0
