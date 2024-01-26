from src.brdata_rag_tools.databases import PGVector, FAISS
from src.brdata_rag_tools.embeddings import EmbeddingConfig

from sqlalchemy.orm import Mapped, mapped_column
from sqlalchemy import String, text

from pytest import fixture

@fixture
def remove_table():
    database = PGVector()
    database.drop_table("test")
    yield
    database.drop_table("test")

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

    podcast1 = Podcast(title="TRUE CRIME - Unter Verdacht",
                       id="1",
                       url = "example.com",
                       embedding_source="Wer wird hier zu Recht, wer zu Unrecht verdächtigt? Was, wenn Menschen unschuldig verurteilt werden und ihnen niemand glaubt? Oder andersherum: Wenn der wahre Täter oder die wahre Täterin ohne Strafe davonkommen? Unter Verdacht - In der 7. Staffel des erfolgreichen BAYERN 3 True Crime Podcasts sprechen Strafverteidiger Dr. Alexander Stevens und BAYERN 3 Moderatorin Jacqueline Belle über neue spannende Kriminalfälle. Diesmal geht es um Menschen, die unter Verdacht geraten sind. Wer ist schuldig? Wer lügt, wer sagt die Wahrheit? Und werden am Ende immer die Richtigen verurteilt?")

    database.write_rows([podcast1], create_embeddings=True)

    with database.session() as s:
        response = s.execute(text("SELECT count(1) as count FROM test;")).first()

    assert response.count == 1

    simcont = database.retrieve_similar_content(prompt = "Hallo Test.", embedding_type=EmbeddingConfig.TEST, table=Podcast)

    assert len(simcont) == 1
    assert isinstance(simcont[0], dict)
    assert simcont[0]["cosine_dist"] == 0


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

    podcast1 = Podcast(title="TRUE CRIME - Unter Verdacht",
                       id="1",
                       url = "example.com",
                       embedding_source="Wer wird hier zu Recht, wer zu Unrecht verdächtigt? Was, wenn Menschen unschuldig verurteilt werden und ihnen niemand glaubt? Oder andersherum: Wenn der wahre Täter oder die wahre Täterin ohne Strafe davonkommen? Unter Verdacht - In der 7. Staffel des erfolgreichen BAYERN 3 True Crime Podcasts sprechen Strafverteidiger Dr. Alexander Stevens und BAYERN 3 Moderatorin Jacqueline Belle über neue spannende Kriminalfälle. Diesmal geht es um Menschen, die unter Verdacht geraten sind. Wer ist schuldig? Wer lügt, wer sagt die Wahrheit? Und werden am Ende immer die Richtigen verurteilt?")

    database.write_rows([podcast1], create_embeddings=True)

    with database.session() as s:
        response = s.execute(text("SELECT count(1) FROM test;")).first()

    assert response.count == 1

    simcont = database.retrieve_similar_content(prompt = "Hallo Test.", embedding_type=EmbeddingConfig.TEST, table=Podcast)

    assert len(simcont) == 1
    assert isinstance(simcont[0], dict)
    assert simcont[0]["cosine_dist"] == 0
