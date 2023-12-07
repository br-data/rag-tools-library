import os

from sqlalchemy import String
from sqlalchemy.orm import Mapped, mapped_column

from databases import PGVector
from embeddings import EmbeddingType
from models import LLMName
from models import LLM

if __name__ == "__main__":
    # load model
    language_model = LLM(model_name=LLMName.IGEL)
    embed_type = EmbeddingType.SENTENCE_TRANSFORMERS
    # set up database
    database = PGVector(password=os.environ.get("DATABASE_PASSWORD"))
    database.drop_table("podcast")
    EmbeddingTable = database.create_abstract_embedding_table(embed_type=embed_type)

    class Podcast(EmbeddingTable):
        __tablename__ = "podcast"
        text: Mapped[str] = mapped_column(String)
        title: Mapped[str] = mapped_column(String)
        url: Mapped[str] = mapped_column(String)

    database.create_tables()

    # load stuff into db
    podcast1 = Podcast(text="Das ist der Beschreibungstext.",
                       title="Das ist der Titel",
                       id="1",
                       url = "example.com",
                       embedding_source="Daraus generieren wir das Embedding.")

    podcast2 = Podcast(text="Das ist der Beschreibungstext.",
                       title="Das ist der Titel",
                       id="2",
                       url = "br24.de",
                       embedding_source="Daraus generieren wir das Embedding.")

    database.write_rows([podcast1, podcast2], create_embeddings = True)

    # create the prompt and template
    user_prompt = "Gib mir spannende Podcasts zum Thema True Crime."

    prompt_template = ("Du bist der Podcast-Experte des Bayerischen Rundfunks. "
                       "Eine Userin stellt dir folgende Frage:\n"
                       "{}\n"
                       "Dazu fallen dir folgende Podcasts ein, die du ihr empfehlen kannst:\n"
                       "-{}")

    context = database.retrieve_similar_content(user_prompt,
                                                Podcast,
                                                embed_type, limit=5)

    context = [x[0].text for x in context]

    prompt = prompt_template.format(user_prompt, "\n-".join(context))
    response = language_model.prompt(prompt)
    print(response)