from sqlalchemy import String
from sqlalchemy.orm import Mapped, mapped_column

from databases import PGVector, FAISS
from embeddings import EmbeddingType
from models import LLMName
from models import LLM

if __name__ == "__main__":

    # load model
    language_model = LLM(model_name=LLMName.GPT35TURBO)
    embed_type = EmbeddingType.SENTENCE_TRANSFORMERS
    # set up database
    database = FAISS()
    database.drop_table("podcast")
    EmbeddingTable = database.create_abstract_embedding_table(embed_type=embed_type)

    class Podcast(EmbeddingTable):
        __tablename__ = "podcast"
        title: Mapped[str] = mapped_column(String)
        url: Mapped[str] = mapped_column(String)

    database.create_tables()

    # load stuff into db

    podcast1 = Podcast(title="TRUE CRIME - Unter Verdacht",
                       id="1",
                       url = "example.com",
                       embedding_source="Wer wird hier zu Recht, wer zu Unrecht verdächtigt? Was, wenn Menschen unschuldig verurteilt werden und ihnen niemand glaubt? Oder andersherum: Wenn der wahre Täter oder die wahre Täterin ohne Strafe davonkommen? Unter Verdacht - In der 7. Staffel des erfolgreichen BAYERN 3 True Crime Podcasts sprechen Strafverteidiger Dr. Alexander Stevens und BAYERN 3 Moderatorin Jacqueline Belle über neue spannende Kriminalfälle. Diesmal geht es um Menschen, die unter Verdacht geraten sind. Wer ist schuldig? Wer lügt, wer sagt die Wahrheit? Und werden am Ende immer die Richtigen verurteilt?")
    
    podcast2 = Podcast(title="SCHOENHOLTZ - Der Orchester-Podcast",
                       id="2",
                       url = "br24.de",
                       embedding_source="Wie funktioniert ein Orchester? Wie kommt man da rein? Und wieso tragen Orchestermusiker*innen eigentlich immer nur schwarz? Wer könnte diese Fragen besser beantworten als eine Orchestermusikerin selbst! Anne Schoenholtz ist Geigerin im Symphonieorchester des Bayerischen Rundfunks, kurz BRSO - einem Orchester, das gerade zum drittbesten der Welt gewählt wurde. Als Host des Orchesterpodcasts nimmt Anne uns mit hinter die Kulissen des BRSO und entlockt ihren Kolleg*innen so manches intime Geständnis und auch witzige Geschichten aus dem Orchesterleben. In der dritten Staffel klären wir, wie Konzertprogramme überhaupt zustandekommen, welche Wehwehchen die Musiker*innen typischerweise plagen und warum es über die Bratschen so viele Witze gibt. Sir Simon Rattle, der neue Chefdirigent des BRSO, beantwortet am Ende jeder Folge Fragen aus der Community.")

    podcast3 = Podcast(title = "Tatort Geschichte – True Crime meets History",
                       id = "3",
                       url = "bla.com",
                       embedding_source = "Bei Tatort Geschichte verlassen Niklas Fischer und Hannes Liebrandt, zwei Historiker von der Ludwig-Maximilians-Universität in München, den Hörsaal und reisen zurück zu spannenden Verbrechen aus der Vergangenheit: eine mysteriöse Wasserleiche im Berliner Landwehrkanal, der junge Stalin als Anführer eines blutigen Raubüberfalls oder die Jagd nach einem Kriegsverbrecher um die halbe Welt. True Crime aus der Geschichte unterhaltsam besprochen. Im Fokus steht die Frage, was das eigentlich mit uns heute zu tun hat. Tatort Geschichte ist ein Podcast von Bayern 2 in Zusammenarbeit mit der Georg-von-Vollmar-Akademie.")

    database.write_rows([podcast1, podcast2, podcast3], create_embeddings = True)

    # create the prompt and template
    user_prompt = "Gib mir spannende Podcasts zum Thema Musik."

    prompt_template = ("Du bist der Podcast-Experte des Bayerischen Rundfunks. "
                       "Eine Userin stellt dir folgende Frage:\n"
                       "{}\n"
                       "Dazu fallen dir folgende Podcasts ein, die du ihr empfehlen kannst:\n"
                       "-{}\n"
                       "Beschränke dich auf die Auswahl der Podcasts und erfinde keine neuen Podcasts dazu. Empfiehl der"
                       "Userin ausschließlich einen Podcasts aus der Liste und begründe deine Entscheidung kurz."
                       "Schreibe in der zweiten Person, sprich die Userin also direkt an.")

    context = database.retrieve_similar_content(user_prompt,
                                                Podcast,
                                                embed_type, limit=2)

    context = [x["Podcast"].title + ": " + x["Podcast"].embedding_source for x in context]

    context = language_model.model.fit_to_context_window(prompt_template + user_prompt, context)

    prompt = prompt_template.format(user_prompt, "\n-".join(context))
    print(prompt + "\nModel Response\n==============\n\n")
    response = language_model.prompt(prompt)
    print(response)