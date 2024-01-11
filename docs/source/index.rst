.. brdata-rag-tools documentation main file, created by
sphinx-quickstart on Thu Dec  7 18:15:58 2023.
You can adapt this file completely to your liking, but it should at least
contain the root `toctree` directive.

.. highlight:: python

Welcome to brdata-rag-tools's documentation!
============================================

.. toctree::
   :maxdepth: 2
   :caption: Contents:


Tutorial
========

Introduction:
-------------

Welcome to the brdata-rag-tools tutorial. In this brief introduction, I will guide you through using the library with a simple example.

Installation:
-------------

To install the package run:

.. code-block:: bash

    # create virtual environment
    python3 -m venv venv

    # Activate the virtual environment
    source venv/bin/activate

    # install the package
    python3 -m pip install brdata-rag-tools


Basic Usage
-----------

Choose the language model (LLM) you want by initiating the LLM class with a value from the LLMConfig enum.

All LLMs from brdata-rag-tools are connected via an API. The library serves as a wrapper to make the models more easily
accessible.

.. code-block:: python

   from brdata_rag_tools.models import LLM, LLMConfig

   llm = LLM(model_name=LLMConfig.GPT35TURBO)

Select the desired LLM, such as GPT 3.5 Turbo, GPT 3, GPT 4, IGEL, or Google's Bison models.

All GPT models may be used by anyone with an API token, the IGEL and Bison Model is only accessible from BR Data's
infrastructure.

Next we set the environment with OpenAI's access token:

.. code-block:: python

   os.environ["OPENAI_TOKEN"] = "YOUR TOKEN HERE"

For IGEL set the env var "IGEL_TOKEN" or "GOOGLE_TOKEN" for the Bison model respectively.

The LLM class holds merely the token and the actual model class.

The model class holds the actual logic and connections to interact with the language model endpoints.

Now, interact with the model using the `prompt` method:

.. code-block:: python

   joke = llm.prompt("Please tell me a joke.")
   print(joke)

Chat with the model
-------------------

Models of the GPT Family also support chat functionality – this means that models are aware of prompts sent earlier to
the model. Use the chat method.

.. code-block:: python

   answer = llm.chat("Please return 'test')
   print(answer)
   answer = llm.chat("What did I tell you in the last message?")
   print(answer)

For a new chat and to make the model forget earlier messages, use the `new_chat` method:

.. code-block:: python

   llm.new_chat()
   answer = llm.chat("What did I tell you in the last message?")
   print(answer)

Databases
----------
We do not only want to talk to our LLM, we want to augment it's prompt. This means we want to query a database for
relevant content.

This is done using so called semantic, or vector, search. For semantic search the searchable content is transformed
into a numerical representation, a vector embedding.

To retrieve relevant content, the user's prompt is also transformed into a vector. The prompt-vector is then compared
to all vectors in the database and the most similar vectors are retrieved.

You can choose of two different database types:

1. PgVector a database based on Postgres with an extension for vector search. It is a good choice to use if you plan to build production services. You need to deploy the database yourself.
2. SQLite with FAISS is a good choice if you want to try out something. While FAISS is a very capable library the usage in this library is not optimized for production.

FAISS and SQLite
~~~~~~~~~~~~~~~~

Create your database by importing and invoking it. Without any parameters it will be a memory-only database.
This means, if you stop your program, the data will be lost.

.. code-block:: python
    from brdata_rag_tools.databases import FAISS
    database = FAISS()

To write your database to disk, use the database parameter and pass it the path to your database file. If it does not
exist, it will be created in the specified directory. You may use absolute or relative paths.

.. code-block:: python

    database = FAISS(database="FAISS.db")


PGVector
~~~~~~~~

The easiest way to run it, if you're not on the BR data infrastructure, is to run it via docker.

The following command is not safe for a production environment! Don't use trust mode in critical applications.

.. code-block:: bash

   docker run -p 5432:5432 -e POSTGRES_HOST_AUTH_METHOD=trust ankane/pgvector

Connect to the database with `psql` if you want to make sure pgvector is up and running

.. code-block:: bash

   psql -U postgres -h localhost -p 5432


Follow the instructions to set a password or trust all hosts.

If you're on the BR data infrastructure, simply add pgvector as database type to your project's config.yaml file and
forward port 5432 to localhost.

Once you have your instance of pgvector running instance the PGVector class and supply it with the database's password.

.. code-block:: python

   from brdata_rag_tools.databases import PGVector
   database = PGVector(password="PASSWORD")

Populate your database
----------------------

To search for relevant content, you first need to ingest it in the database.

Therefore you need a table in the database to ingest your data. You get a bare minimum of such a table with the
following method:

.. code-block:: python

   embeding_table = database.create_abstract_embedding_table()

This method returns an abstract Database Table. Those table always contain the following Columns:

- id (string)
- embedding (Vector)
- embedding_source (string)

The Embedding Type
~~~~~~~~~~~~~~~~~~

The embedding column will be generated by the Database from the content in embedding_source. The id needs to be unique
for each row.

To actually use it, you need to inherit from the abstract table. In the following example, we will use our little search
for podcast recommendations.

The table needs to know which kind of embedding you want to use. The most universal embedding type is Sentence
Transformers, which is fine tuned for cosine similarity comparison of German texts.

.. code-block:: python

   from brdata_rag_tools.embeddings import EmbeddingConfig

   embedding = EmbeddingConfig.SENTENCE_TRANSFORMERS

   embeding_table = database.create_abstract_embedding_table(embed_type=embedding)

The database table
~~~~~~~~~~~~~~~~~~

The returned abstract table is an SQLAlchemy table object. You may add your own Columns to it to store data additional
to the three aforementioned items.

Give the table any name you like using the __tablename__ attribute. This is the only necessary field. Other columns,
like title and url in the example above, are introduced using the SQLAlchemy logic.

For more information on this topic, please refer to the `SQLAlchemy Tutorial <https://docs.sqlalchemy.org/en/20/tutorial/metadata.html#declaring-mapped-classes>`_. A list of types to use in your mapped_column attributes is available `here <https://docs.sqlalchemy.org/en/20/core/type_basics.html#generic-camelcase-types>`_.

Next, create the tables in the database:

.. code-block:: python

   class Podcast(embedding_table):
       __tablename__ = "podcast"
       title: Mapped[str] = mapped_column(String)
       url: Mapped[str] = mapped_column(String)

   # Create tables
   database.create_tables()

   # Fill with content
   podcast1 = Podcast(title="TRUE CRIME - Under Suspicion",
                      id="1",
                      url="example.com",
                      embedding_source="Who is rightfully, who is wrongly suspected here? What if people are wrongly convicted, and no one believes them? Or vice versa: If the true perpetrator goes unpunished? Under Suspicion - In the 7th season of the successful BAYERN 3 True Crime Podcast, defense attorney Dr. Alexander Stevens and BAYERN 3 host Jacqueline Belle discuss new exciting criminal cases. This time, it's about people who have come under suspicion. Who is guilty? Who is lying, who is telling the truth? And in the end, are the right ones always convicted?")

   podcast2 = Podcast(title="SCHOENHOLTZ - The Orchestra Podcast",
                      id="2",
                      url="br24.de",
                      embedding_source="How does an orchestra work? How do you get in? And why do orchestra musicians always wear black? Who could answer these questions better than an orchestra musician herself! Anne Schoenholtz is a violinist in the Bavarian Radio Symphony Orchestra, BRSO - an orchestra that has just been voted the third best in the world. As the host of the orchestra podcast, Anne takes us behind the scenes of the BRSO and elicits intimate confessions and funny stories from her colleagues about orchestra life. In the third season, we find out how concert programs are created, what ails musicians typically, and why there are so many jokes about violas. Sir Simon Rattle, the new chief conductor of the BRSO, answers community questions at the end of each episode.")

   podcast3 = Podcast(title="Crime Scene History – True Crime meets History",
                      id="3",
                      url="bla.com",
                      embedding_source="In Crime Scene History, Niklas Fischer and Hannes Liebrandt, two historians from Ludwig Maximilian University in Munich, leave the lecture hall and travel back to exciting crimes from the past: a mysterious water corpse in the Berlin Landwehr Canal, young Stalin as the leader of a bloody robbery, or the hunt for a war criminal halfway around the world. True crime from history discussed in an entertaining way. The focus is on the question of what this actually has to do with us today. Crime Scene History is a podcast from Bayern 2 in collaboration with the Georg von Vollmar Academy.")

   # Write to database
   database.write_rows([podcast1, podcast2, podcast3])

Since we ware using SQLAlchemy's Table classes, those tables are the exact representation of what will be stored in our
database and we will interact only through those Table classes with the content from the vector store.

Right now, we only have content in our tables and no embedding so far. The embedding is automatically computed when you
send your table to the database:


Querying the database
---------------------

Remember the following line:

.. code-block:: python

   embedding_table = database.create_abstract_embedding_table(embed_type=embedding)

Here we've specified the embedding type for the Table. The embeddings are now created from the type we've specified in
this line and sent to the vector store. Now we can query the database for content. Via the `database.session()`
attribute we may also interact with it as a normal database via SQLAlchemy.

.. code-block:: python

   with database.session() as session:
       response = session.execute(text("SELECT * from podcast;")).all()

   for row in response:
       print(row.title)

This statement now prints out all of the three podcasts in the database. Just alike you can write your custom SQL
queries to filter your results.

To select only those podcasts hosted on br24.de, you would write

.. code-block:: python

   with database.session() as session:
       response = session.execute(text("SELECT * from podcasts where url = 'br24.de';")).all()

   for row in response:
       print(row.title)

Alternatively you may use the sqlalchemy ORM syntax to query the database:

.. code-block:: python

   from sqlalchemy import select

   with database.session() as session:
       response = session.execute(select(Podcast).where(Podcast.url == 'br24.de')).all()

   for row in response:
       print(row.title)

Finding similar results
~~~~~~~~~~~~~~~~~~~~~~~

But conventional queries are not the strength of vector databases. We want to find content similar to a user query to
augment our prompts to the LLM with.

Therefore we query the database with a question, using the `retrieve_similar_content` method.
To find us some podcasts on music, we simply ask for them:

.. code-block:: python

   context = database.retrieve_similar_content("Please show me some podcasts on music.",
                                               table=Podcast,
                                               embedding_type=embedding.SENTENCE_TRANSFORMERS)


The returned context object is a list of dictionaries, with the table name as key for the context and the key
`cosine_dist`
which indicates the distance of the search term's vector and the the content's vector.

The smaller `cosine_dist` is, the more similar are query and result.

.. code-block:: python

   for row in context:
       print(row["cosine_dist"], row["Podcast"].title)

Adding context to your prompt
-----------------------------

Now we may augment our prompt to the LLM. Therefore we need to write a prompt template:

.. code-block:: python

   prompt_template = ("You are the podcast expert at Bayerischer Rundfunk. "
                      "A user asks you the following question:\n"
                      "{}\n"
                      "Here are some podcasts that you can recommend:\n"
                      "-{}\n"
                      "Limit yourself to the selection of podcasts and do not invent new ones. Recommend only one podcast from the list and briefly justify your decision. "
                      "Write in the second person, addressing the user directly.")


In the template we see two placeholders: One for the user question. If we develop an app, this would be the prompt given
to us by the user. For now, we just write it ourselves:

.. code-block:: python

   user_prompt = "Please recommend me some podcasts on music."

The second placeholder is for the context we retrieved from our database. We just need to restrucutre it as a human
readable list.

.. code-block:: python

    context = [x["Podcast"].title + ": " + x["Podcast"].embedding_source for x in context]

Then we put everything together using a python format string and send it to the LLM:

.. code-block:: python

    prompt = prompt_template.format(user_prompt, "\n-".join(context))
    response = language_model.prompt(prompt)

    print(response)

LLM usually only have a limited amount of tokens you may pass to them. If you run your RAG applicatoin on the server,
there is a little helper function to make sure you don't exceed the token limit. You pass it your template and the user
prompt as a string and the context as a list of strings.

If the context is too long, the function will pop the last elements of your context until it fits the context window.

.. code-block:: python

    context = language_model.model.fit_to_context_window(prompt_template + user_prompt, context)

Simply use this function before you pass the context to the LLM.

Registering your own Language Model
-----------------------------------

If you have your own Language Model deployed, you may want to use it with this library.

This library assumes that a language model is available through an REST API. In principle you mal also run it locally.
You may need to fill in some dummy values in the connection related fields in the upcoming example.

To register your own model you need to inherit from the Generator class in the models package.

The most simple way to do so is as follows. Your init method needs the paramters `model` and `auth_token`.
Those parameters are then passed to the init function of the `super` or parent class:

.. code-block:: python

    from brdata_rag_tools.models import Generator

    class Bison(Generator):
        def __init__(self, model, auth_token):
            super().__init__(model=model,
                             auth_token=auth_token)

Those parameters are needed internally. `model` will hold the `LLMCofig` class and `auth_token` the token to connect to
the REST API. Optional parameters you may pass are the following.

- temperature: float
- max_new_tokens: int
- top_p: float
- top_k: int
- length_penalty: float
- number_of_responses: int
- max_token_length: int

Those are only used by your service, so you don't have to stick too closely to the definitions, but for the sake of reusability
it is advised not to overload these parameters. You can always introduce your own parameters if you need to.

Each `Generator` needs a prompt method. In this method you query your service with the parameters you've specified above.

The `prompt()`-Method is usually just a wrapper around your REST API. If you choose to run the model locally you may also
query it directly from here.

The `prompt()` method should take a string as input and should return a string.

.. code-block:: python

    class Bison(models.Generator):
        def __init__(self, model, auth_token):
            super().__init__(model=model,
                             temperature=1.0,
                             max_new_tokens=256,
                             top_p=0.9,
                             length_penalty=1.0,
                             auth_token=auth_token)

        def prompt(self, prompt: str) -> str:
            import requests

            headers = {
                'accept': 'application/json',
                'Content-Type': 'application/json',
                'Authorization': f'Bearer {self.auth_token}'
            }

            json_data = {
                'id': str(time.time()),
                'prompt': prompt,
                'model_name': 'bison001',
                'max_tokens': self.max_new_tokens,
                'temperature': self.temperature,
                'top_k': self.top_k,
                'top_p': self.top_p,
            }

            response = requests.post('https://google-models-proxy.brdata-dev.de/v1/bison', headers=headers,
                                     json=json_data)

            return response.json()["response"]


If your model supports chat functionality, you can also implement a `chat()` method.

If you choose not to, the generic chat method will be used by adding the chat history to the end of your prompt.
This will not produce the best results and may also confuse the model. Use at your own risk.

Now that you've created your model, you need to register it as a last step. Use the `register` method from models.

.. code-block:: python

    from brdata_rag_tools.models import register
    register(Bison, name="bison001", max_input_tokens=8192)

You pass the method your Model class, and need to specify the name of your model and the maximum amount of tokens the
model can handle as input.

After registration you may use your model just as you would with any pre-registered model:

.. code-block:: python

    language_model = models.LLM(model=models.LLMConfig.BISON001, auth_token=os.environ.get("BISON_TOKEN"))

    response = language_model.prompt("Mighty language model, what is your name?.")
    print(response)


Registering your own Embedding Models
-------------------------------------

Registering your own embedding model follows the same principles as with language models. You create your own embedding
class by inheriting from the `Embedder` parent class.

In this example we will not query an endpoint but only return a dummy value of `[1, 2, 3]` for each row.


.. code-block:: python
    from brdata_rag_tools.embeddings import Embedder

    class Test(Embedder):
        def __init__(self):
            super().__init__(endpoint = "example.com", auth_token = None)


The parent class expects two parameters which you need to pass:

1. The `endpoint` under which the service is available. Since we don't call an external service here, we simply fill in a dummy value.
2. The `auth_token` used for authentication to your service. We leave this with `None` here as we don't call an actual endpoint.

Each embedder needs two methods:

1. `create_embedding(text)` which takes a string as input and returns the embedding as numpy array. This method is used to create the embedding for your user prompt, which is used as an input to the database.
2. `create_embedding_bulk(rows)` which takes a list of SQLAlchemy table classes as input and assigns the created embedding directly to the class's `embedding` attribute. This method is used for ingesting your data into the database. Those separate methods exist to allow you to optimize for large throughput during ingest and allows you to minimize the number of requests to your service.


.. code-block:: python

    class Test(Embedder):
        def __init__(self):
            super().__init__(endpoint = "example.com", auth_token = None)

        def create_embedding_bulk(self, rows):
        """
        Takes an  list of SQLAlchemy Table classes as input and returns them with embeddings assigned.
        """
            for row in rows:
                row.embedding = np.array([1, 2, 3])

            return rows

        def create_embedding(self, text: str) -> np.array:
            return np.array([1, 2, 3])


After you've created your class, you may register it using the `register` function from embeddings:

.. code-block:: python

    from brdata_rag_tools.embeddings import register

    embeddings.register(Test, name = "test", dimensions = 3)

You need to pass the name of your embedding model and the dimensionality to the register function.

Then you can use it just as you would with the pre-registered embedding methods. The name of your Embedder is always
stored in all-caps in the selection Enum.

.. code-block:: python

    embed_type = EmbeddingConfig.TEST

    database = databases.FAISS()
    EmbeddingTable = database.create_abstract_embedding_table(embed_type=embed_type)


Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`


Models
======

.. automodule:: models
   :members:

Databases
=========

Databases are vector databases for similarity search. Right now, only PGVector databases are supported.

.. automodule:: databases
   :members:

Embeddings
==========

.. automodule:: embeddings
   :members:
