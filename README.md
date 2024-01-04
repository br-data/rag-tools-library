# rag-tools-library
Library to support common tasks in retrieval augmented generation (RAG).

This library is in a very early stage and all the documentation is AI generated.

## Tutorial and Documentation

You find a brief tutorial and the documentation under [br-data.github.io/rag-tools-library](https://br-data.github.io/rag-tools-library/).

## Roadmap

- [ ] Add Google Bison to available LLMs
- [x] Add an offline database alternative
  - [x] FAISS and SQLite
- [x] Allow users to register their own LLMs 
- [ ] Allow users to register their own Embedding models
- [ ] Support Semantic Scholar endpoint to generate embeddings for scientific papers.
- [x] Support chat functionality; e.g. let the user give feedback on the result to the LLM.

# Deployment

Run the `build_and_deploy.sh` script in the root folder. Once prompted for the username, pass `__token__` and the pypi API 
token you've received. If you don't have an API token and feel like you should, feel free to contact the maintainers.

# Contact

Marco Lehner

[marco.lehner@br.de](mailto:marco.lehner@br.de)