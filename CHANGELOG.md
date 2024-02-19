# Changelog

## 0.1.0 -> 0.1.1
- Allow users to write tables without embedding col with the Database.write_rows() interface.

## 0.1.1 -> 0.1.2
- Use env var to set sentence embedding endpoint

## 0.1.2 -> 0.1.3
- Add expunge parameter to Database.write_rows() method to allow users to re-use their classes.
  (#4)[https://github.com/br-data/rag-tools-library/issues/4]
- Users may now inherit from PGVector and FAISS classes 
  (#3)[https://github.com/br-data/rag-tools-library/issues/3]

## 0.1.3 -> 0.1.4
- Allow users to pass a maximum cosine distance to the database.retrieve_similar_content() method.

## 0.1.4 -> 0.1.4.1
- Added parameters to handle the lifecycle of DB objects. Since SQLAlchemy expires objects after commiting they can't 
be accessed anymore. The `expire_on_commit=False` parameter disables this behaviour.

## 0.1.4.1 -> 0.1.4.2
- Fixed a bug with which it was not possible to init the model without passing a auth token
- IGEL now also accepts "IGEL_URL" to query the llm endpoint.

## 0.1.4.2 -> 0.1.5
- Closed issues [2](https://github.com/br-data/rag-tools-library/issues/2) and [10](https://github.com/br-data/rag-tools-library/issues/10)
by considering the max_new_tokens param in the `fit_to_context_window()` method. 