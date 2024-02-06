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



