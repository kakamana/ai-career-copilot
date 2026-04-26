# Notebook 03 — Retriever + templated generator

## 1. Build index
>>> `vec, X = fit_retriever(postings)` — TF-IDF artefacts + dense matrix for cosine.

## 2. Profile blend
>>> `q = blend(query, profile)` — concatenate profile text + question.

## 3. Top-K
>>> `top = retrieve(q, k=5)` — cosine over X.

## 4. Templated answer
>>> `answer = render_answer(profile, top)` — uses retrieved chunks only.

## 5. Persist
>>> `models.save({"vec": vec, "X": X, "postings_idx": ids}, "rag_index.joblib")`
