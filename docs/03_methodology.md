# Methodology — AI Career Copilot (Agentic RAG stub)

## 1. Retrieval-Augmented Generation overview

A standard RAG loop:

1. **Encode the user query** $q$ (or, here, the *employee profile* $p$ blended with the question $q$) into a vector $\mathbf{e}_q$.
2. **Retrieve** the top-$K$ documents from a corpus $\mathcal{D} = \{d_1, \dots, d_N\}$ by cosine similarity:
   $$ \text{top-K}(q) = \arg\max_{|S|=K, S \subset \mathcal{D}} \sum_{d \in S} \frac{\mathbf{e}_q^\top \mathbf{e}_d}{\|\mathbf{e}_q\| \|\mathbf{e}_d\|}. $$
3. **Generate** an answer conditioned on $(q, \text{top-K})$. The generator is normally an LLM; here it is a **template** that renders the retrieved chunks.

This project's corpus $\mathcal{D}$ = internal job postings (title + required_skills + description). The query $q$ blends the user message with the employee's profile text (current role, aspirations, skills) so retrieval is personalised.

## 2. Why TF-IDF instead of an embedding model in the demo

We use `sklearn.feature_extraction.text.TfidfVectorizer` with `(1, 2)` ngrams and `sublinear_tf=True`. Trade-offs:

| | TF-IDF | Sentence-transformers |
|---|---|---|
| Dependency | sklearn only | torch + downloaded weights (~80-400 MB) |
| Latency | < 20 ms | 100-300 ms (CPU) |
| Synonymy handling | weak (lexical only) | strong (dense semantic) |
| Reproducibility | fully deterministic | dependent on weights pinned |
| API key needed | no | no (open weights) |

For a portfolio demo that must run anywhere with no downloads and no network calls, TF-IDF is a deliberate choice. The **production** answer is a sentence-transformer (e.g. `all-MiniLM-L6-v2`) or a hosted embedding API; the retriever interface in `src/career_copilot/models.py` is designed so it slots in without a code change in `serve.py`.

## 3. The templated answer generator

Given the top-$K$ retrieved postings, we render an answer of the form:

> Based on your profile (current role: $\langle$role$\rangle$, top skills: $\langle$top-3 skills$\rangle$, aspiration: $\langle$aspiration$\rangle$), here are 3 internal opportunities that look like a fit:
>
> 1. **$\langle$title 1$\rangle$** ($\langle$dept 1$\rangle$, level $\langle$level 1$\rangle$) — overlap on $\langle$skill list 1$\rangle$. Missing: $\langle$gap list 1$\rangle$.
> 2. ...
>
> Suggested next actions:
> - Build skill X via [the H7 micro-cert recommender]
> - Talk to the hiring manager for $\langle$job_id 1$\rangle$
> - Update your aspiration tags so we can track the path

Every line is **directly grounded** in the retrieved chunk + the employee profile — there is no free generation.

## 4. The LangGraph slot (production)

In production we'd lift the templated generator into a small **LangGraph state machine**:

```
START
  -> [profile_loader]      reads emp profile from HRIS
  -> [router]              classifies question (mobility / skill-gap / promotion / relocation)
  -> [retriever]           runs the TF-IDF or embedding retriever
  -> [skill_gap_tool]      computes set difference: required_skills - emp.skills
  -> [generator]           LLM call (Claude / GPT-4o-mini) with retrieved chunks
  -> [verifier]            asserts every claim is grounded in retrieved sources
  -> END
```

The state object carries the conversation history, retrieval log, and a `sources` list that the UI renders. The verifier rejects any LLM output whose claims aren't backed by a retrieved chunk; on failure it short-circuits to the templated answer used in the demo. This keeps the demo's grounding guarantee intact even after the LLM is added.

## 5. Evaluation
| Metric | Why |
|---|---|
| Recall@5 on (employee, target_job) ground truth | Standard retrieval metric |
| Grounded-answer rate | Every answer must cite >= 1 source |
| Latency (CPU, p95) | Operational gate |
| Action diversity | At least 2 distinct suggested next actions per response |

## 6. References
- Lewis et al., *Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks*, 2020.
- Karpukhin et al., *Dense Passage Retrieval for Open-Domain Question Answering*, 2020.
- Robertson & Zaragoza, *The Probabilistic Relevance Framework: BM25 and Beyond*, 2009.
- Reimers & Gurevych, *Sentence-BERT: Sentence Embeddings using Siamese BERT-Networks*, 2019.
- LangGraph documentation, *State machines for agent workflows*.
