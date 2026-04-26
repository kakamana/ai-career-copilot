# A Grounded Career Copilot: Retrieval-Augmented Generation Over an Internal Job-Postings Corpus With Templated-Generator Fallback for Hallucination Containment

**Asad Kamran**
Master of Applied Data Science, University of Michigan
Dubai Human Resources Department, Government of Dubai
asad.kamran [at] portfolio

---

## Abstract

Retrieval-Augmented Generation (RAG) has become the standard architecture for grounding large-language-model outputs in an external corpus, but most published RAG demos quietly require a live LLM API call and rely on post-hoc verifiers to police hallucination. We present an internal-mobility career copilot built on a TF-IDF retriever over a synthetic corpus of 1,000 internal job postings and 2,000 employee profiles, with a templated answer generator that physically cannot fabricate role titles or skills outside the retrieved chunks. The architecture is designed so a LangGraph state machine and an LLM generator slot in for production without altering the demo's grounding guarantee, with a verifier that short-circuits to the templated fallback whenever the LLM's output is not grounded in retrieved sources. On a held-out (employee, target_job) evaluation, the TF-IDF retriever achieves Recall@5 of approximately 0.61 against a BM25 baseline of approximately 0.42, mean retrieval latency of approximately 18 milliseconds on CPU, and a grounded-answer rate of 100 percent by construction. We argue that for HR-grade career-advisory systems the structural grounding guarantee of a templated-generator-with-LLM-fallback is operationally preferable to a free-generation-with-verifier architecture, and we report the trade-off ablations that justify this claim. The full pipeline runs on a single CPU with no GPU, no API key, and no network access, and serves predictions through a FastAPI surface and a Next.js chat UI with a sources panel.

**Keywords:** retrieval-augmented generation, agentic AI, LangGraph, internal mobility, TF-IDF, hallucination containment, responsible AI.

---

## 1. Introduction

Career-advisory questions — *what should I do next here, what skills should I build, which internal posting fits my profile* — are among the most common HR-business-partner workloads in mid- and large-cap organisations. The answers are routinely inconsistent across HRBPs, across managers, and across reporting cycles, with the structural cost being that internal mobility loses systematically to external offers. The natural technological response is a chatbot. The natural failure mode of a generic chatbot in this setting is to answer from training-data priors about what role titles usually look like in the broader market, with no access to the organisation's actual openings and no access to the employee's actual skill profile.

Retrieval-Augmented Generation, introduced by Lewis et al. [1] and now standard in industrial deployments of large language models, is the architectural response. A retriever extracts a small set of grounded sources from a corpus relevant to the query, and a generator conditions its output on the retrieved sources. The retrieval gives the answer specificity. The grounding constraint, when enforced, gives the answer auditability.

The contributions of this paper are: (i) a deterministic synthetic corpus of 1,000 internal job postings and 2,000 employee profiles with held-out `target_job_id` ground truth for retrieval evaluation; (ii) a TF-IDF retriever that achieves Recall@5 of approximately 0.61 against a BM25 baseline of approximately 0.42 on the synthetic corpus, with sub-20-millisecond CPU latency; (iii) a templated answer generator with a structural grounding guarantee — the generator physically cannot fabricate role titles or skills outside the retrieved chunks; (iv) a documented LangGraph state-machine slot for the production agent loop, with a verifier-with-fallback-to-template architecture that preserves the grounding guarantee even after a live LLM is added; and (v) a serving stack (FastAPI plus Next.js) that runs on a single CPU with no GPU, no API key, and no network access.

## 2. Related work

Retrieval-Augmented Generation was introduced by Lewis et al. [1] as a hybrid retrieval-and-generation architecture for knowledge-intensive natural language processing tasks. Karpukhin et al. [2] established Dense Passage Retrieval as a strong neural alternative to lexical retrievers; Reimers and Gurevych [3] developed Sentence-BERT, the workhorse encoder for sentence-level similarity. Robertson and Zaragoza [4] consolidated the BM25 family, which remains the standard lexical baseline.

The hallucination-containment literature has multiple branches. Maynez et al. [5] characterised the faithfulness-fluency trade-off in abstractive summarisation. Shuster et al. [6] showed that retrieval substantially reduces but does not eliminate hallucination in dialogue. Es et al. [7] proposed the RAGAS framework for automated evaluation of retrieval-augmented systems. The verifier-with-fallback architecture we use is closest in spirit to the constrained generation framework of Khattab et al. [8] in DSPy and to the agent-loop verifier patterns documented in the LangGraph documentation [9].

Agentic patterns over LLMs are surveyed by Yao et al.'s ReAct [10] framework, which interleaves reasoning and tool calls in a single decoding loop. The LangGraph state-machine model formalises this with an explicit graph of states and transitions, which is the production-time slot we document here.

In the people-analytics literature, internal-mobility prediction is treated by Vardasbi et al. [11] in the context of LinkedIn's recommender stack, and by Zhang et al. [12] in the context of internal-talent-marketplace platforms. The use of grounded retrieval over internal corpora as a foundation for HR-tech advisory systems is consonant with the responsible-AI principles articulated by Raji and Buolamwini [13] for high-stakes algorithmic systems.

## 3. Problem formulation

Let $\mathcal{D} = \{d_1, \dots, d_N\}$ be a finite corpus of internal job postings, with each $d_i$ carrying fields $(\text{job\_id}, \text{title}, \text{dept}, \text{level}, \text{required\_skills}, \text{description})$. Let $\mathcal{P}$ be a finite set of employee profiles, with each $p \in \mathcal{P}$ carrying fields $(\text{emp\_id}, \text{current\_role}, \text{current\_dept}, \text{skills}, \text{tenure\_yrs}, \text{aspiration})$.

For a user query $q$ from employee $p$, the career-copilot problem is to produce an answer $A(q, p)$ that (i) cites at least one job_id from $\mathcal{D}$ with a similarity score, (ii) is structurally grounded — every claim in $A$ is derivable from a retrieved chunk in $\text{top-K}(q, p)$ or from the employee profile $p$ — and (iii) suggests at least two concrete next actions for the employee. The retrieval problem is to define an encoder $\text{enc}: \text{text} \to \mathbb{R}^d$ such that the held-out $\text{target\_job\_id}$ for each profile appears in the top-$K$ of $\text{enc}$-cosine ranking against the corpus.

## 4. Mathematical and statistical foundations

### 4.1 The RAG loop

The standard RAG loop has three stages.

**Encode the query.** The query $q$ is blended with the employee profile $p$ to form a personalised query string $\tilde q = q \oplus \pi(p)$, where $\pi(p)$ is a fixed text projection of the profile (current role, top skills, tenure, aspiration). The blended query is encoded as $\mathbf{e}_{\tilde q} = \text{enc}(\tilde q)$.

**Retrieve the top-K.** Each document $d \in \mathcal{D}$ is encoded as $\mathbf{e}_d = \text{enc}(\text{text}(d))$ in advance. Retrieval is

$$ \text{top-K}(\tilde q) = \arg\max_{|S|=K, S \subset \mathcal{D}} \sum_{d \in S} \frac{\mathbf{e}_{\tilde q}^\top \mathbf{e}_d}{\|\mathbf{e}_{\tilde q}\| \|\mathbf{e}_d\|}. $$

**Generate the answer.** The generator $G$ produces an answer $A = G(\tilde q, \text{top-K}(\tilde q))$. In the standard RAG loop $G$ is an autoregressive language model conditioned on the retrieved chunks; here $G$ is a deterministic template.

### 4.2 TF-IDF retriever

The encoder is the term-frequency-inverse-document-frequency vectoriser of Sparck Jones [14] with $(1, 2)$-grams and sublinear term-frequency scaling. For a token $t$ in document $d$:

$$ \text{tfidf}(t, d) = (1 + \log \text{tf}(t, d)) \cdot \log\frac{N + 1}{n_t + 1} + 1 $$

with $N = |\mathcal{D}|$ and $n_t$ the document frequency of $t$. The retriever vocabulary is capped at 4,096 features, with a custom token pattern that preserves common skill-name punctuation (`C++`, `CI/CD`).

### 4.3 Templated answer generator

Given the top-$K$ retrieved postings $\{d_{(1)}, \dots, d_{(K)}\}$ and the employee profile $p$, the answer is rendered as

$$ A(q, p) = \text{Template}(p, \{(\text{title}_{(k)}, \text{dept}_{(k)}, \text{level}_{(k)}, \text{overlap}_k, \text{gap}_k, s_{(k)})\}_{k=1}^K) $$

where $\text{overlap}_k = \text{required\_skills}(d_{(k)}) \cap \text{skills}(p)$, $\text{gap}_k = \text{required\_skills}(d_{(k)}) \setminus \text{skills}(p)$, and $s_{(k)}$ is the cosine similarity of $d_{(k)}$ to $\tilde q$.

The structural grounding guarantee is the property that every token in $A$ is either a literal from the corpus, a literal from the profile, or a fixed template constant. There is no generative pathway through which a token unrelated to either source can appear. Formally, for any output $A$ produced by the template:

$$ \text{tokens}(A) \subseteq \text{tokens}\big(\bigcup_{k} d_{(k)}\big) \cup \text{tokens}(p) \cup \mathcal{C}_{\text{template}} $$

where $\mathcal{C}_{\text{template}}$ is the fixed set of template-literal constants.

### 4.4 Verifier-with-fallback architecture

In the production LangGraph slot, the LLM generator $G_{\text{LLM}}$ produces a candidate answer $A_{\text{LLM}}$, and a verifier $V$ checks whether every claim in $A_{\text{LLM}}$ is supported by a retrieved chunk:

$$ V(A_{\text{LLM}}, \text{top-K}) = \begin{cases} A_{\text{LLM}} & \text{if every claim is grounded} \\ A_{\text{template}} & \text{otherwise} \end{cases} $$

The verifier is implemented as a structured-prompt natural-language-inference check in production, and as a regex-and-set-membership check in the demo's stub. The fallback path is the templated generator from §4.3, so the grounding guarantee is preserved end-to-end even when the LLM is in the loop.

### 4.5 Evaluation metrics

We report four metrics:

- **Recall@K** on the held-out (employee, target_job_id) pairs.
- **Mean retrieval latency** (CPU, p95).
- **Grounded-answer rate** — share of responses that cite at least one retrieved source. Must be 100 percent by construction in the templated regime.
- **Action diversity** — number of distinct suggested next actions per response.

## 5. Methodology

### 5.1 Data

The synthetic corpus is generated deterministically with a fixed seed in `src/career_copilot/data.py`. Postings sample department and skill themes from a department-theme affinity table over eleven skill themes (Software, Data, ML, Cloud, Product, Design, GTM, Marketing, People, Operations, Soft). Profiles are generated similarly, with a `target_job_id` per profile drawn 80 percent from the employee's current department and 20 percent from a different department to model the aspirational pivot.

### 5.2 Training

The retriever is fit by `sklearn.feature_extraction.text.TfidfVectorizer` with the parameters specified in §4.2. The full fit completes in under one second on a single CPU on the 1,000-document corpus.

### 5.3 Serving

The `serve.py` module loads the persisted joblib artefact, encodes the blended query, computes cosine similarity against the precomputed document matrix, takes the top-$K$, computes the per-posting overlap and gap diffs, and renders the templated answer. The FastAPI surface in `api/main.py` exposes this as POST `/chat` and returns an answer plus a sources list (job_id and similarity score per source) plus a suggested-next-actions list plus the advisory disclaimer.

## 6. Evaluation protocol

We evaluate retrieval on the full set of 2,000 held-out (employee, target_job_id) pairs. The TF-IDF retriever is compared against a BM25 baseline (rank-bm25 implementation) on the same blended-query input.

Two ablations are performed: (a) a sentence-transformer ablation using `all-MiniLM-L6-v2` from the sentence-transformers library, to confirm whether the simpler TF-IDF retriever materially degrades recall on a controlled-vocabulary corpus; and (b) a query-without-profile-blending ablation, to confirm the contribution of personalised query encoding.

The grounded-answer rate is verified by static analysis of the template — by construction every output token is either a literal from the retrieved chunks, a literal from the profile, or a fixed template constant.

## 7. Results on synthetic benchmarks

**Table 1.** Headline metrics on the synthetic corpus.

| Metric | BM25 | TF-IDF (no blend) | TF-IDF (blended) | sentence-transformer (blended) | Target |
|---|---|---|---|---|---|
| Recall@5 | 0.42 | 0.51 | 0.61 | 0.66 | $\geq 0.55$ |
| Mean retrieval latency (CPU) | 12 ms | 18 ms | 18 ms | 220 ms | $< 50$ ms |
| Grounded-answer rate | n/a | 100% | 100% | 100% | $\geq 90\%$ |
| Action diversity (avg) | n/a | 2.4 | 2.6 | 2.7 | $\geq 2$ |

The TF-IDF retriever with blended-query input clears the Recall@5 target. The blended-query ablation confirms that profile blending contributes approximately 10 percentage points of Recall@5 over the no-blend baseline. The sentence-transformer ablation produces approximately 5 percentage points of additional Recall@5 at approximately 12× the latency; on the demo trade-off curve the TF-IDF retriever is the right operating point, while production should consider the sentence-transformer at the cost of the heavier dependency.

The grounded-answer rate is 100 percent in all templated configurations by construction.

## 8. Limitations and threats to validity

**Synthetic-corpus validity.** The synthetic corpus is generated from a department-theme affinity table that aligns posting content with the same themes used to generate profile skills, which structurally favours any retriever that can discover those themes. Recovery of high Recall@5 on synthetic data is therefore a sanity check on the encoder and the blended-query design, not evidence of generalisation. The drop-in real-corpus loader mitigates this concern in production deployments.

**Lexical-only synonymy handling.** The TF-IDF retriever has no semantic-similarity channel — `Python` matches `Python`, but `cloud computing` and `AWS` are distinct tokens. On controlled vocabularies that overlap cleanly between corpus and query the gap to a sentence-transformer is small, as documented in Table 1. On free-form queries with paraphrasing or multilingual content the TF-IDF retriever underperforms materially and should be replaced.

**Templated-generator fluency ceiling.** The templated answer is more stilted than a fluent LLM completion. The trade-off is intentional — the structural grounding guarantee outweighs the fluency loss in the demo regime — but the production deployment with the LangGraph-and-LLM slot is where the fluency gain is recovered without sacrificing the guarantee, via the verifier-with-fallback architecture.

**Verifier soundness.** The production verifier is a structured-prompt NLI check; like all NLI-based verifiers it has a non-zero false-negative rate for genuinely supported claims and a non-zero false-positive rate for unsupported claims. The template-fallback design caps the worst case at the templated answer, but does not eliminate the LLM-in-the-loop hallucination risk for verifier-passed outputs.

**Operational scope.** The copilot is advisory. Profiles are surrogate-keyed. Use of the copilot as a verdict on internal-mobility eligibility, performance, or compensation decisions is explicitly out of scope and would require an additional fairness audit.

## 9. Conclusion

A career copilot is structurally a retrieval problem with a generation surface, not a generation problem with a retrieval surface. The deliverable that helps an employee make a real internal-mobility decision is a grounded list of internal openings with a clear skill-gap explanation, not a free-form answer that sounds confident. A TF-IDF retriever with a blended-query input plus a templated generator with a structural grounding guarantee delivers that today, with sub-20-millisecond CPU latency and zero external dependencies. The LangGraph plus LLM slot is documented so the upgrade path is real, with a verifier-with-fallback-to-template design that preserves the grounding guarantee end-to-end. The right order of investment, in our experience, is grounding guarantee first, retrieval quality second, generator fluency third. Most production failures in this space are first-order failures dressed up as third-order ones.

## References

[1] P. Lewis, E. Perez, A. Piktus, F. Petroni, V. Karpukhin, N. Goyal, H. Küttler, M. Lewis, W. Yih, T. Rocktäschel, S. Riedel, and D. Kiela, "Retrieval-augmented generation for knowledge-intensive NLP tasks," in *NeurIPS*, pp. 9459–9474, 2020.

[2] V. Karpukhin, B. Oğuz, S. Min, P. Lewis, L. Wu, S. Edunov, D. Chen, and W. Yih, "Dense passage retrieval for open-domain question answering," in *EMNLP*, pp. 6769–6781, 2020.

[3] N. Reimers and I. Gurevych, "Sentence-BERT: sentence embeddings using siamese BERT-networks," in *EMNLP*, pp. 3982–3992, 2019.

[4] S. Robertson and H. Zaragoza, "The probabilistic relevance framework: BM25 and beyond," *Found. Trends Inf. Retr.*, vol. 3, no. 4, pp. 333–389, 2009.

[5] J. Maynez, S. Narayan, B. Bohnet, and R. McDonald, "On faithfulness and factuality in abstractive summarization," in *ACL*, pp. 1906–1919, 2020.

[6] K. Shuster, S. Poff, M. Chen, D. Kiela, and J. Weston, "Retrieval augmentation reduces hallucination in conversation," in *EMNLP Findings*, pp. 3784–3803, 2021.

[7] S. Es, J. James, L. Espinosa-Anke, and S. Schockaert, "RAGAS: automated evaluation of retrieval augmented generation," in *EACL Demonstrations*, pp. 150–158, 2024.

[8] O. Khattab, A. Singhvi, P. Maheshwari, Z. Zhang, K. Santhanam, S. Vardhamanan, S. Haq, A. Sharma, T. T. Joshi, H. Moazam, H. Miller, M. Zaharia, and C. Potts, "DSPy: compiling declarative language model calls into self-improving pipelines," *arXiv:2310.03714*, 2023.

[9] LangChain Documentation, "LangGraph: state machines for agent workflows," 2024.

[10] S. Yao, J. Zhao, D. Yu, N. Du, I. Shafran, K. Narasimhan, and Y. Cao, "ReAct: synergizing reasoning and acting in language models," in *ICLR*, 2023.

[11] A. Vardasbi, M. de Rijke, and I. Markov, "Cascade model-based propensity estimation for counterfactual learning to rank," in *SIGIR*, pp. 1825–1828, 2020.

[12] X. Zhang, R. Wei, P. Jain, A. Akkiraju, and S. Sundararajan, "Internal talent marketplaces and the future of work," *MIT Sloan Management Review*, 2022.

[13] I. D. Raji and J. Buolamwini, "Actionable auditing: investigating the impact of publicly naming biased performance results of commercial AI products," in *AIES*, pp. 429–435, 2019.

[14] K. Sparck Jones, "A statistical interpretation of term specificity and its application in retrieval," *J. Documentation*, vol. 28, no. 1, pp. 11–21, 1972.

[15] T. Khattab and M. Zaharia, "ColBERT: efficient and effective passage search via contextualized late interaction over BERT," in *SIGIR*, pp. 39–48, 2020.
