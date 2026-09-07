# %% [markdown]
# # Module 10 — Text, Features and Embeddings
#
# scikit-learn's text tooling is thirty-year-old technology that refuses to
# die, because on a large class of real problems it is still the right answer:
# it trains in seconds, runs in microseconds, needs no GPU, and every prediction
# can be traced to specific words.
#
# This module covers that tooling properly, and then draws the line: what
# TF-IDF structurally *cannot* do, where embeddings take over, and how
# scikit-learn remains the harness on the other side of that line.
#
# ### Learning objectives
#
# 1. Use `CountVectorizer` / `TfidfVectorizer` deliberately — every parameter
#    that matters, and why.
# 2. Combine text and tabular features in one pipeline.
# 3. Interpret a linear text model down to individual tokens.
# 4. Name the three things bag-of-words cannot represent.
# 5. Wrap an embedding model as a scikit-learn transformer, and reason about the
#    cost/latency/accuracy trade honestly.
# 6. Know where sklearn sits in a modern LLM-adjacent stack.

# %%
import sys
from pathlib import Path

ROOT = Path.cwd().parent if Path.cwd().name in {"notebooks", "solutions"} else Path.cwd()
sys.path.insert(0, str(ROOT / "src"))

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from skmastery import load_support_tickets, set_plot_style

set_plot_style()
pd.set_option("display.width", 130)
pd.set_option("display.max_columns", 40)
rng = np.random.default_rng(0)

tickets = load_support_tickets()
print(tickets["category"].value_counts().to_string())
tickets.head(4)

# %% [markdown]
# ## 1. From text to a matrix
#
# `CountVectorizer` does four things, in order, and each is a parameter you can
# control:
#
# 1. **Preprocess** — lowercase, strip accents (`lowercase`, `strip_accents`).
# 2. **Tokenise** — split into terms (`token_pattern`, or your own `tokenizer`).
# 3. **Build the vocabulary** — with cut-offs (`min_df`, `max_df`, `max_features`,
#    `stop_words`, `ngram_range`).
# 4. **Count** — producing a sparse document-term matrix.

# %%
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer

toy = [
    "unauthorised payment of £250 on my card",
    "I did not authorise this payment",
    "please reset my password, the app will not let me log in",
    "payment failed and the app crashed",
]

cv = CountVectorizer()
M = cv.fit_transform(toy)
pd.DataFrame(M.toarray(), columns=cv.get_feature_names_out(), index=[f"doc{i}" for i in range(4)])

# %% [markdown]
# Note what the default `token_pattern` (`(?u)\b\w\w+\b`) already threw away:
# the `£250`, the single-letter `I`, and all punctuation. Those choices are
# defaults, not laws — and `£250` might be exactly the token you need.

# %%
print("default pattern drops:", set("unauthorised payment of £250 on my card I".lower().split()) - set(cv.get_feature_names_out()))

# A pattern that keeps currency amounts and single characters.
cv2 = CountVectorizer(token_pattern=r"(?u)\b\w+\b|£\d+(?:\.\d+)?")
print("\nwith a custom pattern:", [t for t in cv2.fit(toy).get_feature_names_out() if "£" in t or len(t) == 1])

# %% [markdown]
# ### TF-IDF: down-weighting words that are everywhere
#
# Term frequency times inverse document frequency. A word appearing in every
# document ("payment", in a payments company's ticket queue) carries no
# discriminating information, so IDF shrinks it.
#
# $$ \text{idf}(t) = \log\frac{1 + n}{1 + \text{df}(t)} + 1 $$
#
# (sklearn's smoothed default. The `+1`s prevent division by zero; the trailing
# `+1` stops terms that appear everywhere from being zeroed entirely.)

# %%
tfidf = TfidfVectorizer()
tfidf.fit(toy)
idf = pd.Series(tfidf.idf_, index=tfidf.get_feature_names_out()).sort_values()
print("lowest IDF (common, uninformative):")
print(idf.head(4).round(3).to_string())
print("\nhighest IDF (rare, discriminating):")
print(idf.tail(4).round(3).to_string())

# %%
# Verify the formula by hand.
n_docs = len(toy)
df_payment = sum("payment" in d.lower() for d in toy)
print(f"'payment' appears in {df_payment}/{n_docs} docs")
print(f"manual idf : {np.log((1 + n_docs) / (1 + df_payment)) + 1:.6f}")
print(f"sklearn idf: {tfidf.idf_[list(tfidf.get_feature_names_out()).index('payment')]:.6f}")

# %% [markdown]
# ### The parameters that actually change your results
#
# | Parameter | What it does | Practical guidance |
# |---|---|---|
# | `ngram_range` | include multi-word terms | `(1,2)` is the standard upgrade — captures "not authorised" |
# | `min_df` | drop terms in fewer than *k* docs | **the most valuable knob**; `min_df=2` removes typos and hapaxes |
# | `max_df` | drop terms in more than *x* of docs | a data-driven stop-word list, better than a fixed one |
# | `sublinear_tf` | use `1+log(tf)` instead of raw count | on by default in most competition code; helps on longer docs |
# | `stop_words` | remove a fixed list | **be careful** — the English list removes "not" and "no" |
# | `analyzer="char_wb"` | character n-grams within word boundaries | robust to typos and misspellings; excellent on noisy input |
# | `norm` | L2 (default) / L1 / None | L2 makes documents comparable regardless of length |
#
# The `stop_words="english"` warning is not pedantic. On a sentiment or
# dispute-classification task, "not" is one of the most informative tokens in the
# language, and the built-in list deletes it.

# %%
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.pipeline import Pipeline

X_text, y_text = tickets["text"], tickets["category"]

configs = {
    "unigrams, defaults": TfidfVectorizer(),
    "+ bigrams": TfidfVectorizer(ngram_range=(1, 2)),
    "+ bigrams, min_df=2": TfidfVectorizer(ngram_range=(1, 2), min_df=2),
    "+ sublinear_tf": TfidfVectorizer(ngram_range=(1, 2), min_df=2, sublinear_tf=True),
    "stop_words='english'": TfidfVectorizer(ngram_range=(1, 2), min_df=2, stop_words="english"),
    "char_wb 3-5 grams": TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5), min_df=2),
    "CountVectorizer (no idf)": CountVectorizer(ngram_range=(1, 2), min_df=2),
}

rows = []
for name, vec in configs.items():
    pipe = Pipeline([("vec", vec), ("clf", LogisticRegression(max_iter=2000, C=5))])
    sc = cross_val_score(pipe, X_text, y_text, cv=5, scoring="f1_macro", n_jobs=-1)
    n_feat = vec.fit(X_text).transform(X_text[:5]).shape[1]
    rows.append({"config": name, "f1_macro": round(sc.mean(), 4), "±": round(sc.std(), 4), "n_features": n_feat})
pd.DataFrame(rows).set_index("config")

# %% [markdown]
# **Read the `±` column before the `f1_macro` column.** Every configuration lands
# between 0.836 and 0.842, with a fold-to-fold standard deviation of ~0.009. The
# spread across seven quite different feature representations is *smaller than
# the noise*. Plain unigrams with stock defaults are as good as anything.
#
# That is a real and common result, and it has a cause: this dataset has a hard
# ceiling (5% label noise plus 16% genuinely ambiguous messages), and every
# configuration is already at it. **When a whole family of representations ties,
# stop tuning the representation and go measure the ceiling** — which is
# Exercise 10.2, and the most useful thing in this module.
#
# Two observations that survive the tie:
#
# **Character n-grams reach the same ceiling** with 3.5× the features and no
# vocabulary assumptions. They are doing this despite the deliberate typos
# (`teh`, `acount`) and shouting-case that break word tokens — which is why
# `char_wb` is the right first move on genuinely messy customer text, even
# though it shows no advantage here.
#
# **Stop-word removal did not hurt on this data — but do not generalise that.**
# The English list deletes "not", "no", "did" and "will". Here the category
# signal lives in nouns ("password", "statement", "instalment"), so losing the
# function words costs nothing. On a sentiment, dispute-outcome or
# complaint-severity task the same list would delete the distinction between "I
# did not authorise this" and "I authorised this". Check, do not assume.

# %% [markdown]
# ## 2. Interpreting a text model
#
# A linear model over TF-IDF features gives you the most directly auditable
# explanation available in machine learning: a signed weight per word.

# %%
Xt_tr, Xt_te, yt_tr, yt_te = train_test_split(X_text, y_text, test_size=0.3, stratify=y_text, random_state=0)
text_model = Pipeline([
    ("vec", TfidfVectorizer(ngram_range=(1, 2), min_df=2, sublinear_tf=True)),
    ("clf", LogisticRegression(max_iter=3000, C=5)),
]).fit(Xt_tr, yt_tr)

vocab = text_model.named_steps["vec"].get_feature_names_out()
coefs = text_model.named_steps["clf"].coef_
classes = text_model.named_steps["clf"].classes_

top = {}
for i, cls in enumerate(classes):
    order = np.argsort(coefs[i])[::-1][:8]
    top[cls] = [f"{vocab[j]} ({coefs[i][j]:+.2f})" for j in order]
pd.DataFrame(top).T.rename(columns=lambda c: f"#{c + 1}")

# %%
# Per-prediction explanation: which tokens in THIS document drove THIS decision?
def explain(model, doc, top_k=6):
    vec, clf = model.named_steps["vec"], model.named_steps["clf"]
    x = vec.transform([doc])
    pred = clf.predict(x)[0]
    idx = list(clf.classes_).index(pred)
    contrib = x.multiply(clf.coef_[idx]).toarray().ravel()
    names = vec.get_feature_names_out()
    order = np.argsort(np.abs(contrib))[::-1][:top_k]
    return pred, pd.Series(contrib[order], index=names[order]).round(4)


doc = "I did not authorise the £320.15 charge from AMZN MKTPLACE on my card"
pred, contrib = explain(text_model, doc)
print(f"document : {doc}\npredicted: {pred}\n")
contrib.to_frame("contribution")

# %% [markdown]
# That is a complete, exact, per-token explanation computed in microseconds. No
# sampling, no approximation, no SHAP background dataset. For a regulated
# complaint-classification or dispute-routing system, this property is worth a
# great deal — and it is the first thing you lose when you move to embeddings.

# %% [markdown]
# ## 3. Text plus tabular in one pipeline
#
# `ColumnTransformer` takes a *string* column name (not a list) for a text
# vectoriser, because vectorisers expect a 1-D sequence of documents.

# %%
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder

mixed = ColumnTransformer([
    ("text", TfidfVectorizer(ngram_range=(1, 2), min_df=2, sublinear_tf=True), "text"),   # a string, not ["text"]
    ("chan", OneHotEncoder(handle_unknown="ignore"), ["channel"]),
])

mixed_model = Pipeline([("prep", mixed), ("clf", LogisticRegression(max_iter=3000, C=5))])
sc_mixed = cross_val_score(mixed_model, tickets[["text", "channel"]], y_text, cv=5, scoring="f1_macro", n_jobs=-1)
sc_text = cross_val_score(text_model, X_text, y_text, cv=5, scoring="f1_macro", n_jobs=-1)
print(f"text only        : {sc_text.mean():.4f}")
print(f"text + channel   : {sc_mixed.mean():.4f}")

# %%
# The common error, and its error message.
try:
    ColumnTransformer([("text", TfidfVectorizer(), ["text"])]).fit_transform(tickets[["text"]])
except Exception as e:
    print(f"{type(e).__name__}: {str(e).splitlines()[0][:120]}")
    print("\n-> a list gives the vectoriser a DataFrame; it needs a Series.")

# %% [markdown]
# ## 4. Which classifier for text
#
# Text is high-dimensional and sparse, which favours linear models — the data is
# almost always linearly separable in a 50,000-dimensional space.

# %%
from sklearn.linear_model import SGDClassifier
from sklearn.naive_bayes import ComplementNB, MultinomialNB
from sklearn.svm import LinearSVC
from time import perf_counter

vec_common = TfidfVectorizer(ngram_range=(1, 2), min_df=2, sublinear_tf=True)
clfs = {
    "MultinomialNB": MultinomialNB(),
    "ComplementNB": ComplementNB(),
    "LogisticRegression": LogisticRegression(max_iter=3000, C=5),
    "LinearSVC": LinearSVC(C=1.0),
    "SGDClassifier(log_loss)": SGDClassifier(loss="log_loss", alpha=1e-5, max_iter=2000, random_state=0),
}
rows = []
for name, clf in clfs.items():
    t0 = perf_counter()
    sc = cross_val_score(Pipeline([("vec", vec_common), ("clf", clf)]), X_text, y_text, cv=5, scoring="f1_macro", n_jobs=-1)
    rows.append({"classifier": name, "f1_macro": round(sc.mean(), 4), "±": round(sc.std(), 4),
                 "seconds": round(perf_counter() - t0, 2),
                 "gives probabilities": hasattr(clf, "predict_proba")})
pd.DataFrame(rows).set_index("classifier")

# %% [markdown]
# `ComplementNB` is the one people miss: it is a variant of multinomial naive
# Bayes designed for **imbalanced** text, and it usually beats `MultinomialNB` on
# skewed class distributions while costing the same (essentially nothing).
#
# `LinearSVC` is frequently the strongest and has no `predict_proba` — wrap it in
# `CalibratedClassifierCV` if you need probabilities (Module 08).

# %% [markdown]
# ## 5. `HashingVectorizer`: text at scale
#
# `TfidfVectorizer` must hold the whole vocabulary in memory and needs two passes
# (one to build the vocabulary, one to transform). `HashingVectorizer` hashes
# tokens into a fixed number of buckets: **stateless, single-pass, constant
# memory, and streamable**.

# %%
from sklearn.feature_extraction.text import HashingVectorizer

hv = HashingVectorizer(n_features=2**16, alternate_sign=False, ngram_range=(1, 2))
Hm = hv.transform(X_text)
print(f"HashingVectorizer : {Hm.shape[1]:,} fixed buckets, no vocabulary stored, no fit needed")
print(f"TfidfVectorizer   : {len(vec_common.fit(X_text).vocabulary_):,} learned terms, dict held in memory")

# Out-of-core learning: stream batches through partial_fit. This trains on data
# that never fits in RAM.
sgd = SGDClassifier(loss="log_loss", alpha=1e-5, random_state=0)
classes = np.unique(y_text)
for start in range(0, len(X_text), 500):
    batch_X = hv.transform(X_text.iloc[start:start + 500])
    sgd.partial_fit(batch_X, y_text.iloc[start:start + 500], classes=classes)
print(f"\nout-of-core SGD training accuracy: {sgd.score(hv.transform(Xt_te), yt_te):.4f}")

# %% [markdown]
# The trade: hashing is one-way, so `get_feature_names_out()` does not exist and
# you lose the token-level interpretability of section 2. Collisions also merge
# unrelated terms — with 2¹⁶ buckets and a 20,000-term vocabulary, collisions are
# rare enough not to matter; with 2¹² they would.

# %% [markdown]
# ## 6. What bag-of-words cannot do
#
# Three structural limitations. None is fixable by tuning.

# %%
pairs = [
    ("my card was stolen", "someone took my credit card"),                    # synonymy
    ("the payment was not authorised", "the payment was authorised"),         # negation
    ("dog bites man", "man bites dog"),                                       # word order
]
v = TfidfVectorizer().fit([s for p in pairs for s in p])
from sklearn.metrics.pairwise import cosine_similarity

for a, b in pairs:
    sim = cosine_similarity(v.transform([a]), v.transform([b]))[0, 0]
    print(f"{sim:.3f}   {a!r}  vs  {b!r}")

# %% [markdown]
# 1. **Synonymy.** Two sentences meaning the same thing share few words, so
#    similarity is near zero. TF-IDF has no notion that "stolen" and "took" are
#    related.
# 2. **Negation.** Two sentences meaning *opposite* things share almost every
#    word, so similarity is near one. (Bigrams help — "not authorised" becomes a
#    token — but only for negations adjacent to the verb.)
# 3. **Word order.** "Dog bites man" and "man bites dog" are *identical* under
#    unigrams: similarity exactly 1.0.
#
# These are the three problems embeddings solve, and they are the honest reason
# to reach past scikit-learn's vectorisers.

# %% [markdown]
# ## 7. Dense representations
#
# ### The classical version: LSA
#
# Truncated SVD on a TF-IDF matrix gives dense vectors where related terms move
# together, because it factors the term–document co-occurrence structure. This is
# latent semantic analysis, it predates neural embeddings by thirty years, and it
# runs in a second.

# %%
from sklearn.decomposition import TruncatedSVD
from sklearn.preprocessing import Normalizer

lsa = Pipeline([
    ("tfidf", TfidfVectorizer(ngram_range=(1, 2), min_df=2, sublinear_tf=True)),
    ("svd", TruncatedSVD(n_components=150, random_state=0)),
    ("norm", Normalizer()),                 # SVD output is not normalised; cosine needs it
])
Z_lsa = lsa.fit_transform(X_text)
print(f"{Z_lsa.shape[1]}-dimensional dense vectors, "
      f"{lsa.named_steps['svd'].explained_variance_ratio_.sum():.1%} of variance retained")

sc_lsa = cross_val_score(Pipeline([("lsa", lsa), ("clf", LogisticRegression(max_iter=3000))]),
                         X_text, y_text, cv=5, scoring="f1_macro", n_jobs=-1)
print(f"\nLSA + LogisticRegression f1_macro : {sc_lsa.mean():.4f}")
print(f"raw TF-IDF + LogisticRegression   : {sc_text.mean():.4f}")

# %% [markdown]
# LSA lands in the same place as everything else — inside the noise band again,
# with 150 dense dimensions instead of ~2,000 sparse ones. Do not read that as
# "LSA is better": on a task with real headroom, compressing to 150 components
# usually costs a little accuracy, because the SVD optimises *reconstruction*,
# not class separation, and it will happily discard a low-variance direction that
# happens to be the one that separates your classes.
#
# What LSA reliably buys you is a **compact dense representation**: cheap
# nearest-neighbour retrieval, a feature block small enough to concatenate with
# tabular features without swamping them, and vectors you can cluster. Those are
# its uses — not squeezing out classifier accuracy.

# %% [markdown]
# ### The modern version: sentence embeddings
#
# A transformer model maps a sentence to a dense vector in which *semantic*
# similarity is geometric. Synonymy and word order — two of the three failures
# above — go away.
#
# Wrapping one as a scikit-learn transformer takes about fifteen lines, and once
# wrapped it composes with everything you have learned: pipelines, grid search,
# cross-validation, calibration.
#
# ```python
# from sentence_transformers import SentenceTransformer
# from sklearn.base import BaseEstimator, TransformerMixin
#
# class SentenceEmbedder(BaseEstimator, TransformerMixin):
#     """Any sentence-transformers model, as a scikit-learn transformer."""
#
#     def __init__(self, model_name="sentence-transformers/all-MiniLM-L6-v2", batch_size=64):
#         self.model_name = model_name          # hyperparameters: stored unmodified
#         self.batch_size = batch_size
#
#     def fit(self, X, y=None):
#         self.model_ = SentenceTransformer(self.model_name)   # learned state: trailing underscore
#         self.n_features_out_ = self.model_.get_sentence_embedding_dimension()
#         return self
#
#     def transform(self, X):
#         from sklearn.utils.validation import check_is_fitted
#         check_is_fitted(self)
#         return self.model_.encode(list(X), batch_size=self.batch_size,
#                                   show_progress_bar=False, normalize_embeddings=True)
#
#     def get_feature_names_out(self, input_features=None):
#         return np.array([f"emb{i}" for i in range(self.n_features_out_)])
#
#
# pipe = Pipeline([("embed", SentenceEmbedder()), ("clf", LogisticRegression(max_iter=2000))])
# ```
#
# That is the whole integration. Note that it follows the Module 00 contract
# exactly: hyperparameters stored unmodified in `__init__`, learned state with a
# trailing underscore in `fit`. Because of that, `GridSearchCV` can search over
# `embed__model_name` and compare three different embedding models the same way
# it compares three values of `C`.
#
# It is left as code rather than executed here because `sentence-transformers`
# pulls in PyTorch (~1 GB) and downloads model weights — worth doing on your own
# machine, not worth making a prerequisite of this notebook. Exercise 10.4 walks
# through it.

# %% [markdown]
# ### The honest comparison
#
# | | TF-IDF + linear | Sentence embeddings + linear | Fine-tuned transformer | LLM API |
# |---|---|---|---|---|
# | Training time | seconds | minutes (encoding) | hours, GPU | none |
# | Inference latency | ~10 µs | ~5–50 ms (GPU/CPU) | ~10–100 ms | 200 ms–2 s |
# | Cost per million docs | pennies | ~£1–10 | GPU hours | £100s–1000s |
# | Handles synonymy | ✗ | ✓ | ✓ | ✓ |
# | Handles negation | partly (bigrams) | mostly | ✓ | ✓ |
# | Handles word order | ✗ | ✓ | ✓ | ✓ |
# | Needs labelled data | yes | yes (but fewer) | yes (more) | **no** |
# | Per-token explanation | **exact** | none | attribution methods | none |
# | Runs air-gapped | ✓ | ✓ | ✓ | ✗ |
# | Deterministic | ✓ | ✓ | ✓ | not reliably |
#
# **When TF-IDF is still the right answer:** short, formulaic, domain-specific
# text; very high volume with tight latency budgets; a hard explainability
# requirement; no GPU; or a strong baseline you need in an hour.
#
# **When to move to embeddings:** paraphrase and synonymy matter; you have few
# labels and want to exploit pre-training; you need semantic search or
# clustering; the text is long or linguistically varied.
#
# > 💼 **Consulting lens.** The failure mode I see most often is skipping the
# > baseline. A team spends six weeks on an embedding pipeline, ships F1 = 0.87,
# > and nobody ever ran `TfidfVectorizer` + `LogisticRegression`, which would
# > have taken twenty minutes and scored 0.84. The right sequence is: build the
# > cheap baseline first, *then* decide whether the gap justifies the cost — in
# > latency, in GPU spend, in explainability, and in the ongoing maintenance of
# > a model you did not train.

# %% [markdown]
# ## 8. Where sklearn sits in an LLM-era stack
#
# It does not compete with the language model. It surrounds it.
#
# 1. **A classifier head on frozen embeddings.** Embed once, then train a
#    logistic regression on the vectors. Cheap, fast, calibratable, and often
#    within a point or two of a fine-tuned model.
# 2. **Retrieval.** `NearestNeighbors` over embeddings is a perfectly good vector
#    index up to ~1M documents, and you already know how to cross-validate it.
# 3. **Routing and triage.** A cheap sklearn classifier decides which requests
#    need the expensive model at all — often the single largest cost saving in an
#    LLM system.
# 4. **Evaluating LLM outputs.** Classifying, clustering and scoring generated
#    text is a tabular ML problem.
# 5. **Guardrails.** A fast classifier on inputs and outputs, with a threshold
#    tuned to an explicit cost matrix (Module 08).
# 6. **The whole evaluation harness.** Cross-validation, metrics, calibration,
#    significance — none of that changed because the features got denser.

# %%
# (1) and (2), demonstrated with LSA vectors standing in for embeddings.
from sklearn.neighbors import NearestNeighbors

index = NearestNeighbors(n_neighbors=4, metric="cosine").fit(Z_lsa)


def similar(query, k=4):
    q = lsa.transform([query])
    dist, idx = index.kneighbors(q, n_neighbors=k)
    return pd.DataFrame({
        "similarity": (1 - dist[0]).round(3),
        "category": tickets["category"].iloc[idx[0]].to_numpy(),
        "text": tickets["text"].iloc[idx[0]].str.slice(0, 62).to_numpy(),
    })


similar("someone used my card without permission")

# %%
# (3) Routing: send only the uncertain cases to the expensive model.
proba = text_model.predict_proba(Xt_te)
confidence = proba.max(axis=1)

rows = []
for thr in [0.5, 0.7, 0.8, 0.9, 0.95]:
    auto = confidence >= thr
    acc_auto = (text_model.predict(Xt_te)[auto] == yt_te[auto]).mean() if auto.any() else np.nan
    rows.append({"confidence_threshold": thr,
                 "auto_handled": round(auto.mean(), 3),
                 "accuracy_on_auto": round(acc_auto, 4),
                 "escalated_to_LLM": round(1 - auto.mean(), 3)})
routing = pd.DataFrame(rows).set_index("confidence_threshold")
routing

# %% [markdown]
# That table is a cost model. At a 0.9 confidence threshold you handle most
# tickets locally at high accuracy for effectively nothing, and pay for the
# expensive model only on the remainder. The accuracy on the auto-handled slice
# rises with the threshold — which is the whole point, and it is only trustworthy
# if the confidences are **calibrated**. Module 08 is a prerequisite for this
# pattern, not an aside.

# %% [markdown]
# ## 9. Two more feature-extraction tools
#
# `DictVectorizer` and `FeatureHasher` are for feature dictionaries rather than
# free text — the natural representation when features come from an event log or
# a JSON payload.

# %%
from sklearn.feature_extraction import DictVectorizer, FeatureHasher

events = [
    {"merchant": "AMZN", "hour": 14, "device": "ios"},
    {"merchant": "STEAM", "hour": 3, "device": "web", "vpn": 1},
    {"merchant": "AMZN", "hour": 22, "device": "android"},
]
dv = DictVectorizer(sparse=False)
pd.DataFrame(dv.fit_transform(events), columns=dv.get_feature_names_out())

# %% [markdown]
# `DictVectorizer` one-hots string values and passes numbers through, handling
# missing keys as zeros. `FeatureHasher` does the same thing statelessly, into a
# fixed number of buckets — the same trade as `HashingVectorizer`.

# %% [markdown]
# ---
# ## Exercises

# %% [markdown]
# ### Exercise 10.1 — Vectoriser bake-off
#
# On the support tickets, run a proper `GridSearchCV` over the vectoriser itself:
# `ngram_range`, `min_df`, `sublinear_tf`, `analyzer` (word vs char_wb), `norm`,
# and `use_idf` — with `LogisticRegression` fixed.
#
# 1. Report the best configuration and its macro-F1.
# 2. Plot macro-F1 against vocabulary size. Where does it saturate?
# 3. Find the configuration with the best *score per feature* and say whether you
#    would ship it instead.

# %%
# Your code here.


# %% [markdown]
# ### Exercise 10.2 — Find the ceiling
#
# The tickets dataset has ~5% label noise and ~16% deliberately ambiguous
# messages, so there is a hard accuracy ceiling.
#
# 1. Estimate that ceiling empirically. (Hint: the ambiguous messages are drawn
#    from a fixed pool — find duplicated texts with conflicting labels.)
# 2. Compare your best model's error rate to it.
# 3. For the residual errors, classify them by hand into: genuinely ambiguous,
#    mislabelled, and model failure. What fraction is actually fixable?
#
# This is the most valuable exercise in the module. "The model is 84% accurate
# and the ceiling is 87%" is a completely different conversation from "the model
# is 84% accurate".

# %%
# Your code here.


# %% [markdown]
# ### Exercise 10.3 — Negation
#
# Build a small dataset of 60 sentence pairs that differ only by a negation.
# Measure how well each representation separates them: unigram TF-IDF,
# bigram TF-IDF, trigram TF-IDF, char_wb, and LSA.
#
# Then design a `FunctionTransformer` that explicitly marks negation scope (e.g.
# rewriting "not authorised the payment" as "not_authorised not_the
# not_payment"), add it to the pipeline, and measure the gain. Is it worth the
# complexity?

# %%
# Your code here.


# %% [markdown]
# ### Exercise 10.4 — Embeddings, for real
#
# Install `sentence-transformers` and implement the `SentenceEmbedder` from
# section 7. Then:
#
# 1. Verify it passes `sklearn.utils.estimator_checks.check_estimator` (you may
#    need to relax some checks — say which, and why).
# 2. Compare TF-IDF, LSA and embeddings on the tickets data: macro-F1, fit time,
#    inference latency per 1,000 documents, and model size on disk.
# 3. Repeat with only 200 labelled training examples. Does the ranking change?
#    (This is the scenario where pre-training pays.)
# 4. Use `GridSearchCV` to compare two embedding models by searching over
#    `embed__model_name`.
# 5. Write the recommendation for a bank routing 50,000 tickets a day, being
#    explicit about latency, cost and explainability.

# %%
# Your code here.


# %% [markdown]
# ### Exercise 10.5 — A cost-aware routing system
#
# Extend section 8's routing table into a full cost model. Assume: local model
# ~£0 per prediction, LLM API £0.004 per call with 800 ms latency, and a
# misclassification costs £3 in rework.
#
# 1. Find the confidence threshold minimising total cost.
# 2. Show what happens when the local model's confidences are *not* calibrated —
#    use an uncalibrated `RandomForest` — and quantify the extra cost.
# 3. Add a third tier (human review) for the lowest-confidence band and optimise
#    the two thresholds jointly.
# 4. Present it as the one-page business case.

# %%
# Your code here.


# %% [markdown]
# ---
# ## Takeaways
#
# - `min_df` and `ngram_range` are the two vectoriser parameters that earn their
#   keep. `stop_words="english"` deletes "not" — think before using it.
# - **Character n-grams (`char_wb`) are robust to typos** and often beat word
#   n-grams on real customer text.
# - A linear model over TF-IDF gives an **exact per-token explanation**, computed
#   instantly. That property is genuinely valuable and it is the first thing you
#   give up when you move to embeddings.
# - `ColumnTransformer` takes a *string* column name for a vectoriser, not a list.
# - Bag-of-words cannot represent synonymy, negation or word order. That is the
#   honest reason to move to embeddings — not fashion.
# - An embedding model wrapped as a transformer composes with every sklearn tool
#   you already know. **The harness outlives the representation.**
# - In an LLM stack, sklearn is the classifier head, the retrieval index, the
#   cheap router, the guardrail, and the entire evaluation harness.
#
# **Next:** Module 11 — writing your own estimators so they behave like
# first-class citizens of the library.
