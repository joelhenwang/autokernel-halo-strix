# Part 08: Data Pipeline — CLIMB Mixture Optimization

## Goal
Build a data pipeline that samples, clusters, quality-filters, and optimally mixes training data from large web corpora. By the end, you will have a single `.bin` file with the best possible mixture of domains for your model size, ready for the training loop from Part 09.

## Why This Matters
A mediocre model on excellent data beats an excellent model on mediocre data. This part is where your benchmarks will gain the most points per hour invested.

---

## 8.1 Why Data Quality Matters More Than Architecture

### The Uncomfortable Truth

Here is an experiment you can run yourself: take GPT-2 124M (the exact same architecture, same hyperparameters) and train it twice:

1. On random web crawl data (The Pile, unfiltered)
2. On curated educational text (FineWeb-Edu)

The results are not close. Run 2 will beat run 1 on HellaSwag by 5-8 percentage points — the equivalent of doubling model size.

This is not a hypothetical. SmolLM2-135M (HuggingFace, 2024) trained on 2T carefully curated tokens outperforms models 2-3x its size that were trained on lower-quality data. The lesson: **data mixture is a hyperparameter, and it might be your most important one.**

### What "Data Quality" Actually Means

Quality is not a single number. It decomposes into:

| Dimension | Description | Example of Low Quality |
|-----------|-------------|----------------------|
| **Linguistic quality** | Grammar, coherence, fluency | OCR garbage, machine-translated text |
| **Information density** | Facts and reasoning per token | SEO spam, boilerplate legal text |
| **Domain relevance** | Matches downstream evaluation | Training on code when evaluating prose |
| **Diversity** | Coverage of topics and styles | 80% of data from one domain |
| **Decontamination** | No benchmark leakage | Test set answers in training data |

### Data Mixture: The Outsized Lever

A "mixture" is the proportion of tokens drawn from each domain. For example:

```
Web text:       40%
Books:          20%
Wikipedia:      15%
Code:           10%
Academic:       10%
Conversations:   5%
```

Changing these proportions — with the same total token count, same model, same hyperparameters — can swing HellaSwag by 3-5 points. The problem: the space of possible mixtures is enormous, and intuition is unreliable. You need a systematic search.

---

## 8.2 The CLIMB Method (NVIDIA)

CLIMB (CLustering-Informed Mixture of data Before training) is a method from NVIDIA (2024) that finds near-optimal data mixtures without training hundreds of models. The key insight: **you can predict which mixture works best using a cheap surrogate model instead of full training runs.**

### The Algorithm in Plain English

```
1. Take your raw data (millions of documents)
2. Embed each document into a vector (sentence-transformers)
3. Cluster the vectors into K groups (FAISS k-means)
4. Sample N random mixtures (Dirichlet distribution)
5. For each mixture, train a tiny proxy model for a few hundred steps
6. Fit a surrogate (LightGBM) that maps mixture weights -> loss
7. Use the surrogate to find the optimal mixture
8. Assemble your final dataset with those proportions
```

### Why Proxy Models Work

The critical question: does the best mixture for a 10M-parameter proxy transfer to your real 124M-350M model?

Yes. NVIDIA showed that mixture preferences are remarkably stable across model sizes. The ranking of mixtures barely changes from 25M to 1B parameters. A mixture that produces low loss on a tiny model also produces low loss on a large model. This is why CLIMB works: you can do the expensive search at 1/100th the cost.

### Why LightGBM Instead of More Training Runs

Even with proxy models, you cannot try all mixtures. With K=16 clusters, the mixture space is a 15-dimensional simplex — astronomically large. The solution:

1. Train ~50 proxy models with random mixtures (a few hours total)
2. Fit a LightGBM regressor: inputs = 16 mixture weights, output = validation loss
3. Optimize the LightGBM prediction (milliseconds)

The surrogate gives you a smooth, differentiable-ish approximation of the loss landscape. It is not perfect, but it is vastly better than uniform mixing.

---

## 8.3 Phase-by-Phase Implementation

### Prerequisites

```bash
pip install sentence-transformers faiss-cpu lightgbm datasets tiktoken numpy
```

We will work through four phases:

```
Phase 0: Sample documents from HuggingFace
Phase 1: Embed documents (CPU)
Phase 2: Cluster with FAISS k-means
Phase 3: Proxy model search + LightGBM surrogate
```

### Phase 0: Sampling Documents

**Never download the full dataset.** FineWeb-Edu is ~50TB. We stream a small sample.

```python
"""phase0_sample.py — Stream-sample documents from HuggingFace."""
import json
import random
from pathlib import Path
from datasets import load_dataset

# Configuration
DATASET = "HuggingFaceFW/fineweb-edu"
SUBSET = "sample-10BT"        # 10B token sample (still large, we subsample further)
TARGET_DOCS = 100_000         # How many documents to keep
OUTPUT_DIR = Path("data/sampled")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Stream — no disk download of the full dataset
ds = load_dataset(DATASET, SUBSET, split="train", streaming=True)

# Reservoir sampling: keep TARGET_DOCS from the stream
reservoir = []
for i, example in enumerate(ds):
    text = example["text"]
    
    # Basic length filter: skip very short or very long documents
    word_count = len(text.split())
    if word_count < 50 or word_count > 5000:
        continue
    
    if len(reservoir) < TARGET_DOCS:
        reservoir.append(text)
    else:
        # Replace with decreasing probability
        j = random.randint(0, i)
        if j < TARGET_DOCS:
            reservoir[j] = text
    
    if i % 50_000 == 0:
        print(f"  Scanned {i:,} documents, kept {len(reservoir):,}")
    
    # Safety cap: don't scan more than 2M documents
    if i >= 2_000_000:
        break

print(f"Sampled {len(reservoir):,} documents")

# Save as JSONL
output_path = OUTPUT_DIR / "sampled_docs.jsonl"
with open(output_path, "w") as f:
    for doc in reservoir:
        f.write(json.dumps({"text": doc}) + "\n")

print(f"Saved to {output_path} ({output_path.stat().st_size / 1e6:.1f} MB)")
```

**Runtime:** 5-15 minutes depending on network speed. You should see ~100K documents in ~200-500 MB of JSONL.

**Verification checkpoint:**
```bash
wc -l data/sampled/sampled_docs.jsonl
# Should print: 100000
python3 -c "
import json
with open('data/sampled/sampled_docs.jsonl') as f:
    doc = json.loads(f.readline())
    print(f'First doc length: {len(doc[\"text\"].split())} words')
    print(doc['text'][:200])
"
```

### Phase 1: Embedding

We embed each document into a 384-dimensional vector using `all-MiniLM-L6-v2`. This runs on CPU — the model is tiny and embedding 100K documents takes only a few minutes.

```python
"""phase1_embed.py — Embed sampled documents with sentence-transformers."""
import json
import numpy as np
from pathlib import Path
from sentence_transformers import SentenceTransformer

INPUT_PATH = Path("data/sampled/sampled_docs.jsonl")
OUTPUT_DIR = Path("data/embeddings")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Load documents
print("Loading documents...")
documents = []
with open(INPUT_PATH) as f:
    for line in f:
        documents.append(json.loads(line)["text"])
print(f"Loaded {len(documents):,} documents")

# Load model (CPU is fine — this model is 80MB)
print("Loading embedding model...")
model = SentenceTransformer("all-MiniLM-L6-v2", device="cpu")

# Embed in batches
# We embed only the first 512 characters of each document (enough for topic classification)
print("Embedding documents...")
truncated = [doc[:512] for doc in documents]
embeddings = model.encode(
    truncated,
    batch_size=256,
    show_progress_bar=True,
    normalize_embeddings=True,  # L2 normalize for cosine similarity
)

# Save
embeddings = np.array(embeddings, dtype=np.float32)
np.save(OUTPUT_DIR / "embeddings.npy", embeddings)
print(f"Saved embeddings: shape={embeddings.shape}, dtype={embeddings.dtype}")
print(f"File size: {(OUTPUT_DIR / 'embeddings.npy').stat().st_size / 1e6:.1f} MB")
```

**Runtime:** 2-5 minutes on a Zen 3 CPU for 100K documents.

**Expected output:** `embeddings.npy` with shape `(100000, 384)`, approximately 150 MB.

### Phase 2: K-Means Clustering

We cluster the 100K embeddings into K=16 groups using FAISS. Each cluster represents a "domain" — FAISS will discover natural groupings like code, science, news, fiction, etc.

```python
"""phase2_cluster.py — K-means clustering with FAISS."""
import json
import numpy as np
from pathlib import Path
import faiss

INPUT_EMBEDDINGS = Path("data/embeddings/embeddings.npy")
INPUT_DOCS = Path("data/sampled/sampled_docs.jsonl")
OUTPUT_DIR = Path("data/clusters")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

K = 16  # Number of clusters

# Load embeddings
print("Loading embeddings...")
embeddings = np.load(INPUT_EMBEDDINGS)
n, d = embeddings.shape
print(f"Loaded {n:,} embeddings of dimension {d}")

# K-means with FAISS (CPU, fast)
print(f"Running k-means with K={K}...")
kmeans = faiss.Kmeans(
    d=d,
    k=K,
    niter=50,       # 50 iterations is plenty for 100K points
    verbose=True,
    seed=42,
)
kmeans.train(embeddings)

# Assign each document to its nearest cluster
_, assignments = kmeans.index.search(embeddings, 1)
assignments = assignments.flatten()

# Report cluster sizes
print("\nCluster sizes:")
for c in range(K):
    count = (assignments == c).sum()
    print(f"  Cluster {c:2d}: {count:6,} documents ({count / n * 100:.1f}%)")

# Save assignments
np.save(OUTPUT_DIR / "assignments.npy", assignments)

# Save per-cluster document indices for later
cluster_indices = {}
for c in range(K):
    indices = np.where(assignments == c)[0].tolist()
    cluster_indices[c] = indices

with open(OUTPUT_DIR / "cluster_indices.json", "w") as f:
    json.dump(cluster_indices, f)

# Inspect clusters: print first 100 characters of 3 random docs per cluster
print("\n--- Cluster Samples ---")
documents = []
with open(INPUT_DOCS) as f:
    for line in f:
        documents.append(json.loads(line)["text"])

import random
random.seed(42)
for c in range(K):
    print(f"\nCluster {c}:")
    samples = random.sample(cluster_indices[c], min(3, len(cluster_indices[c])))
    for idx in samples:
        print(f"  [{idx}] {documents[idx][:100]}...")

print(f"\nSaved to {OUTPUT_DIR}")
```

**Runtime:** Under 1 minute for 100K points on CPU.

**What to look for:** Clusters should be semantically coherent. You might see one cluster dominated by code, another by news, another by academic text. If all clusters look the same, try increasing K to 32.

### Phase 3: Proxy Model Search

This is the core of CLIMB. We train many tiny models on different mixtures, then fit a surrogate to predict the best mixture without more training.

```python
"""phase3_proxy_search.py — Train proxy models + LightGBM surrogate."""
import json
import numpy as np
import torch
import torch.nn as nn
from pathlib import Path
import lightgbm as lgb
from scipy.optimize import minimize

# Configuration
K = 16
NUM_MIXTURES = 50          # Number of random mixtures to try
PROXY_STEPS = 300          # Training steps per proxy model
PROXY_DIM = 256            # Hidden dim for proxy (tiny model)
PROXY_LAYERS = 4           # Number of layers
SEQ_LEN = 256              # Context length for proxy
BATCH_SIZE = 16
VOCAB_SIZE = 50257         # GPT-2 tokenizer
DEVICE = "cuda"

DATA_DIR = Path("data")
OUTPUT_DIR = Path("data/mixture_search")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# -------------------------------------------------------------------
# Step 1: Tokenize documents per cluster
# -------------------------------------------------------------------
import tiktoken
enc = tiktoken.get_encoding("gpt2")

print("Tokenizing per-cluster documents...")
cluster_indices = json.load(open(DATA_DIR / "clusters/cluster_indices.json"))
documents = []
with open(DATA_DIR / "sampled/sampled_docs.jsonl") as f:
    for line in f:
        documents.append(json.loads(line)["text"])

cluster_tokens = {}
for c in range(K):
    tokens = []
    for idx in cluster_indices[str(c)]:
        encoded = enc.encode_ordinary(documents[idx])
        tokens.extend(encoded)
        tokens.append(50256)  # EOS between documents
    cluster_tokens[c] = torch.tensor(tokens, dtype=torch.long)
    print(f"  Cluster {c}: {len(cluster_tokens[c]):,} tokens")

# -------------------------------------------------------------------
# Step 2: Define the tiny proxy model
# -------------------------------------------------------------------
class ProxyModel(nn.Module):
    """Minimal GPT for mixture search. Not meant to be good — just consistent."""
    def __init__(self, vocab_size, d_model, n_layers, seq_len):
        super().__init__()
        self.tok_emb = nn.Embedding(vocab_size, d_model)
        self.pos_emb = nn.Embedding(seq_len, d_model)
        layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=4, dim_feedforward=d_model * 4,
            dropout=0.0, batch_first=True, norm_first=True,
        )
        self.transformer = nn.TransformerEncoder(layer, num_layers=n_layers)
        self.ln_f = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, vocab_size, bias=False)
        self.seq_len = seq_len
    
    def forward(self, x):
        B, T = x.shape
        pos = torch.arange(T, device=x.device).unsqueeze(0)
        h = self.tok_emb(x) + self.pos_emb(pos)
        # Causal mask
        mask = torch.nn.Transformer.generate_square_subsequent_mask(T, device=x.device)
        h = self.transformer(h, mask=mask, is_causal=True)
        h = self.ln_f(h)
        return self.head(h)

# -------------------------------------------------------------------
# Step 3: Batch sampler that respects mixture weights
# -------------------------------------------------------------------
def sample_batch(cluster_tokens, weights, batch_size, seq_len):
    """Sample a batch according to mixture weights."""
    # How many sequences from each cluster
    counts = np.random.multinomial(batch_size, weights)
    sequences = []
    for c, count in enumerate(counts):
        tokens = cluster_tokens[c]
        for _ in range(count):
            if len(tokens) <= seq_len:
                seq = tokens[:seq_len]
            else:
                start = np.random.randint(0, len(tokens) - seq_len)
                seq = tokens[start:start + seq_len]
            sequences.append(seq)
    
    # Shuffle order within batch
    np.random.shuffle(sequences)
    return torch.stack(sequences[:batch_size])

# -------------------------------------------------------------------
# Step 4: Train one proxy model and return validation loss
# -------------------------------------------------------------------
def train_proxy(weights, cluster_tokens, steps=PROXY_STEPS):
    """Train a proxy model with given mixture weights. Return final val loss."""
    model = ProxyModel(VOCAB_SIZE, PROXY_DIM, PROXY_LAYERS, SEQ_LEN).to(DEVICE)
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4)
    
    model.train()
    losses = []
    for step in range(steps):
        batch = sample_batch(cluster_tokens, weights, BATCH_SIZE, SEQ_LEN + 1)
        batch = batch.to(DEVICE)
        x, y = batch[:, :-1], batch[:, 1:]
        
        logits = model(x)
        loss = nn.functional.cross_entropy(logits.reshape(-1, VOCAB_SIZE), y.reshape(-1))
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        if step >= steps - 50:  # Average last 50 steps as validation
            losses.append(loss.item())
    
    return np.mean(losses)

# -------------------------------------------------------------------
# Step 5: Sample random mixtures and train proxies
# -------------------------------------------------------------------
print(f"\nTraining {NUM_MIXTURES} proxy models...")
results = []

for i in range(NUM_MIXTURES):
    # Dirichlet sampling: random mixture weights that sum to 1
    # alpha=1.0 gives uniform distribution over the simplex
    weights = np.random.dirichlet(np.ones(K))
    
    loss = train_proxy(weights, cluster_tokens)
    results.append({"weights": weights.tolist(), "loss": loss})
    
    print(f"  Mixture {i+1}/{NUM_MIXTURES}: loss={loss:.4f}  "
          f"[{', '.join(f'{w:.2f}' for w in weights[:5])}...]")

# Save raw results
with open(OUTPUT_DIR / "proxy_results.json", "w") as f:
    json.dump(results, f, indent=2)

# -------------------------------------------------------------------
# Step 6: Fit LightGBM surrogate
# -------------------------------------------------------------------
print("\nFitting LightGBM surrogate...")
X = np.array([r["weights"] for r in results])
y = np.array([r["loss"] for r in results])

model_lgb = lgb.LGBMRegressor(
    n_estimators=200,
    max_depth=6,
    learning_rate=0.05,
    min_child_samples=3,
    verbose=-1,
)
model_lgb.fit(X, y)

# Report feature importance (which clusters matter most)
importances = model_lgb.feature_importances_
print("\nCluster importance for loss prediction:")
for c in sorted(range(K), key=lambda c: -importances[c]):
    print(f"  Cluster {c:2d}: importance={importances[c]:4d}")

# -------------------------------------------------------------------
# Step 7: Optimize the surrogate to find best mixture
# -------------------------------------------------------------------
print("\nOptimizing surrogate...")

def objective(w):
    """Predict loss from mixture weights."""
    return model_lgb.predict(w.reshape(1, -1))[0]

best_loss = float("inf")
best_weights = None

# Multi-start optimization (simplex constraint: weights sum to 1, all >= 0)
for _ in range(1000):
    w0 = np.random.dirichlet(np.ones(K))
    result = minimize(
        objective,
        w0,
        method="L-BFGS-B",
        bounds=[(0.01, 1.0)] * K,  # Minimum 1% per cluster
    )
    # Re-normalize to sum to 1
    w_opt = result.x / result.x.sum()
    predicted_loss = objective(w_opt)
    
    if predicted_loss < best_loss:
        best_loss = predicted_loss
        best_weights = w_opt

print(f"\nOptimal mixture (predicted loss={best_loss:.4f}):")
for c in range(K):
    bar = "#" * int(best_weights[c] * 50)
    print(f"  Cluster {c:2d}: {best_weights[c]:.3f}  {bar}")

# Save optimal mixture
mixture_config = {
    "k": K,
    "weights": best_weights.tolist(),
    "predicted_loss": float(best_loss),
    "num_proxy_runs": NUM_MIXTURES,
    "proxy_steps": PROXY_STEPS,
}
with open(OUTPUT_DIR / "mixture_config.json", "w") as f:
    json.dump(mixture_config, f, indent=2)

print(f"\nSaved to {OUTPUT_DIR / 'mixture_config.json'}")
```

**Runtime:** 50 proxy models x 300 steps each = ~15K training steps on your RTX 4060 Ti. With the tiny proxy model, this should take 30-60 minutes.

**What to expect:** The optimal mixture will NOT be uniform. Some clusters will be heavily upweighted (high-quality educational text, Wikipedia-like content) and others nearly zeroed (boilerplate, low-quality web text). The predicted loss from the surrogate should be 0.1-0.3 lower than the average random mixture.

**Verification checkpoint:**
```bash
cat data/mixture_search/mixture_config.json
# Should show: K=16, weights that sum to ~1.0, predicted_loss < mean of proxy results
python3 -c "
import json
config = json.load(open('data/mixture_search/mixture_config.json'))
print(f'Weights sum: {sum(config[\"weights\"]):.4f}')
print(f'Max cluster weight: {max(config[\"weights\"]):.3f}')
print(f'Min cluster weight: {min(config[\"weights\"]):.3f}')
"
```

---

## 8.4 Self-Improving Quality Filtering

Clustering tells you WHAT kind of data you have. Quality filtering tells you which documents within each cluster are worth keeping. We use a two-stage approach: LLM scoring on a sample, then a trained classifier on the rest.

### Stage 1: LLM Scoring

Score 1,000 documents per cluster using an LLM API. This costs about $1-2 total for 16K calls with a cheap model.

```python
"""phase4_quality_score.py — LLM-based quality scoring."""
import json
import random
import time
from pathlib import Path
from openai import OpenAI  # Works with any OpenAI-compatible API

# Configuration
K = 16
SAMPLES_PER_CLUSTER = 1000
API_BASE = "http://localhost:8080/v1"  # Local LLM server, or use OpenAI
API_KEY = "your-api-key"              # Or "not-needed" for local
MODEL = "gpt-4o-mini"                 # Cheapest option that works

DATA_DIR = Path("data")
OUTPUT_DIR = Path("data/quality")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

client = OpenAI(base_url=API_BASE, api_key=API_KEY)

cluster_indices = json.load(open(DATA_DIR / "clusters/cluster_indices.json"))
documents = []
with open(DATA_DIR / "sampled/sampled_docs.jsonl") as f:
    for line in f:
        documents.append(json.loads(line)["text"])

QUALITY_PROMPT = """Rate the educational and informational quality of this text on a scale of 0-5.

0 = Gibberish, spam, or completely uninformative
1 = Very low quality (boilerplate, ads, repetitive)
2 = Low quality (shallow, poorly written but has some content)
3 = Medium quality (informative but not exceptional)
4 = High quality (well-written, educational, informative)
5 = Excellent quality (could be in a textbook or encyclopedia)

Respond with ONLY a single digit 0-5, nothing else.

Text:
{text}"""

all_scores = {}

for c in range(K):
    indices = cluster_indices[str(c)]
    sample_indices = random.sample(indices, min(SAMPLES_PER_CLUSTER, len(indices)))
    
    scores = []
    for i, idx in enumerate(sample_indices):
        text = documents[idx][:1000]  # First 1000 chars only
        
        try:
            response = client.chat.completions.create(
                model=MODEL,
                messages=[{"role": "user", "content": QUALITY_PROMPT.format(text=text)}],
                max_tokens=2,
                temperature=0.0,
            )
            score = int(response.choices[0].message.content.strip()[0])
            score = max(0, min(5, score))
        except Exception:
            score = 3  # Default to medium on API errors
        
        scores.append({"doc_idx": idx, "score": score})
        
        if (i + 1) % 100 == 0:
            avg = sum(s["score"] for s in scores) / len(scores)
            print(f"  Cluster {c}, scored {i+1}/{len(sample_indices)}, avg={avg:.2f}")
    
    all_scores[c] = scores
    avg = sum(s["score"] for s in scores) / len(scores)
    print(f"Cluster {c}: avg score = {avg:.2f}")

with open(OUTPUT_DIR / "llm_scores.json", "w") as f:
    json.dump(all_scores, f, indent=2)

print(f"Saved {sum(len(v) for v in all_scores.values())} scores")
```

### Stage 2: Train a Fast Classifier

Now train a logistic regression on (embedding -> quality score). This classifier can score ALL 100K documents in seconds, not hours.

```python
"""phase4b_quality_classifier.py — Train quality classifier from LLM scores."""
import json
import numpy as np
from pathlib import Path
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score
import pickle

DATA_DIR = Path("data")
OUTPUT_DIR = Path("data/quality")

# Load
embeddings = np.load(DATA_DIR / "embeddings/embeddings.npy")
scores_raw = json.load(open(OUTPUT_DIR / "llm_scores.json"))

# Flatten into arrays
train_indices = []
train_labels = []
for c, score_list in scores_raw.items():
    for entry in score_list:
        train_indices.append(entry["doc_idx"])
        # Binary: 1 if score >= 3, else 0
        train_labels.append(1 if entry["score"] >= 3 else 0)

X_train = embeddings[train_indices]
y_train = np.array(train_labels)

print(f"Training classifier on {len(y_train)} labeled samples")
print(f"  Positive (quality >= 3): {y_train.sum()} ({y_train.mean()*100:.1f}%)")
print(f"  Negative (quality < 3):  {(1 - y_train).sum()} ({(1 - y_train.mean())*100:.1f}%)")

# Train logistic regression
clf = LogisticRegression(max_iter=1000, C=1.0)
cv_scores = cross_val_score(clf, X_train, y_train, cv=5, scoring="accuracy")
print(f"  5-fold CV accuracy: {cv_scores.mean():.3f} +/- {cv_scores.std():.3f}")

clf.fit(X_train, y_train)

# Score ALL documents
print("Scoring all documents...")
quality_probs = clf.predict_proba(embeddings)[:, 1]  # P(quality >= 3)

print(f"  Mean quality probability: {quality_probs.mean():.3f}")
print(f"  Documents above 0.5: {(quality_probs > 0.5).sum():,} / {len(quality_probs):,}")

np.save(OUTPUT_DIR / "quality_scores.npy", quality_probs)
with open(OUTPUT_DIR / "quality_classifier.pkl", "wb") as f:
    pickle.dump(clf, f)

print("Saved quality_scores.npy and quality_classifier.pkl")
```

### Filtering Strategy

Do not use a hard threshold. Instead, use quality scores as sampling weights — higher quality documents are sampled more often, but low quality documents are not completely excluded.

```python
# Soft filtering: quality score as sampling weight
# Documents with quality_prob=0.9 are 9x more likely to be sampled
# than documents with quality_prob=0.1
sampling_weights = quality_probs ** 2  # Square to sharpen the distribution
```

---

## 8.5 Assembling the Final Dataset

Now we combine everything: mixture weights (from CLIMB) + quality scores (from the classifier) into a single pre-tokenized `.bin` file.

```python
"""phase5_assemble.py — Assemble final training dataset."""
import json
import numpy as np
import tiktoken
from pathlib import Path

DATA_DIR = Path("data")
OUTPUT_DIR = Path("data/assembled")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Load everything
mixture_config = json.load(open(DATA_DIR / "mixture_search/mixture_config.json"))
cluster_indices = json.load(open(DATA_DIR / "clusters/cluster_indices.json"))
quality_scores = np.load(DATA_DIR / "quality/quality_scores.npy")

documents = []
with open(DATA_DIR / "sampled/sampled_docs.jsonl") as f:
    for line in f:
        documents.append(json.loads(line)["text"])

K = mixture_config["k"]
weights = np.array(mixture_config["weights"])
enc = tiktoken.get_encoding("gpt2")

# Target: ~50M tokens for a training run
TARGET_TOKENS = 50_000_000
EOS_TOKEN = 50256

# Compute how many tokens per cluster
tokens_per_cluster = (weights * TARGET_TOKENS).astype(int)
print("Token budget per cluster:")
for c in range(K):
    print(f"  Cluster {c:2d}: {tokens_per_cluster[c]:>10,} tokens ({weights[c]:.3f})")

# Assemble
all_tokens = []
for c in range(K):
    target = tokens_per_cluster[c]
    indices = np.array(cluster_indices[str(c)])
    
    # Quality-weighted sampling within the cluster
    cluster_quality = quality_scores[indices]
    sampling_probs = cluster_quality ** 2
    sampling_probs = sampling_probs / sampling_probs.sum()
    
    cluster_tokens = []
    while len(cluster_tokens) < target:
        # Sample a document according to quality weights
        idx = np.random.choice(indices, p=sampling_probs)
        doc_tokens = enc.encode_ordinary(documents[idx])
        cluster_tokens.extend(doc_tokens)
        cluster_tokens.append(EOS_TOKEN)  # EOS between documents
    
    # Trim to target
    cluster_tokens = cluster_tokens[:target]
    all_tokens.extend(cluster_tokens)
    print(f"  Cluster {c:2d}: assembled {len(cluster_tokens):,} tokens")

# Shuffle at document boundaries (optional — helps with batching)
# For simplicity, we just concatenate in cluster order
all_tokens = np.array(all_tokens, dtype=np.uint16)  # vocab < 65536, so uint16 is fine

# Save as raw binary
output_path = OUTPUT_DIR / "train.bin"
all_tokens.tofile(output_path)

# Save metadata
metadata = {
    "num_tokens": len(all_tokens),
    "vocab_size": 50257,
    "dtype": "uint16",
    "mixture_weights": weights.tolist(),
    "quality_filtered": True,
    "source": "FineWeb-Edu (sampled)",
}
with open(OUTPUT_DIR / "metadata.json", "w") as f:
    json.dump(metadata, f, indent=2)

print(f"\nFinal dataset: {len(all_tokens):,} tokens ({output_path.stat().st_size / 1e6:.1f} MB)")
print(f"Saved to {output_path}")
```

**Verification checkpoint:**
```python
"""verify_dataset.py — Verify the assembled dataset."""
import numpy as np
import tiktoken
import json

data = np.fromfile("data/assembled/train.bin", dtype=np.uint16)
meta = json.load(open("data/assembled/metadata.json"))
enc = tiktoken.get_encoding("gpt2")

print(f"Total tokens: {len(data):,}")
print(f"Expected: ~{meta['num_tokens']:,}")
print(f"File size: {len(data) * 2 / 1e6:.1f} MB")

# Decode a sample
sample = data[1000:1100]
text = enc.decode(sample.tolist())
print(f"\nSample text (tokens 1000-1100):\n{text}")

# Check EOS distribution
eos_count = (data == 50256).sum()
avg_doc_len = len(data) / eos_count
print(f"\nEOS tokens: {eos_count:,}")
print(f"Average document length: {avg_doc_len:.0f} tokens")
```

---

## 8.6 MixtureDataset

Two modes: pre-mixed (load the assembled `.bin`, no overhead) and online-mixed (per-cluster `.bin` files, dynamic sampling for experiments).

### Pre-Mixed Mode

```python
"""dataset.py — MixtureDataset for training."""
import numpy as np
import torch
from torch.utils.data import Dataset
from pathlib import Path

class MixtureDataset(Dataset):
    """
    Pre-mixed dataset: loads a single .bin file.
    No per-step overhead — data is already mixed and quality-filtered.
    """
    def __init__(self, bin_path, seq_len):
        self.data = np.memmap(bin_path, dtype=np.uint16, mode='r')
        self.seq_len = seq_len
        self.n_tokens = len(self.data)
        self.n_samples = self.n_tokens // (seq_len + 1)
        print(f"MixtureDataset: {self.n_tokens:,} tokens, {self.n_samples:,} samples")
    
    def __len__(self):
        return self.n_samples
    
    def __getitem__(self, idx):
        start = idx * (self.seq_len + 1)
        chunk = self.data[start:start + self.seq_len + 1].astype(np.int64)
        x = torch.from_numpy(chunk[:-1])
        y = torch.from_numpy(chunk[1:])
        return x, y

# Usage:
# ds = MixtureDataset("data/assembled/train.bin", seq_len=1024)
# loader = torch.utils.data.DataLoader(ds, batch_size=8, shuffle=True, num_workers=4)
```

### Online-Mixed Mode

For experiments where you want to change mixture weights without re-assembling:

```python
class OnlineMixtureDataset(Dataset):
    """
    Online-mixed dataset: samples from per-cluster .bin files at runtime.
    Slower than pre-mixed (random reads across files) but flexible.
    """
    def __init__(self, cluster_dir, weights, seq_len):
        self.seq_len = seq_len
        self.weights = np.array(weights, dtype=np.float64)
        self.weights /= self.weights.sum()
        
        # Load per-cluster data
        self.cluster_data = []
        for c in range(len(weights)):
            path = Path(cluster_dir) / f"cluster_{c:02d}.bin"
            data = np.memmap(path, dtype=np.uint16, mode='r')
            self.cluster_data.append(data)
        
        self.total_tokens = sum(len(d) for d in self.cluster_data)
        self.n_samples = self.total_tokens // (seq_len + 1)
        print(f"OnlineMixtureDataset: {self.total_tokens:,} tokens across "
              f"{len(weights)} clusters")
    
    def __len__(self):
        return self.n_samples
    
    def __getitem__(self, idx):
        # Pick cluster according to mixture weights
        c = np.random.choice(len(self.weights), p=self.weights)
        data = self.cluster_data[c]
        
        # Random position within the cluster
        max_start = len(data) - self.seq_len - 1
        if max_start <= 0:
            start = 0
        else:
            start = np.random.randint(0, max_start)
        
        chunk = data[start:start + self.seq_len + 1].astype(np.int64)
        x = torch.from_numpy(chunk[:-1])
        y = torch.from_numpy(chunk[1:])
        return x, y
```

### Creating Per-Cluster .bin Files

```python
"""split_clusters.py — Create per-cluster .bin files for online mixing."""
import json
import numpy as np
import tiktoken
from pathlib import Path

DATA_DIR = Path("data")
OUTPUT_DIR = Path("data/clusters_tokenized")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

cluster_indices = json.load(open(DATA_DIR / "clusters/cluster_indices.json"))
quality_scores = np.load(DATA_DIR / "quality/quality_scores.npy")
enc = tiktoken.get_encoding("gpt2")

documents = []
with open(DATA_DIR / "sampled/sampled_docs.jsonl") as f:
    for line in f:
        documents.append(json.loads(line)["text"])

for c in range(16):
    indices = cluster_indices[str(c)]
    tokens = []
    for idx in indices:
        if quality_scores[idx] < 0.3:  # Hard filter for very low quality
            continue
        doc_tokens = enc.encode_ordinary(documents[idx])
        tokens.extend(doc_tokens)
        tokens.append(50256)
    
    arr = np.array(tokens, dtype=np.uint16)
    output_path = OUTPUT_DIR / f"cluster_{c:02d}.bin"
    arr.tofile(output_path)
    print(f"Cluster {c:2d}: {len(arr):>10,} tokens -> {output_path}")
```

---

## Exercises

1. **Run the full pipeline on FineWeb-Edu with 10K documents** (not 100K) as a fast test. Verify that you get 16 clusters, a `mixture_config.json`, and a `train.bin` file. The whole pipeline should complete in under 15 minutes with 10K docs.

2. **Visualize cluster distribution.** Use the following to produce a 2D scatter plot:
   ```python
   """visualize_clusters.py"""
   import numpy as np
   import matplotlib.pyplot as plt
   from sklearn.decomposition import PCA
   
   embeddings = np.load("data/embeddings/embeddings.npy")
   assignments = np.load("data/clusters/assignments.npy")
   
   # PCA to 2D for visualization
   pca = PCA(n_components=2)
   coords = pca.fit_transform(embeddings[:5000])  # Subsample for speed
   
   plt.figure(figsize=(12, 8))
   scatter = plt.scatter(coords[:, 0], coords[:, 1],
                         c=assignments[:5000], cmap='tab20', s=2, alpha=0.5)
   plt.colorbar(scatter, label="Cluster")
   plt.title("Document Clusters (PCA projection)")
   plt.xlabel("PC1")
   plt.ylabel("PC2")
   plt.savefig("data/cluster_visualization.png", dpi=150)
   plt.show()
   ```

3. **Compare training with uniform vs optimized mixture.** Train the proxy model for 1000 steps with (a) uniform weights `[1/K] * K` and (b) your optimized weights. Plot the loss curves on the same axes. You should see the optimized mixture converge to a lower loss.

---

## Checkpoint

Before moving to Part 09, verify:
- [ ] `data/mixture_search/mixture_config.json` exists and contains K=16 weights that sum to ~1.0
- [ ] `data/assembled/train.bin` loads correctly as `np.uint16` and decodes to readable text
- [ ] EOS tokens (50256) appear between documents in the assembled data
- [ ] `MixtureDataset` returns (x, y) tensors of the correct shape
- [ ] You can explain why Dirichlet sampling covers the mixture space, and why a surrogate is cheaper than grid search

---

**Next: [Part 09 — Train-Evaluate-Benchmark Framework](09_train_eval_benchmark.md)**
