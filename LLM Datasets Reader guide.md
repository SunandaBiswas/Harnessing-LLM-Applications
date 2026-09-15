# Datasets Reader
### Code snippets for loading public datasets into a Jupyter Notebook

A reference sheet of ready-to-run Python snippets, organized by data source. Copy the relevant cell into your notebook and adjust the dataset name/URL.

---

## 1. Hugging Face Hub (`datasets` library)

```bash
pip install datasets
```

**1.1 Basic load (text/NLP — IMDB reviews)**
```python
from datasets import load_dataset

dataset = load_dataset("imdb")
print(dataset)                     # shows train/test/unsupervised splits
dataset["train"][0]                # peek at one example
```

**1.2 Load a specific config/subset (e.g. GLUE — SST-2)**
```python
from datasets import load_dataset

dataset = load_dataset("glue", "sst2")
dataset["train"].to_pandas().head()
```

**1.3 Image dataset (CIFAR-10)**
```python
from datasets import load_dataset

ds = load_dataset("cifar10", split="train")
ds[0]["img"]          # PIL Image
ds[0]["label"]
```

**1.4 Audio dataset (Common Voice)**
```python
from datasets import load_dataset

ds = load_dataset("mozilla-foundation/common_voice_11_0", "en", split="train", streaming=True)
sample = next(iter(ds))
print(sample["audio"], sample["sentence"])
```

**1.5 Tabular / CSV-style dataset**
```python
from datasets import load_dataset

ds = load_dataset("scikit-learn/iris", split="train")
df = ds.to_pandas()
df.head()
```

**1.6 Streaming a large dataset (no full download)**
```python
from datasets import load_dataset

ds = load_dataset("c4", "en", split="train", streaming=True)
for example in ds.take(5):
    print(example["text"][:200])
```

**1.7 Converting any split to pandas**
```python
df = dataset["train"].to_pandas()
df.head()
```

---

## 2. Kaggle Datasets

```bash
pip install kaggle
# Place your kaggle.json API token in ~/.kaggle/kaggle.json first
```

```python
import kaggle
import pandas as pd

kaggle.api.dataset_download_files(
    "zynicide/wine-reviews",
    path="./data",
    unzip=True
)

df = pd.read_csv("./data/winemag-data-130k-v2.csv")
df.head()
```

---

## 3. UCI Machine Learning Repository

**3.1 Direct CSV/data file over HTTP**
```python
import pandas as pd

url = "https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data"
columns = ["sepal_length", "sepal_width", "petal_length", "petal_width", "class"]
df = pd.read_csv(url, names=columns)
df.head()
```

**3.2 Using the official `ucimlrepo` package**
```bash
pip install ucimlrepo
```
```python
from ucimlrepo import fetch_ucirepo

iris = fetch_ucirepo(id=53)         # id from the dataset's UCI page
X = iris.data.features
y = iris.data.targets
```

---

## 4. Scikit-learn Built-in / Fetchable Datasets

```python
from sklearn.datasets import load_iris, load_wine, fetch_california_housing
import pandas as pd

data = load_iris()
df = pd.DataFrame(data.data, columns=data.feature_names)
df["target"] = data.target
df.head()

# Larger, downloaded-on-first-use dataset
housing = fetch_california_housing(as_frame=True)
housing.frame.head()
```

---

## 5. TensorFlow Datasets (`tfds`)

```bash
pip install tensorflow-datasets
```
```python
import tensorflow_datasets as tfds

ds, info = tfds.load("mnist", split="train", as_supervised=True, with_info=True)
for image, label in ds.take(1):
    print(image.shape, label.numpy())
```

---

## 6. PyTorch `torchvision` / `torchtext` Datasets

```python
import torchvision

train_set = torchvision.datasets.CIFAR10(
    root="./data", train=True, download=True
)
img, label = train_set[0]
```

---

## 7. OpenML

```python
from sklearn.datasets import fetch_openml

titanic = fetch_openml(name="titanic", version=1, as_frame=True)
X = titanic.data
y = titanic.target
```

---

## 8. CSV / Excel / JSON Directly from a URL (e.g. GitHub raw)

```python
import pandas as pd

# CSV
url_csv = "https://raw.githubusercontent.com/owner/repo/main/data.csv"
df = pd.read_csv(url_csv)

# Excel
url_xlsx = "https://example.com/data.xlsx"
df_xlsx = pd.read_excel(url_xlsx)

# JSON
url_json = "https://raw.githubusercontent.com/owner/repo/main/data.json"
df_json = pd.read_json(url_json)

df.head()
```

---

## 9. Public REST APIs

**9.1 World Bank (economic indicators)**
```python
import requests
import pandas as pd

url = "http://api.worldbank.org/v2/country/BD/indicator/NY.GDP.MKTP.CD?format=json"
response = requests.get(url)
data = response.json()
df = pd.json_normalize(data[1])
df.head()
```

**9.2 Generic public JSON API**
```python
import requests
import pandas as pd

response = requests.get("https://api.publicapis.org/entries")
df = pd.DataFrame(response.json()["entries"])
df.head()
```

---

## 10. Google Drive (public shared file)

```python
import pandas as pd

file_id = "FILE_ID_HERE"
url = f"https://drive.google.com/uc?id={file_id}"
df = pd.read_csv(url)
```

---

## 11. Downloading & Extracting a ZIP Archive

```python
import requests, zipfile, io
import pandas as pd

url = "https://example.com/dataset.zip"
r = requests.get(url)
z = zipfile.ZipFile(io.BytesIO(r.content))
z.extractall("./data")

df = pd.read_csv("./data/extracted_file.csv")
df.head()
```

---

## 12. LLM Jailbreak Datasets — Full Hugging Face Catalog (30 datasets)

Mirrors the 30-dataset table from `LLM_Jailbreak_HuggingFace_Resources.md`, same order (sorted by downloads). A registry + loop is used instead of 30 hand-written cells — each dataset still lands in its **own separate DataFrame**, just addressed by key instead of 30 bespoke variable names, which is far easier to maintain and re-run.

```bash
pip install datasets huggingface_hub pandas
```

```python
from datasets import load_dataset
import pandas as pd

# (short_name, repo_id, config, split, note)
HF_JAILBREAK_DATASETS = [
    ("jbb_behaviors",          "JailbreakBench/JBB-Behaviors",                     "behaviors", "harmful", None),
    ("rubend18",                "rubend18/ChatGPT-Jailbreak-Prompts",               None, "train", None),
    ("jailbreakv_28k",          "JailbreakV-28K/JailBreakV-28k",                    None, "train", None),
    ("vigil_minilm",            "deadbits/vigil-jailbreak-all-MiniLM-L6-v2",        None, "train", "embeddings only"),
    ("frascuchon_mirror",       "frascuchon/ChatGPT-Jailbreak-Prompts",             None, "train", None),
    ("vigil_mpnet",             "deadbits/vigil-jailbreak-all-mpnet-base-v2",       None, "train", "embeddings only"),
    ("deepseek_v32exp",         "lvogel123/jailbreak-deepseek-v3.2-exp",            None, "train", None),
    ("vigil_ada002",            "deadbits/vigil-jailbreak-ada-002",                 None, "train", "embeddings only"),
    ("multiturn_attacks",       "tom-gibbs/multi-turn_jailbreak_attack_datasets",   None, "train", None),
    ("disaster_tweet",          "IDA-SERICS/Disaster-tweet-jailbreaking",           None, "train", None),
    ("jailbreak_complete_ds",   "GeorgeDaDude/Jailbreak_Complete_DS_labeled",       None, "train", None),
    ("walledai_jbb",            "walledai/JailbreakBench",                          None, "train", None),
    ("walledai_hub",            "walledai/JailbreakHub",                            None, "train", None),
    ("rs_jailbreaks",           "dhruvbpai/rs_jailbreaks",                          None, "train", None),
    ("evil_jailbreak",          "Granther/evil-jailbreak",                          None, "train", None),
    ("llama3_jailbreaks",       "Mechanistic-Anomaly-Detection/llama3-jailbreaks",  None, "train", None),
    ("gemma2_jailbreaks",       "Mechanistic-Anomaly-Detection/gemma2-jailbreaks",  None, "train", None),
    ("itw_jailbreak_prompts",   "TrustAIRLab/in-the-wild-jailbreak-prompts", "jailbreak_2023_12_25", "train", None),
    ("jailbreak_classification","jackhhao/jailbreak-classification",                None, "train", None),
    ("jailbreakdb",             "youbin2014/JailbreakDB",                           None, "train", "~6.6M rows — consider streaming=True"),
    ("all_prompt_jailbreak",    "AiActivity/All-Prompt-Jailbreak",                  None, "train", None),
    ("audiobench",              "researchtopic/Jailbreak-AudioBench",               None, "train", "audio modality"),
    ("audiobench_plus",         "researchtopic/Jailbreak-AudioBench-Plus",          None, "train", "audio modality"),
    ("necent_unified",          "Necent/llm-jailbreak-prompt-injection-dataset",    None, "train", "gated — needs login + accepted terms"),
    ("alice_persona",           "avalonsec/Prompt_engineering_Assistant_Alice_jailbreak", None, "train", None),
    ("semantic_router_detect",  "llm-semantic-router/jailbreak-detection-dataset",  None, "train", None),
    ("guardrail_benchmark",     "xunguangwang/JailbreakGuardrailBenchmark",         None, "train", None),
    ("simsonsun_prompts",       "Simsonsun/JailbreakPrompts",                       None, "train", "may be gated"),
    ("jailbreak_success",       "sevdeawesome/jailbreak_success",                   None, "train", None),
    ("guardrails_ai_detect",    "GuardrailsAI/detect-jailbreak",                    None, "train", "contains unsafe content by design"),
]

dfs, errors = {}, {}

for short_name, repo_id, config, split, note in HF_JAILBREAK_DATASETS:
    try:
        ds = load_dataset(repo_id, config, split=split) if config else load_dataset(repo_id, split=split)
        dfs[short_name] = ds.to_pandas()
        flag = f"  [{note}]" if note else ""
        print(f"OK   {short_name:<26} shape={dfs[short_name].shape}{flag}")
    except Exception as e:
        errors[short_name] = str(e)
        print(f"FAIL {short_name:<26} {repo_id}  ->  {type(e).__name__}: {e}")

print(f"\nLoaded {len(dfs)}/{len(HF_JAILBREAK_DATASETS)} datasets. {len(errors)} failed (see `errors` dict).")

# each dataset is its own DataFrame, accessed by key:
df_rubend18 = dfs["rubend18"]
df_jbb_behaviors = dfs["jbb_behaviors"]
# ...same pattern for any other key in `dfs`

# optional: also expose every entry as a top-level df_<name> variable in the notebook
globals().update({f"df_{k}": v for k, v in dfs.items()})
```

**Datasets that fail the standard loader** (custom loading scripts, no default config, non-HF-datasets file layout) — fall back to downloading the raw file directly:
```python
from huggingface_hub import HfApi, hf_hub_download

def load_hf_dataset_manual(repo_id, filename=None):
    api = HfApi()
    files = api.list_repo_files(repo_id, repo_type="dataset")
    if filename is None:
        filename = next(f for f in files if f.endswith((".csv", ".parquet", ".json", ".jsonl")))
    path = hf_hub_download(repo_id, filename, repo_type="dataset")
    if path.endswith(".csv"):
        return pd.read_csv(path)
    if path.endswith(".parquet"):
        return pd.read_parquet(path)
    return pd.read_json(path, lines=path.endswith(".jsonl"))

# retry any failed one manually, e.g.:
dfs["walledai_hub"] = load_hf_dataset_manual("walledai/JailbreakHub")
```

**Gated datasets** (`necent_unified`, possibly `simsonsun_prompts`) need a one-time login after accepting terms on the dataset's HF page:
```bash
huggingface-cli login          # paste a token from huggingface.co/settings/tokens
```

**Before analyzing any of these:** a few are flagged above as containing unsafe content by design or being very large — check each dataset card's license and content warning first, same as noted in the source catalog.

---

## Quick Tips

- **Large datasets:** prefer `streaming=True` (Hugging Face) or chunked reads (`pd.read_csv(url, chunksize=...)`) instead of loading everything into memory.
- **Repeated runs:** downloaded files are cached by default (`~/.cache/huggingface`, `~/.keras`, etc.) — no need to re-download each notebook run.
- **Auth-gated sources:** Kaggle needs an API token in `~/.kaggle/kaggle.json`; some Hugging Face datasets need `huggingface-cli login` first.
- **Converting to pandas:** almost every loader above ends the same way — `.to_pandas()`, `as_frame=True`, or a plain `pd.read_csv()` — so once loaded, all downstream analysis code looks the same regardless of source.

---

*Let me know which specific datasets you're using and I can add exact, ready-to-paste snippets for each.*
