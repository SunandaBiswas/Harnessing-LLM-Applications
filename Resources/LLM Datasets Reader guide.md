

##  LLM Jailbreak Datasets — Full Hugging Face Catalog (30 datasets)

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
