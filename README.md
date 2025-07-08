# RAG-for-Code

**RAG-for-Code** is a **function-level retrieval-augmented generation (RAG) pipeline** that
automatically fetches relevant code from a target GitHub commit, inserts the retrieved
snippets into the prompt and calls an LLM (via **vLLM**) to complete the missing code.
It was designed for **line-completion** tasks such as *CC-Eval* and extends the ideas
popularised by recent research on code-oriented RAG frameworks like **CodeRAG** ([arxiv.org][1]).

```text
RAG-for-Code
│
├── BlocksCutting.py            # split a repo into per-function JSON blocks
├── FunctionsRetrieval.py       # BM25 / TF-IDF / Jaccard / Embedding ranker
├── GetInput.py                 # compose RAG prompts with retrieved code
├── RunModel.py                 # end-to-end pipeline (GitHub → RAG → vLLM)
├── EvaluatePred.py             # exact-match / edit / identifier metrics
├── ResultsConcluding.py        # aggregate experiment scores
│
├── data/                       # CC-Eval & other JSONL benchmarks
├── formalinput/                # prepared model-input files
├── output.jsonl                # example model outputs
└── code_completion_2554_cceval.zip   # pre-built pilot dataset
```


---

## Highlights

| Module                | What it does                                                                                                                               | Key script                                               |
| --------------------- | ------------------------------------------------------------------------------------------------------------------------------------------ | -------------------------------------------------------- |
| **Repo slicing**      | Walks every `.py` file in a repo, records each function’s name, class, location, intra-repo calls & imports, then dumps one big JSON index | `BlocksCutting.py` ([raw.githubusercontent.com][2])      |
| **Hybrid retrieval**  | Supports **BM25, TF-IDF, Jaccard** and **OpenAI-embedding** similarity; returns the *top-k* most relevant blocks                           | `FunctionsRetrieval.py` ([raw.githubusercontent.com][3]) |
| **Prompt builder**    | Wraps the retrieved blocks, identifier list and incomplete code into a single JSON-style instruction prompt                                | `GetInput.py` ([raw.githubusercontent.com][4])           |
| **Generation runner** | Downloads the exact commit via GitHub API, caches file contents, runs retrieval → prompt → vLLM generation → scoring in one command        | `RunModel.py` ([raw.githubusercontent.com][5])           |
| **Evaluation**        | Computes *Exact Match*, *Levenshtein similarity* and *Identifier overlap* on the first two lines of predicted code                         | `EvaluatePred.py` ([raw.githubusercontent.com][6])       |

---

## Quick Start

### 1 · Set up the environment

```bash
git clone https://github.com/XMZhangAI/RAG-for-code.git
cd RAG-for-code
python3 -m venv .venv && source .venv/bin/activate

# core deps
pip install torch transformers scikit-learn rank-bm25 vllm openai
```

> You will also need a **GitHub Personal Access Token** (set `GITHUB_TOKEN`
> in `RunModel.py`) for repository download.

### 2 · Index a repository

```python
import BlocksCutting, json, pathlib

repo_files = {p.as_posix(): p.read_text()        # fake example
              for p in pathlib.Path("./my_repo").rglob("*.py")}
BlocksCutting.BC_main(repo_files)                # writes `json_temp.json`
```

### 3 · Retrieve similar functions

```python
import FunctionsRetrieval, json
blocks = FunctionsRetrieval.run_FR(
            "json_temp.json",
            query       = "def decode_logits(logits):",
            files_content_arg = repo_files,
            rank_fn     = "bm25",
            top_n       = 3)
print(blocks[0])       # source of the most similar function
```

### 4 · End-to-end line completion (CC-Eval)

```bash
python RunModel.py \
        --data_path ./data/line_completion_oracle_openai_cosine_sim.jsonl \
        --model Salesforce/codegen25-7b-mono_P \
        --test_num 500 --test_start_line 1
```

This will

1. fetch each sample’s GitHub repo at the recorded commit,
2. build a JSON index → retrieve top-3 functions → craft a prompt,
3. call the model via **vLLM** and save predictions,
4. score them and print aggregated metrics.

---

## Repository Layout

| Path                              | Purpose                                                  |
| --------------------------------- | -------------------------------------------------------- |
| `BlocksCutting.py`                | AST-based code slicer producing a *function-block index* |
| `FunctionsRetrieval.py`           | Hybrid lexical + embedding ranker                        |
| `GetInput.py`                     | Prompt template with step-by-step JSON schema            |
| `RunModel.py`                     | Orchestrates GitHub download → retrieval → generation    |
| `EvaluatePred.py`                 | Metrics implementation                                   |
| `ResultsConcluding.py`            | Average results over a JSONL file                        |
| `data/`                           | Benchmark JSONL files (CC-Eval, etc.)                    |
| `formalinput/`                    | Ready-to-run input bundles                               |
| `code_completion_2554_cceval.zip` | Pilot dataset + ground-truth                             |

---

## Extending RAG-for-Code

1. **Add a new similarity metric**
   Implement it in `FunctionsRetrieval.compute_*` and wire it in `lexical_ranking`.
2. **Support another language**
   Swap the AST parser in `BlocksCutting.py` and adjust the regex rules in `EvaluatePred.py`.
3. **Plug in your favourite model**
   Any vLLM-compatible checkpoint works—just change `--model` in `RunModel.py`.

---

## Citation

If you use this toolkit, please cite:

```bibtex
@misc{ragforcode2024,
  title   = {RAG-for-Code: Function-Level Retrieval-Augmented Generation Pipeline},
  author  = {Zhang, Xuanming and Mao, Chuan},
  year    = {2024},
  howpublished = {\url{https://github.com/XMZhangAI/RAG-for-code}}
}
```

---

## License

Unless noted otherwise, the code is released under the **MIT License**.
The bundled datasets inherit the licenses of their original sources.

---

*Last updated – 8 Jul 2025*

[1]: https://arxiv.org/abs/2504.10046 "CodeRAG: Supportive Code Retrieval on Bigraph for Real-World Code Generation"
[2]: https://raw.githubusercontent.com/XMZhangAI/RAG-for-code/master/BlocksCutting.py "raw.githubusercontent.com"
[3]: https://raw.githubusercontent.com/XMZhangAI/RAG-for-code/master/FunctionsRetrieval.py "raw.githubusercontent.com"
[4]: https://raw.githubusercontent.com/XMZhangAI/RAG-for-code/master/GetInput.py "raw.githubusercontent.com"
[5]: https://raw.githubusercontent.com/XMZhangAI/RAG-for-code/master/RunModel.py "raw.githubusercontent.com"
[6]: https://raw.githubusercontent.com/XMZhangAI/RAG-for-code/master/EvaluatePred.py "raw.githubusercontent.com"
