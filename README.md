# SPLADERunner

*Title is Dedicated to the Original Blade Runners - Harrison Ford and the Author  Philip K. Dick of "Do Androids Dream of Electric Sheep?"*

*Thanks to [Nils Reimers](https://www.linkedin.com/in/reimersnils/) for the trolls :-) and for all the timely inputs especially around evaluation*

## What is it?

A Ultra-lite &amp; Super-fast Python wrapper for the [independent implementation of SPLADEv2 (SParse Lexical And Expanion) models](https://huggingface.co/prithivida/Tiny-SPLADE-Doc) for your search & retrieval pipelines. Implementation is based on SPLADEv2 paper: https://arxiv.org/abs/2109.10086 with some minor tweaks.

1. ⚡ **Ultra-lite & Superfast**: 
    - **No Torch or Transformers** needed. Runs on CPU.
    - Boasts the 
        - **tiniest document expander, ~8MB**.
        - **Faster inference**
    
2. 💡 **Why learn Sparse Representations?**

- **Lexical search** based on BOW models are strong baselines, but they famously suffer from **vocabulary mismatch** problem, as they can only do exact term matching. *(BoW representations are calculated and not learned.)*

- ***Semantic Search** Learned Neural /  Dense retrievers with approximate nearest neighbors search has shown impressive results but they can 
    - Suffer from **token amnesia** (sometimes misses simple term matching), 
    - Can be **resource intensive** (both for storage and retreival), 
    - Are famously **not interpretable** and 
    - Scaling to newer domains needs data/finetuning.

- Best of both worlds made sense, hence gave rise to interest in **learning sparse representations** for queries and documents with some interpretability. The sparse representations also double as implicit or explicit (latent, contextualized) expansion mechanisms for both query and documents. If you are new to [query expansion learn more here from Daniel Tunkelang.](https://queryunderstanding.com/query-expansion-2d68d47cf9c8)


3. **What the Models learn?**
- The model learns to project it's learned dense representations over a MLM head to give a vocabulary distribution.
- <center><img src="./images/vocproj.png" width=300/></center>

4. 💸 **Why SPLADERunner?**:
    - **$ Concious:** Serverless deployments like Lambda are charged by memory & time per invocation*
    - **Smaller package size** = shorter cold start times, quicker re-deployments for Serverless.
    - **Permissive License**. (You can use it commercially with Source attribution).

5. 🎯 **Models**:
    - Below are the list of models supported as of now.
        * [prithivida/Tiny-SPLADE-Doc](https://huggingface.co/prithivida/Tiny-SPLADE-Doc) (default model)
        * [prithivida/Tiny-SPLADE-Query](https://huggingface.co/prithivida/Tiny-SPLADE-Query) (Coming soon)

## 🚀 Installation:
```python 
pip install spladerunner
```

## Usage:
```python
#One-time init
from spladerunner import Expander
# Default model is the document expander.
exapander = Expander()

#Sample Document expansion
sparse_rep = expander.expand("The Manhattan Project and its atomic bomb helped bring an end to World War II. Its legacy of peaceful uses of atomic energy continues to have an impact on history and science.")
```

4. 💸 **Where and How can you use?**
- [TBD]

5. **How (and what) to contribute?**
- [TBD]

6. **Criticisms and Competitions to SPLADE and Learned Sparse representations:**

- [Wacky Weights in Learned Sparse Representations and the Revenge of Score-at-a-Time Query Evaluation](https://arxiv.org/pdf/2110.11540.pdf)

- [Query2doc: Query Expansion with Large Language Models] https://arxiv.org/pdf/2303.07678.pdf (don't mistake this for docT5query, this is a recent work)