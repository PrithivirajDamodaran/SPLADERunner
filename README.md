# SPLADERunner

Ultra-lite &amp; Super-fast Python wrapper for the unofficial implementation of SPLADEv2 (SParse Lexical And Expanion) models for your search & retrieval pipelines. Implementation is based on SPLADEv2 paper: https://arxiv.org/abs/2109.10086 with some minor tweaks.

1. ⚡ **Ultra-lite & Superfast**: 
    - **No Torch or Transformers** needed. Runs on CPU.
    - Boasts the 
        - **tiniest document expander, ~8MB**.
        - **Faster inference**
    
2. 💡 **Why learn Sparse Representations?**:

- Lexical search with BOW models are strong baselines, but they famously suffer from **vocabulary mismatch** problem,as they can only do exact term matching. BoW representations are calculated and not learned.

- Learned (neural) rankers / Dense retrievers with approximate nearest neighbors search has shown impressive results but they can 
    - Suffer from **token amnesia** (sometimes miss simple term matching), 
    - Can be **resource intensive** (both for storage and retreival), 
    - Are famously **not interpretable** and 
    - Scaling to newer domains needs data/finetuning.

- Best of both worlds made sense and gave rise to interest in **learning sparse representations** for queries and documents with some interpretability. The sparse representations also double as implicit or explicit (latent, contextualized) expansion mechanisms for both query and documents.


3. **What the Models learn?**:
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
sparse_rep = expander.expand("Chronic Obstructive Pulmonary Disease (COPD) presents a complex interplay of respiratory and cardiovascular comorbidities, necessitating multidisciplinary management.")
```


4. 💸 **Where and How can you use?**:
- [TBD]