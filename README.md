# SPLADERunner

## 1. What is it?

*Title is dedicated to the Original Blade Runners - Harrison Ford and the Author  Philip K. Dick of "Do Androids Dream of Electric Sheep?"*

A Ultra-lite &amp; Super-fast Python wrapper for the [independent implementation of SPLADE++ models](https://huggingface.co/prithivida/Splade_PP_en_v1) for your search & retrieval pipelines. Based on the papers Naver's "From Distillation to Hard Negative Sampling: Making Sparse Neural IR Models More Effective":https://arxiv.org/pdf/2205.04733.pdf and "Gooogle's SparseEmbed": https://arxiv.org/abs/2109.10086.

- ⚡ **Ultra-lite & Superfast**: 
    - **No Torch or Transformers** needed. Runs on CPU.
    - **Inference / Serving Efficient**: Optimised inference performance.
    - **FLOPS & Retrieval Efficient**: Refer model card for details.

   
## 🚀 Installation:
```python 
pip install spladerunner
```

## Usage:
```python

# One-time only init
from spladerunner import Expander
expander = Expander('Splade_PP_en_v1', 128) #pass model, max_seq_len

# Sample passage expansion
sparse_rep = expander.expand("The Manhattan Project and its atomic bomb helped bring an end to World War II. Its legacy of peaceful uses of atomic energy continues to have an impact on history and science.")
print(sparse_rep)

```

(Feel free to skip to 3 If you are expert in sparse and dense representations)

## Why Sparse Representations? 


    
- **Lexical search** with BOW based sparse vectors are strong baselines, but they famously suffer from **vocabulary mismatch** problem, as they can only do exact term matching. 

<details>
    
Pros

    ✅ Efficient and Cheap.
    ✅ No need to fine-tune models.
    ✅️ Interpretable.
    ✅️ Exact Term Matches.

Cons

    ❌ Vocabulary mismatch (Need to remember exact terms)

</details>


- **Semantic Search** Learned Neural /  Dense retrievers with approximate nearest neighbors search has shown impressive results but they can
  
<details>
    
Pros

    ✅ Search how humans innately think.
    ✅ When finetuned beats sparse by long way.
    ✅ Easily works with Multiple modals.

Cons

    ❌ Suffers token amnesia (misses term matching), 
    ❌ Resource intensive (both index & retreival), 
    ❌ Famously hard to interpret.
    ❌ Needs fine-tuning for OOD data.

</details>

- Getting pros of both searches made sense and that gave rise to interest in **learning sparse representations** for queries and documents with some interpretability. The sparse representations also double as implicit or explicit (latent, contextualized) expansion mechanisms for both query and documents. If you are new to [query expansion learn more here from Daniel Tunkelang.](https://queryunderstanding.com/query-expansion-2d68d47cf9c8)



2b. **What the Models learn?**
- The model learns to project it's learned dense representations over a MLM head to give a vocabulary distribution.
  <center><img src="./images/vocproj.png" width=300/></center>

## 3. 💸 **Why SPLADERunner?**:
- $ Concious: Serverless deployments like Lambda are charged by memory & time per invocation
- Smaller package size = shorter cold start times, quicker re-deployments for Serverless.
    
## 4. 🎯 **Models**:
- Below are the list of models supported as of now.
    * [`prithivida/Splade_PP_en_v1`](https://huggingface.co/prithivida/Splade_PP_en_v1) (default model)

4. 💸 **Where and How can you use?**
- [TBD]

5. **How (and what) to contribute?**
- [TBD]

6. **Criticisms and Competitions to SPLADE and Learned Sparse representations:**

- [Wacky Weights in Learned Sparse Representations and the Revenge of Score-at-a-Time Query Evaluation](https://arxiv.org/pdf/2110.11540.pdf)
- [Query2doc: Query Expansion with Large Language Models](https://arxiv.org/pdf/2303.07678.pdf) 
*note: don't mistake this for docT5query, this is a recent work*


- *Thanks to [Nils Reimers](https://www.linkedin.com/in/reimersnils/) for*
    - All the MS-MARCO hard negatives.
    - The trolls :-) and timely inputs around evaluation.
- *Props to Naver folks, the original authors of the paper for such a robust research.*

