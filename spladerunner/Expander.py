import json
from pathlib import Path
from tokenizers import AddedToken, Tokenizer
import onnxruntime as ort
import numpy as np
import os
import zipfile
import requests
from tqdm import tqdm
from spladerunner.Config import DEFAULT_MODEL, DEFAULT_CACHE_DIR, MODEL_URL, MODEL_FILE_MAP
import collections


class Expander:

    def __init__(self, 
                 model_name = DEFAULT_MODEL, 
                 max_length=512,
                 cache_dir= DEFAULT_CACHE_DIR
                 ):

        self.cache_dir = Path(cache_dir)
        
        if not self.cache_dir.exists():
            print(f"Cache directory {self.cache_dir} not found. Creating it..")
            self.cache_dir.mkdir(parents=True, exist_ok=True)

        self.model_dir = self.cache_dir / model_name
        
        if not self.model_dir.exists():
            print(f"Downloading {model_name}...")
            self._download_model_files(model_name)
            
        model_file = MODEL_FILE_MAP[model_name]
        
        self.session = ort.InferenceSession(self.cache_dir / model_name / model_file)
        self.tokenizer = self._get_tokenizer(max_length)
        self.reverse_voc = {v: k for k, v in self.tokenizer.get_vocab().items()}

    def _download_model_files(self, model_name):
        
        # The local file path to which the file should be downloaded
        local_zip_file = self.cache_dir / f"{model_name}.zip"

        formatted_model_url = MODEL_URL.format(model_name)

        with requests.get(formatted_model_url, stream=True) as r:
            r.raise_for_status()
            total_size = int(r.headers.get('content-length', 0))
            with open(local_zip_file, 'wb') as f, tqdm(
                    desc=local_zip_file.name,
                    total=total_size,
                    unit='iB',
                    unit_scale=True,
                    unit_divisor=1024,
                ) as bar:
                for chunk in r.iter_content(chunk_size=8192):
                    size = f.write(chunk)
                    bar.update(size)

        # Extract the zip file
        with zipfile.ZipFile(local_zip_file, 'r') as zip_ref:
            zip_ref.extractall(self.cache_dir)

        # Optionally, remove the zip file after extraction
        os.remove(local_zip_file)

    def _load_vocab(self, vocab_file):
    
        vocab = collections.OrderedDict()
        with open(vocab_file, "r", encoding="utf-8") as reader:
            tokens = reader.readlines()
        for index, token in enumerate(tokens):
            token = token.rstrip("\n")
            vocab[token] = index
        return vocab
        

    def _get_tokenizer(self, max_length):
      
        config_path = self.model_dir / "config.json"
        tokenizer_path = self.model_dir / "tokenizer.json"
        tokenizer_config_path = self.model_dir / "tokenizer_config.json"
        tokens_map_path = self.model_dir / "special_tokens_map.json"

        # Check for file existence
        for path in [config_path, tokenizer_path, tokenizer_config_path, tokens_map_path]:
            if not path.exists():
                raise FileNotFoundError(f"{path.name} missing in {self.model_dir}")

        config = json.load(open(str(config_path)))
        tokenizer_config = json.load(open(str(tokenizer_config_path)))
        tokens_map = json.load(open(str(tokens_map_path)))

        tokenizer = Tokenizer.from_file(str(tokenizer_path))
        tokenizer.enable_truncation(max_length=min(tokenizer_config["model_max_length"], max_length))
        tokenizer.enable_padding(pad_id=config["pad_token_id"], pad_token=tokenizer_config["pad_token"])

        for token in tokens_map.values():
            if isinstance(token, str):
                tokenizer.add_special_tokens([token])
            elif isinstance(token, dict):
                tokenizer.add_special_tokens([AddedToken(**token)])

        # vocab_file = self.model_dir / "vocab.txt"
        # if vocab_file.exists():
        #     tokenizer.vocab = self._load_vocab(vocab_file)
        #     tokenizer.ids_to_tokens = collections.OrderedDict([(ids, tok) for tok, ids in tokenizer.vocab.items()])                

        return tokenizer
    
    
    def expand(self, request):

        if isinstance(request, str):
            plain_input = [request]
        else:
            plain_input = request

        encoded_input = self.tokenizer.encode_batch(plain_input)
        input_ids = np.array([e.ids for e in encoded_input])
        token_type_ids = np.array([e.type_ids for e in encoded_input])
        attention_mask = np.array([e.attention_mask for e in encoded_input])

        onnx_input = {
            "input_ids": np.array(input_ids, dtype=np.int64),
            "input_mask": np.array(attention_mask, dtype=np.int64),
            "segment_ids": np.array(token_type_ids, dtype=np.int64),
        }


        outputs = self.session.run(None, onnx_input)
        outputs = outputs[0]

        batch_size = outputs.shape[0]
        sparse_representations = []

        # Iterate over each example in the batch
        for i in range(batch_size):
            single_output = outputs[i]
            single_attention_mask = attention_mask[i]

            # Apply ReLU log
            relu_log = np.log1p(np.maximum(single_output, 0))

            # Apply attention mask
            weighted_log = relu_log * single_attention_mask[:, np.newaxis]

            # Find max values
            max_val = np.max(weighted_log, axis=0)
            
            # Find non-zero columns
            cols = np.nonzero(max_val)[0]
            weights = max_val[cols]

            # Create dictionary and sort it
            d = dict(zip(cols, weights))
            sorted_d = dict(sorted(d.items(), key=lambda item: item[1], reverse=True))

            # Construct SPLADE BoW representation for the current sentence
            sparse_representation = {self.reverse_voc[k]: round(v, 2) for k, v in sorted_d.items()}
            sparse_representations.append(sparse_representation)

        return sparse_representations
