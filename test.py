import os
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
import torch
from transformers import AutoModel, AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained('Salesforce/SweRankEmbed-Small', trust_remote_code=True)
model = AutoModel.from_pretrained('Salesforce/SweRankEmbed-Small', trust_remote_code=True, add_pooling_layer=False)
model.eval()

query_prefix = 'Represent this query for searching relevant code: '
queries  = ['Calculate the n-th factorial']
queries_with_prefix = ["{}{}".format(query_prefix, i) for i in queries]
query_tokens = tokenizer(queries_with_prefix, padding=True, truncation=True, return_tensors='pt', max_length=512)

documents = ['def fact(n):\n if n < 0:\n  raise ValueError\n return 1 if n == 0 else n * fact(n - 1)',
             '"def fact(n):\n if n <= 0:\n return 0\n elif n == 1:\n return 1\n else:\n a, b = 0, 1\n for _ in range(2, n + 1):\n a, b = b, a + b\n return b\n\n']
document_tokens =  tokenizer(documents, padding=True, truncation=True, return_tensors='pt', max_length=512)

# Compute token embeddings
with torch.no_grad():
    query_embeddings = model(**query_tokens)[0][:, 0]
    document_embeddings = model(**document_tokens)[0][:, 0]
    # query_embeddings = model(**query_tokens).last_hidden_state.sum(dim=1)
    # document_embeddings = model(**document_tokens).last_hidden_state.sum(dim=1)


# normalize embeddings
query_embeddings = torch.nn.functional.normalize(query_embeddings, p=2, dim=1)
document_embeddings = torch.nn.functional.normalize(document_embeddings, p=2, dim=1)

scores = torch.mm(query_embeddings, document_embeddings.transpose(0, 1))
for query, query_scores in zip(queries, scores):
    doc_score_pairs = list(zip(documents, query_scores))
    doc_score_pairs = sorted(doc_score_pairs, key=lambda x: x[1], reverse=True)
    #Output passages & scores
    print("Query:", query)
    for document, score in doc_score_pairs:
        print(score, document)
