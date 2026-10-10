# Embeddings of a unit that occurs several times

Produced by `studies/repeated_unit_embeddings.py`. The models are small and have random weights: see the documentation of the script.

Parameters: 200 inputs for each model, of 8 to 15 units of 2 to 29 words, one unit occurring three times; batches of 4 units; at most 48 tokens in a unit.

Software: discoursekit 0.1.0.dev0, sentence-transformers 6.1.0, transformers 5.19.0, PyTorch 2.14.1+cu130, NumPy 2.5.3, Python 3.12.3, Linux x86_64, CPU.

Number of inputs in which the three vectors of the repeated unit are not all equal, when the model encodes the input directly and when the adapter of the library does:

| Model | Encoded directly | Largest difference | Through the adapter |
|---|---:|---:|---:|
| 16 dimensions, 1 layer | 27 of 200 | 2.4e-07 | 0 of 200 |
| 64 dimensions, 2 layers | 25 of 200 | 2.4e-07 | 0 of 200 |
