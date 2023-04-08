# NLP_CORD19
Download the data

```sh
# navigate to the data folder
cd data

# download the data file
# which is also available at https://www.semanticscholar.org/cord19/download
wget https://ai2-semanticscholar-cord-19.s3-us-west-2.amazonaws.com/2021-07-26/document_parses.tar.gz

# decompress the file which may take several minutes
tar -xf document_parses.tar.gz

# which creates a folder named document_parses
```
`parse.py` for extracting texts from the original documents.
`tokenizer_nltk.py` for tokenizing.
`word2vec_train.py` for training the word representations.
`visualize_tSNE.py` for visualizing the word representations.
`co-occurence.py` for calculating the co-occurence probability.
