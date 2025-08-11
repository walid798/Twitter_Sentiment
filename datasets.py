from transformers import AutoTokenizer
from src.data.datasets import TweetDataset
from src.data.preprocess import load_vocab

# Example: LSTM mode
vocab = load_vocab("experiments/runs/lstm/vocab.json")  # after you build vocab
ds_lstm = TweetDataset("data/Tweets.csv", vocab=vocab, max_len=50)
print(ds_lstm[0])

# Example: BERT mode
tok = AutoTokenizer.from_pretrained("bert-base-uncased")
ds_bert = TweetDataset("data/Tweets.csv", tokenizer=tok, max_len=50)
print(ds_bert[0])
