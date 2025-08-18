from datasets import load_dataset

# Loads 5,000 Wikipedia articles (adjust number as needed)
wiki = load_dataset("wikimedia/wikipedia", "20231101.en", split="train[:5000]")
wiki_texts = [article["text"] for article in wiki]
