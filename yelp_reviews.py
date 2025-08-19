from datasets import load_dataset
import pandas as pd

ds_yelp = load_dataset("yelp_polarity", split="train[:2000]")
df_yelp = pd.DataFrame({'text': ds_yelp['text'], 'label': [0] * len(ds_yelp)})
df_yelp.to_csv("data/yelp_reviews.csv", index=False)
print("Saved Yelp reviews CSV")
