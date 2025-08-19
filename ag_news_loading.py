import pandas as pd

ag_news_path = r"C:/Users/Manikandan/Desktop/project/llm-text-detector/data/ag"

# Load train and test CSVs
df_ag_train = pd.read_csv(f"{ag_news_path}\\train.csv")
df_ag_test = pd.read_csv(f"{ag_news_path}\\test.csv")

# Combine 'Title' and 'Description' as the text input
df_ag_train['text'] = df_ag_train['Title'] + ". " + df_ag_train['Description']
df_ag_test['text'] = df_ag_test['Title'] + ". " + df_ag_test['Description']

# Use Class Index as label or set to 0 for human since this is human-written data
# Assuming your task treats all these as human-written => label = 0
df_ag_train['label'] = 0
df_ag_test['label'] = 0

# Select only 'text' and 'label' columns
df_ag_train = df_ag_train[['text', 'label']]
df_ag_test = df_ag_test[['text', 'label']]

# Optionally combine train and test splits
df_ag_news = pd.concat([df_ag_train, df_ag_test], ignore_index=True)
print(f"AG News combined samples: {len(df_ag_news)}")

# Now you can continue combining with other human data and machine-generated data for training
