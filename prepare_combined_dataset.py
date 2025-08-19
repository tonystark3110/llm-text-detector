import pandas as pd

def load_ag_news(path):
    df_train = pd.read_csv(f"{path}\\train.csv")
    df_test = pd.read_csv(f"{path}\\test.csv")

    # Combine Title and Description into one text column
    df_train['text'] = df_train['Title'] + ". " + df_train['Description']
    df_test['text'] = df_test['Title'] + ". " + df_test['Description']

    # Mark all as human (label=0)
    df_train['label'] = 0
    df_test['label'] = 0

    df_train = df_train[['text', 'label']]
    df_test = df_test[['text', 'label']]

    return pd.concat([df_train, df_test], ignore_index=True)

def load_other_dataset(csv_path):
    df = pd.read_csv(csv_path)
    # Assuming CSV already has 'text' and 'label' columns
    return df[['text', 'label']]

def main():
    ag_news_path = r"C:/Users/Manikandan/Desktop/project/llm-text-detector/data/ag"
    yelp_path = r"C:/Users/Manikandan/Desktop/project/llm-text-detector/data/yelp_reviews.csv"
    conv_path = r"C:/Users/Manikandan/Desktop/project/llm-text-detector/data/conversational_data.csv"
    machine_path = r"C:/Users/Manikandan/Desktop/project/llm-text-detector/data/machine_generated.csv"

    print("Loading AG News...")
    df_ag = load_ag_news(ag_news_path)
    print(f"Loaded AG News samples: {len(df_ag)}")

    print("Loading Yelp reviews...")
    df_yelp = load_other_dataset(yelp_path)
    print(f"Loaded Yelp samples: {len(df_yelp)}")

    print("Loading conversational data...")
    df_conv = load_other_dataset(conv_path)
    print(f"Loaded conversational samples: {len(df_conv)}")

    print("Loading machine-generated data...")
    df_machine = load_other_dataset(machine_path)
    print(f"Loaded machine-generated samples: {len(df_machine)}")

    print("Combining all datasets...")
    df_all = pd.concat([df_ag, df_yelp, df_conv, df_machine], ignore_index=True)

    print("Shuffling the combined dataset...")
    df_all = df_all.sample(frac=1, random_state=42).reset_index(drop=True)

    output_path = r"C:\Users\Manikandan\Desktop\project\llm-text-detector\data\combined_dataset.csv"
    df_all.to_csv(output_path, index=False)
    print(f"Saved combined dataset to {output_path} with {len(df_all)} samples.")

if __name__ == "__main__":
    main()
