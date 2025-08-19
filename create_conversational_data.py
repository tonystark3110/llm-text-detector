import pandas as pd

# Example conversational texts (each row is one dialogue turn or multiple turns joined)
conversations = [
    "Hey, how are you doing today?",
    "I'm doing great, thanks! What about you?",
    "Pretty good. Just finished reading an interesting book on AI.",
    "That sounds cool! Which one was it?",
    "It's called 'Artificial Intelligence: A Modern Approach'.",
    "Have you tried the new café downtown?",
    "Yes, their coffee is amazing! Highly recommend it.",
    "What’s the weather like on your end?",
    "It's sunny and warm, perfect for a walk in the park."
]

# Create DataFrame with label 0 for human-written
df_conv = pd.DataFrame({'text': conversations, 'label': [0] * len(conversations)})

# Save to CSV file
df_conv.to_csv('data/conversational_data.csv', index=False)
print("Saved conversational data CSV with sample dialogues.")
