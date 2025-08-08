import gradio as gr
import joblib
from sentence_transformers import SentenceTransformer

# Load classifier and embedding model
clf = joblib.load("data/logreg_model.pkl")
embedder = SentenceTransformer("all-MiniLM-L6-v2")  # can reload from Hugging Face

# Prediction function
def detect_origin(text):
    if not text.strip():
        return "Please enter some text."
    embedding = embedder.encode([text], normalize_embeddings=True)
    pred = clf.predict(embedding)[0]
    label = "Human" if pred == 0 else "Machine-Generated"
    return label

# Create interface
interface = gr.Interface(
    fn=detect_origin,
    inputs=gr.Textbox(lines=8, placeholder="Paste your text here..."),
    outputs="text",
    title="LLM Text Detector",
    description="Paste any text and detect if it's Human or Machine-generated."
)

if __name__ == "__main__":
    interface.launch()
