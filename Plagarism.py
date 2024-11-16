from transformers import BertTokenizer, BertModel
import torch
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

# Function to load BERT model and tokenizer
def load_bert_model():
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    model = BertModel.from_pretrained('bert-base-uncased')
    return tokenizer, model
# Function to get BERT embeddings for a sentence
def get_bert_embeddings(sentence, tokenizer, model):
    # Tokenize the sentence
    inputs = tokenizer(sentence, return_tensors='pt', padding=True, truncation=True, max_length=512)
    
    # Get the BERT output (embeddings)
    with torch.no_grad():
        outputs = model(**inputs)
    
    # Get the embeddings from the last hidden state (using the [CLS] token embeddings)
    embeddings = outputs.last_hidden_state[:, 0, :].squeeze().numpy()
    return embeddings

# Function to calculate cosine similarity between two sentence embeddings
def calculate_cosine_similarity(embedding1, embedding2):
    return cosine_similarity([embedding1], [embedding2])[0][0]

# Main function to check for paraphrasing
def check_paraphrasing(text1, text2):
    # Load BERT model and tokenizer
    tokenizer, model = load_bert_model()

    # Get the BERT embeddings for both texts
    embedding1 = get_bert_embeddings(text1, tokenizer, model)
    embedding2 = get_bert_embeddings(text2, tokenizer, model)

    # Calculate cosine similarity between the two embeddings
    similarity_score = calculate_cosine_similarity(embedding1, embedding2)
    
    return similarity_score

# Example texts
text1 = """
Artificial intelligence (AI) is intelligence demonstrated by machines, in contrast to the natural intelligence displayed by humans and animals.
Leading AI textbooks define the field as the study of "intelligent agents": any device that perceives its environment and takes actions that maximize its chance of successfully achieving its goals.
"""
text2 = """
AI refers to machines that are able to perform tasks that normally require human intelligence. It includes learning, reasoning, problem-solving, and decision-making.
Many experts in the field describe AI as systems capable of perceiving their environment and taking actions to achieve goals effectively.
"""

# Check for paraphrasing
similarity_score = check_paraphrasing(text1, text2)

# Output the similarity score
print(f"Cosine Similarity Score: {similarity_score:.4f}")

# Set a threshold to consider it as paraphrasing
threshold = 0.8  # You can adjust this threshold
if similarity_score > threshold:
    print("Paraphrasing detected!")
else:
    print("No paraphrasing detected.")