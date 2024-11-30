
import pickle
import numpy as np
from sklearn.decomposition import LatentDirichletAllocation as LDA
import spacy
from sklearn.feature_extraction.text import CountVectorizer
import pandas as pd
from scipy.spatial.distance import cosine
import torch
from transformers import AutoTokenizer, AutoModel
from transformers import pipeline
import os
import re

# df = pd.read_csv('contract_chunks_embeddings_NER_LDA.csv')
#df = pd.read_csv('all_contract_embeddings_lda.csv')
# pickle_file_path = 'data.pkl'
#pickle_file_path = 'all_contract_embeddings_lda.pkl'
    # Step to open and load the pickle files
# with open(pickle_file_path, 'rb') as file:
#     data = pickle.load(file)
pipe1 = pipeline("question-answering", model="gustavhartz/roberta-base-cuad-finetuned", device=-1)
pipe2 = pipeline("text-generation", model="EleutherAI/gpt-neo-1.3B")
# pipe2 = pipeline("question-answering", model="gustavhartz/roberta-base-cuad-finetuned", device=-1)


def preprocess_context(context):
    # Replace newlines with spaces and remove excessive whitespace
    cleaned_context = ' '.join(context.split())
    return cleaned_context.strip()

def convert_tensor_string(tensor_str):
    # Step 1: Remove 'tensor([[' and trailing ']])'
    clean_str = tensor_str.replace('tensor([', '').replace('])', '')
    
    # Step 2: Remove newline characters and excessive spaces
    clean_str = clean_str.replace('\n', '').replace('  ', ' ').strip()

    # Step 3: Ensure the string is properly formatted, split by commas
    clean_str = clean_str.replace(' ', ',')  # Replace spaces with commas if necessary
    
    # Print the cleaned string for debugging
    # print(f"Cleaned string: {clean_str}")

    # Step 4: Split the cleaned string into individual values
    value_str_list = clean_str.split(',')
    value_str_list[0] = value_str_list[0][1:]
    value_str_list[-1] = value_str_list[-1][:-1]
    # print(value_str_list)
    
    # Step 5: Try converting the string values to float
    try:
        value_list = [float(value) for value in value_str_list if value]  # Avoid empty strings
    except ValueError as e:
        print(f"Error converting string to floats: {e}")
        return None
    
    # Step 6: Convert the list to a numpy array
    tensor_array = np.array(value_list)
    
    return tensor_array

# Function to compute best similarity and construct text of approximately 500 words
def best_similarity_context(single_embedding, embeddings_array, df):
    similarities = []
    
    # Compute cosine similarities for each embedding
    for embedding in embeddings_array:
        a = convert_tensor_string(embedding)  # Convert embedding to an appropriate format
        similarity = 1 - cosine(single_embedding.flatten(), a.flatten())
        similarities.append(similarity)

    # Get the indices of the sorted similarities in descending order
    sorted_indices = np.argsort(similarities)[::-1]

    # Start with the best match
    combined_text = df["Chunk_text"][sorted_indices[0]]
    
    # Check the length of the combined text
    word_count = len(combined_text.split())
    
    # Append text from second, third, etc. until reaching approximately 500 words
    for index in sorted_indices[1:]:
        if word_count >= 500:
            break
        combined_text += " " + df["Chunk_text"][index]
        word_count = len(combined_text.split())

    return word_count, combined_text

# Function to generate embeddings
def generate_embeddings(text):
    # Load the LegalBERT tokenizer and model
    model_name = "nlpaueb/legal-bert-base-uncased"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name)
    # Tokenize the input text
    inputs = tokenizer(text, return_tensors='pt', truncation=True, padding=True, max_length=512)

    # Forward pass to get the output from LegalBERT
    with torch.no_grad():
        outputs = model(**inputs)

    # Get the embeddings from the last hidden state
    # Use the mean pooling of the last hidden states for the [CLS] token
    embeddings = outputs.last_hidden_state.mean(dim=1)  # You can also use [CLS] token instead

    return embeddings

# Function to compute best similarity
def best_similarity(single_embedding, embeddings_array,df):
    similarities = []
    
    for embedding in embeddings_array:
        # print(embedding['Chunk_embeddings'])
        # print(single_embedding)
        # Compute cosine similarity
        a = convert_tensor_string(embedding)
        similarity = 1 - cosine(single_embedding.flatten(), a.flatten())
        similarities.append(similarity)

    # Find the highest similarity score
    best_score = max(similarities)
    best_index = np.argmax(similarities)
    
    return best_score, df["Chunk_text"][best_index]

def user_query(query,document):
    # Load the LegalBERT tokenizer and model
    # File path of the saved pickle file
    output_file = os.path.join('topic_embeddings', f'{document}.csv')
    df = pd.read_csv(output_file)
    sample_question_1=query
    question_embedding = generate_embeddings(sample_question_1)
    #chunk_embeddings_numpy_array = np.array(df["Chunk_embeddings"].tolist())
    word_count, best_text = best_similarity_context(question_embedding, df["Chunk_embeddings"],df)
    best_text = preprocess_context(best_text)
#    print(best_text)
    return best_text,word_count

# best_text=user_query("What rights does ABW have to co-brand PC QUOTE SOFTWARE")
# print(best_text)
def models_output1(query,best_chunk):
    #ans = pipe(question = query, context = best_chunk, top_k=1)['answer']
    ans1 = pipe1(question = query, context = best_chunk, top_k=1)['answer']
    return ans1.strip()

def format_input_text(question, context):
    # Clean and format the context
    cleaned_context = ' '.join(context.split()).replace("\\n", "\n")
    
    # Construct the final input text
    input_text = f"""
    Question: {question}

    Context:
    {cleaned_context}
    """
    return input_text

def format_contract_analysis(input_string):
    # Split the input string into Question and Context
    question, context = input_string.split("\nContext:\n")
    
    # Create the formatted string
    formatted_text = f"""
You are an expert in legal contract analysis. Based on the following context, provide a clear and concise answer to the question.

Context:
{context.strip()}

Question:
{question.replace('Question: ', '').strip()}
"""
    
    # Remove any duplicated "Question:" or trailing extra text
    formatted_text = formatted_text.strip().rsplit('\nQuestion:', 1)[0] + "\nQuestion:\n" + question.replace('Question: ', '').strip()
    
    return formatted_text

def format_text(text):
    # Strip leading/trailing whitespaces from the entire text
    text = text.strip()

    # Replace multiple spaces/newlines around "Question" and "Context" with a single newline
    text = text.replace("\n    Question:", "Question:").replace("\n    Context:", "Context:")

    # Return the formatted string
    return text


def models_output2(query,best_chunk):
    #ans = pipe(question = query, context = best_chunk, top_k=1)['answer']
    response = pipe2(best_chunk, max_new_tokens=100, do_sample=True, temperature=0.3,truncation=True)[0]['generated_text']
    ans2 = response[len(best_chunk):]
    return ans2.strip()
