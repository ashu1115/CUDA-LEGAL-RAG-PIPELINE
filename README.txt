# bits-auto-legal-document-sum

## Pre-requisite

- Python 3.11
- Java - [Download from Oracle](https://www.oracle.com/in/java/technologies/downloads/#jdk21-windows)

## Install Packages

pip install -r requirements.txt

## Create a virtual env

python -m venv env

## Activate the Virtual Environment
  ## On Windows
.\env\Scripts\Activate.ps1

  ## On macOS/Linux
source env/bin/activate

Set-ExecutionPolicy -Scope CurrentUser RemoteSigned -Force

## Usage
Step 1: Run the below command
- python flask_app.py

Step 2: Open the below link in browser.
- http://127.0.0.1:5000

## Model References
- Legal Bert - https://huggingface.co/nlpaueb/legal-bert-base-uncased
- Roberta CUAD - https://huggingface.co/gustavhartz/roberta-base-cuad-finetuned
- GPT Neo - https://huggingface.co/EleutherAI/gpt-neo-1.3B

## Process
- input_data: Make sure this folder is present in working code directory and it should contains all the legal document pdfs for which you want to generate the embeddings along with the CUAD_v1.json file which contains the questions and answers.


- read_preprocess_data.ipynb: This file reads all the contract pdfs from input data folder and generates the file_contents_pdf.pkl in the same working directory which contains the extracted text in the key value pair format. 
  - Input: input_data directory
  - Output: file_contents_pdf.pkl

- generate_topics.ipynb: This file reads all the pdfs text from file_contents_pdf.pkl and generates the csvs, which encloses different topics for each of the contracts using topic modelling using LDA model. 
  - Input: file_contents_pdf.pkl
  - Output: contract_topics/*.csv

- generate_embeddings.ipynb: This file reads all the contracts csvs from contract_topics and generates the embeddings for each topics present in every document using legal-bert-base-uncased and saves the embedding in topic_embeddings folder in csv format
  - Input: contract_topics
  - Output: topic_embeddings

- random_question_generator.ipynb: It takes input from input_data/CUAD_v1.json file and generates multiple output as per requirement. 
  - Input: input_data/CUAD_v1.json
  - Outputs: random_question_dataframe.pkl, random_questions.csv, document_list_available_to_query.csv, full_question_dataframe.pkl, full_questions.csv

- question_answer_functions.py: This function contains to generate embeddings for the use query, generate output throught the models, compute similarity between user embeddings and document embeddings. This function is used in evaluation.ipynb and flask_app.py

evaluation.ipynb: This file will take input from sample questions and answers generated (random_question_dataframe.pkl) and generate rogue scores for the predicted answers using two models gpt-neo-1.3B and roberta-base-cuad-finetuned. 
  - Input: random_question_dataframe.pkl
  - Output: model_predictions_rouge_evaluation.csv

- flask_app.py: This file is used to run the RAG UI application on the flask server which takes document name and user query as input and generate the output based on two models along with different matrix visible on web based ui.  
  - Input: Pick the document name from the file generated through random_question_generator.ipynb , file name - document_list_available_to_query.csv 
  - Output: Visible on UI link - http://127.0.0.1:5000/
