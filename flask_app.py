from flask import Flask, request, render_template
from question_answer_functions import *
import time
import os
import pickle
from rouge_score import rouge_scorer

app = Flask(__name__)

# Open and load the pickle files
pickle_file_path = 'full_question_dataframe.pkl'
with open(pickle_file_path, 'rb') as file:
    random_question_dataframe = pickle.load(file)

# Mock function to load and filter the dataset based on document name
def filter_data_by_document(document_name):
    # Replace this with your actual dataset filtering logic
    output_file = os.path.join('topic_embeddings', f'{document_name}.csv')
    df = pd.read_csv(output_file)  # Assuming the dataset is in CSV format
    filtered_df = df
    return filtered_df

# Function to get model output and time
def get_model_output_and_time(model_function, question, chunk):
    start_time = time.time()
    predicted_answer = model_function(question, chunk)
    end_time = time.time()
    
    computation_time = end_time - start_time  # Time taken in seconds
    return predicted_answer, computation_time

@app.route('/', methods=['GET', 'POST'])
def indexs():
    if request.method == 'POST':
        query = request.form['query']
        document = request.form['document']
        print(document)
        # Filter the dataset based on the document name
        filtered_data = filter_data_by_document(document)

        if filtered_data.empty:
            return render_template('index.html', query=query, document=document, 
                                   answer1="Invalid Document", answer2="Invalid Document", 
                                   time1=0, time2=0, score1=0, score2=0)
        
        
        # Retrieve the best chunk from the document store
        # best_chunk = retrieve_best_chunk(query)
        best_chunk,word_count=user_query(query,document)
        # print(best_score)
       
        if word_count :
            # Get model outputs and computation times
            answer1, time1 = get_model_output_and_time(models_output1, query, best_chunk)
            best_chunk_model_2 = format_input_text(query, best_chunk)
            best_chunk_model_2 = format_text(best_chunk_model_2)
            best_chunk_model_2 = format_contract_analysis(best_chunk_model_2)
            answer2, time2 = get_model_output_and_time(models_output2, query, best_chunk_model_2)
            document_without_extension='.'.join(document.split('.')[:-1])
            print(document_without_extension)
            document_df=random_question_dataframe[random_question_dataframe['document_title'].isin([document_without_extension])]
            # print(document_df['question_text'])
            # print(query)
            
            flag = 0  # Initialize the flag to 0

            # Loop through the Series
            for question in document_df['question_text']:
                if query == question:
                    flag = 1
                    break  # Break out of the loop as soon as we find a match

            print(f"Flag: {flag}")

            if flag==1:
                print('executing')
                best_score1 = 0
                best_answer1 = None
                best_score2 = 0
                best_answer2 = None
                # Initialize ROUGE scorer for ROUGE-1
                scorer = rouge_scorer.RougeScorer(['rouge1'], use_stemmer=True)

                actual_answer_series = document_df.loc[document_df['question_text'] == query, 'answer_text']
                for actual_answer in actual_answer_series:
                    
                    rouge_score1=scorer.score(actual_answer, answer1)['rouge1'].fmeasure
                    rouge_score2=scorer.score(actual_answer, answer2)['rouge1'].fmeasure
                     # Check if the ROUGE score for answer1 is better
                    if rouge_score1 > best_score1:
                        best_score1 = rouge_score1
                        best_answer1 = actual_answer 
                    if rouge_score2 > best_score2:
                        best_score2 = rouge_score2
                        best_answer2 = actual_answer
            else:
                best_score1='ans not available'
                best_score2='ans not available'
                best_answer2='Actual ans not available'
                best_answer1='Actual ans not available'

        else: 
            model_outputs={'none':'invalid query'}
            best_chunk='none'

         # Render the results on the index page
        return render_template('index.html', query=query, document=document, 
                               answer1=answer1, time1=time1, 
                               answer2=answer2, time2=time2, 
                               score1=best_score1, score2=best_score2,best_chunk=best_chunk,best_answer2=best_answer2,best_answer1=best_answer1)
    
    return render_template('index.html')

if __name__ == '__main__':
    app.run(debug=False)
