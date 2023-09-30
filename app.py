import streamlit as st
from pathlib import Path
from streamlit_chat import message
from langchain.document_loaders import CSVLoader
from langchain.indexes import VectorstoreIndexCreator
from langchain.chains import RetrievalQA
from langchain.llms import OpenAI, AzureOpenAI
import os

import os
from dotenv import load_dotenv

load_dotenv()  # take environment variables from .env.

# Code of your application, which uses environment variables (e.g. from `os.environ` or
# `os.getenv`) as if they came from the actual environment.

print(os.environ['OPENAI_API_TYPE'])
print(os.environ['OPENAI_API_VERSION'])
print(os.environ['OPENAI_API_BASE'])
print(os.environ['OPENAI_API_KEY'])
print(os.environ['DEPLOYMENT_NAME'])
print(os.environ['MODEL_NAME'])

st.title('CSV Question and answer ChatBot')
st.info("If no file is uploaded, this app will use the default data: Football Player Stats.csv!")
csv_file_uploaded = st.file_uploader(label="Upload your CSV File here")

data_folder = './data'

if csv_file_uploaded is not None:
    def save_file_to_folder(uploadedFile):
        # Save uploaded file to 'content' folder.
        save_folder = data_folder
        save_path = Path(save_folder, uploadedFile.name)
        with open(save_path, mode='wb') as w:
            w.write(uploadedFile.getvalue())

        if save_path.exists():
            st.success(f'File {uploadedFile.name} is successfully saved!')
            
    save_file_to_folder(csv_file_uploaded)
    loader = CSVLoader(file_path=os.path.join(data_folder, csv_file_uploaded.name))
else:
    # Load the documents
    loader = CSVLoader(file_path='./data/2021-2022 Football Player Stats.csv',encoding = 'ISO-8859-1')

# Create an index using the loaded documents
index_creator = VectorstoreIndexCreator()
docsearch = index_creator.from_loaders([loader])

# print('db has been built')

# Create a question-answering chain using the index
llm = AzureOpenAI(deployment_name=os.environ['DEPLOYMENT_NAME'],
                  model_name=os.environ['MODEL_NAME'],
                  temperature=0,
                  openai_api_base=os.environ['OPENAI_API_BASE'],
                  openai_api_key=os.environ['OPENAI_API_KEY'])

chain = RetrievalQA.from_chain_type(llm, chain_type="stuff", \
    retriever=docsearch.vectorstore.as_retriever(), input_key="question")

#Creating the chatbot interface
st.title("Chat wtih your CSV Data")

# Storing the chat
if 'generated' not in st.session_state:
    st.session_state['generated'] = []

if 'past' not in st.session_state:
    st.session_state['past'] = []

def generate_response(user_query):
    response = chain({"question": user_query})
    return response['result']

# We will get the user's input by calling the get_text function
def get_text():
    input_text = st.text_input("You: ","Ask Question From your Document?", key="input")
    return input_text
user_input = get_text()

if user_input:
    output = generate_response(user_input)
    # store the output 
    st.session_state.past.append(user_input)
    st.session_state.generated.append(output)

if st.session_state['generated']:
    for i in range(len(st.session_state['generated'])-1, -1, -1):
        message(st.session_state["generated"][i], key=str(i))
        message(st.session_state['past'][i], is_user=True, key=str(i) + '_user')

# Pass a query to the chain
# query = "Do you have a column called age?"
# response = chain({"question": query})
# print(response['result'])
# print("Hello CodeSandbox!")
