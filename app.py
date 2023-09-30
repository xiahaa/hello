import streamlit as st
from pathlib import Path
from streamlit_chat import message
from langchain.document_loaders import CSVLoader
from langchain.indexes import VectorstoreIndexCreator
from langchain.chains import RetrievalQA
from langchain.llms import OpenAI, AzureOpenAI
from langchain.chat_models import ChatOpenAI, AzureChatOpenAI
from langchain.agents import create_pandas_dataframe_agent, create_csv_agent
import os
import pandas as pd
import json
import re
import matplotlib.pyplot as plt

from langchain.agents import (
    AgentType, #ZERO_SHOT_REACT_DESCRIPTION default value
)

from dotenv import load_dotenv
import pandas as pd

load_dotenv()  # take environment variables from .env.

# Code of your application, which uses environment variables (e.g. from `os.environ` or
# `os.getenv`) as if they came from the actual environment.

print(os.environ['OPENAI_API_TYPE'])
print(os.environ['OPENAI_API_VERSION'])
print(os.environ['OPENAI_API_BASE'])
print(os.environ['OPENAI_API_KEY'])
print(os.environ['DEPLOYMENT_NAME'])
print(os.environ['MODEL_NAME'])

def query_agent(agent, query):
    """
    Query an agent and return the response as a string.

    Args:
        agent: The agent to query.
        query: The query to ask the agent.

    Returns:
        The response from the agent as a string.
    """

    prompt = (
        """
            For the following query, if it requires drawing, plotting, visualization some data, 
            generate python code and format code as follows
            ```python
            code, ...
            ``` 

            If it is just asking a question that requires neither, reply as follows:
            {"answer": "answer"}
            Example:
            {"answer": "The title with the highest rating is 'Gilead'"}

            If you do not know the answer, reply as follows:
            {"answer": "I do not know."}

            Return all output as a string.

            Lets think step by step.

            Below is the query.
            Query: 
            """
        + query
    )

    # Run the prompt through the agent.
    response = agent.run(prompt)

    # Convert the response to a string.
    return response.__str__()

def csv_agent_func2(file_path, query):
    """Run the CSV agent with the given file path and user message."""
    model = AzureChatOpenAI(
        temperature=0,
        openai_api_base=os.environ['OPENAI_API_BASE'],
        openai_api_version=os.environ['OPENAI_API_VERSION'],
        deployment_name=os.environ['DEPLOYMENT_NAME'],
        openai_api_key=os.environ['OPENAI_API_KEY'],
        openai_api_type=os.environ['OPENAI_API_TYPE'],
    )

    agent = create_csv_agent(
        model,
        file_path, 
        verbose=True,
        agent_type=AgentType.ZERO_SHOT_REACT_DESCRIPTION
    )

    try:
        # response = agent.run(tool_input)
        response = query_agent(agent, query)
        return response
    except Exception as e:
        st.write(f"Error: {e}")
        return None

def csv_agent_func(file_path, user_message):
    """Run the CSV agent with the given file path and user message."""
    # model = AzureChatOpenAI(
    #     temperature=0,
    #     openai_api_base=os.environ['OPENAI_API_BASE'],
    #     openai_api_version=os.environ['OPENAI_API_VERSION'],
    #     deployment_name=os.environ['DEPLOYMENT_NAME'],
    #     openai_api_key=os.environ['OPENAI_API_KEY'],
    #     openai_api_type=os.environ['OPENAI_API_TYPE'],
    # )
    model = AzureChatOpenAI(
        temperature=0,
        openai_api_base=os.environ['OPENAI_API_BASE'],
        openai_api_version=os.environ['OPENAI_API_VERSION'],
        deployment_name=os.environ['DEPLOYMENT_NAME'],
        openai_api_key=os.environ['OPENAI_API_KEY'],
        openai_api_type=os.environ['OPENAI_API_TYPE'],
    )

    agent = create_csv_agent(
        model,
        file_path, 
        verbose=True,
        agent_type=AgentType.ZERO_SHOT_REACT_DESCRIPTION
    )

    try:
        tool_input = {
            "input": {
                "name": "```python```",
                "arguments": user_message
            }
        }
        response = agent.run(tool_input)
        return response
    except Exception as e:
        st.write(f"Error: {e}")
        return None

def display_content_from_json(json_response):
    """
    Display content to Streamlit based on the structure of the provided JSON.
    """
    
    # Check if the response has plain text.
    if "answer" in json_response:
        st.write(json_response["answer"])

    # Check if the response has a bar chart.
    if "bar" in json_response:
        data = json_response["bar"]
        df = pd.DataFrame(data)
        df.set_index("columns", inplace=True)
        st.bar_chart(df)

    # Check if the response has a table.
    if "table" in json_response:
        data = json_response["table"]
        df = pd.DataFrame(data["data"], columns=data["columns"])
        st.table(df)

def extract_code_from_response(response):
    """Extracts Python code from a string response."""
    # Use a regex pattern to match content between triple backticks
    code_pattern = r"```python(.*?)```"
    match = re.search(code_pattern, response, re.DOTALL)
    
    if match:
        # Extract the matched code and strip any leading/trailing whitespaces
        return match.group(1).strip()
    return None

def csv_analyzer_app():
    """Main Streamlit application for CSV analysis."""
    st.title('CSV Assistant')
    st.write('Please upload your CSV file and enter your query below:')
    
    uploaded_file = st.file_uploader("Choose a CSV file", type="csv")
    
    if uploaded_file is not None:
        file_details = {"FileName": uploaded_file.name, "FileType": uploaded_file.type, "FileSize": uploaded_file.size}
        st.write(file_details)
        
        # Save the uploaded file to disk
        file_path = os.path.join("/tmp", uploaded_file.name)
        with open(file_path, "wb") as f:
            f.write(uploaded_file.getbuffer())
        
        df = pd.read_csv(file_path)
        st.dataframe(df[:10])
        
        user_input = st.text_input("Your query")
        if st.button('Run'):
            response = csv_agent_func2(file_path, user_input)

            print("--------********")
            print(response)
            print("********--------")

            # Extracting code from the response
            code_to_execute = extract_code_from_response(response)
            
            if code_to_execute:
                try:
                    # Making df available for execution in the context
                    exec(code_to_execute, globals(), {"df": df, "plt": plt})
                    fig = plt.gcf()  # Get current figure
                    st.pyplot(fig)  # Display using Streamlit
                except Exception as e:
                    st.write(f"Error executing code: {e}")
            else:
                st.write(response)

    st.divider()



# st.title('CSV Question and answer ChatBot')
# st.info("If no file is uploaded, this app will use the default data: titanic.csv!")
# csv_file_uploaded = st.file_uploader(label="Upload your CSV File here")

# question = st.text_input("Enter your query:")

# data_folder = './data'

# if csv_file_uploaded is not None:
#     def save_file_to_folder(uploadedFile):
#         # Save uploaded file to 'content' folder.
#         save_folder = data_folder
#         save_path = Path(save_folder, uploadedFile.name)
#         with open(save_path, mode='wb') as w:
#             w.write(uploadedFile.getvalue())

#         if save_path.exists():
#             st.success(f'File {uploadedFile.name} is successfully saved!')
            
#     save_file_to_folder(csv_file_uploaded)
#     # loader = CSVLoader(file_path=os.path.join(data_folder, csv_file_uploaded.name))
#     document = pd.read_csv(os.path.join(data_folder, csv_file_uploaded.name))#,encoding='ISO-8859-1')
# else:
#     # Load the documents
#     # loader = CSVLoader(file_path='./data/2021-2022 Football Player Stats.csv',encoding = 'ISO-8859-1')
#     document = pd.read_csv(os.path.join(data_folder, 'titanic.csv'))

# from langchain.chat_models import AzureChatOpenAI
# from langchain.agents import create_csv_agent
# model = AzureChatOpenAI(
#     temperature=0,
#     openai_api_base=os.environ['OPENAI_API_BASE'],
#     openai_api_version=os.environ['OPENAI_API_VERSION'],
#     deployment_name=os.environ['DEPLOYMENT_NAME'],
#     openai_api_key=os.environ['OPENAI_API_KEY'],
#     openai_api_type=os.environ['OPENAI_API_TYPE'],
# )

# agent = create_csv_agent(
#     model,
#     os.path.join(data_folder, 'titanic.csv'), 
#     verbose=True)

# if st.button("Submit"):
#     result = agent(question)
#     st.write(result["result"])
# else:
#     st.error("Please upload a document and enter a query!")

# tool_input = {
#             "input": {
#                 "name": "python",
#                 "arguments": 'visualize the age range of passengers.'
#             }
#         }
        
# response = agent.run(tool_input)

if __name__ == '__main__':
    csv_analyzer_app()