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
            import matplotlib as plt
            ...
            ```
            Example:
            ```python
            import matplotlib.pyplot as plt
            plt.hist(df['Age'])
            plt.title('Age Range of All Passengers')
            plt.xlabel('Age')
            plt.ylabel('Count')
            plt.show()
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

def init_agent(file_path):
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
    return agent

def csv_agent_func2(agent, query):
    """Run the CSV agent with the given file path and user message."""
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
    # if "code" not in response:
        # return None

    code_pattern = r"```python(.*?)```"
    match = re.search(code_pattern, response, re.DOTALL)
    
    if match:
        # Extract the matched code and strip any leading/trailing whitespaces
        return match.group(1).strip()
    return None

def csv_analyzer_app():
    """Main Streamlit application for CSV analysis."""
    st.title('HSBC HW: CSV analyzer')
    
    option = st.selectbox(
        "How would you like to start?",
        ("Upload your own data", "Use default Titanic data"),
        index=None,
        placeholder="Select one option...",
    )
    st.write('You selected:', option)
    
    # st.info("If no file is uploaded, this app will use the default data: titanic.csv!")
    file_path = None
    if option == "Upload your own data":
        st.write('Please upload your CSV file and enter your query below:')
        uploaded_file = st.file_uploader("Choose a CSV file", type="csv")

        file_details = {"FileName": uploaded_file.name, "FileType": uploaded_file.type, "FileSize": uploaded_file.size}
        st.write(file_details)

        # Save the uploaded file to disk
        file_path = os.path.join("/tmp", uploaded_file.name)
        with open(file_path, "wb") as f:
            f.write(uploaded_file.getbuffer())



    elif option == "Use default Titanic data":
        file_path = './data/titanic.csv'

    if file_path is not None:
        st.info("csv file is: {}".format(file_path))
    else:
        st.error("You have to select one option!")
    
    if file_path is not None:
        df = pd.read_csv(file_path)
        st.write("Preview: ")
        st.dataframe(df[:10])

        agent = init_agent(file_path)
        user_input = st.text_input("Your query")
        if st.button('Run'):
            response = csv_agent_func2(agent, user_input)
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

    # if uploaded_file is not None:
    st.divider()


if __name__ == '__main__':
    csv_analyzer_app()