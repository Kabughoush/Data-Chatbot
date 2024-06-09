import os
import sqlite3
import pandas as pd
from langchain_openai import ChatOpenAI
from langchain.chains import LLMChain
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder, FewShotChatMessagePromptTemplate
from operator import itemgetter
import streamlit as st

# Set the OpenAI API key
os.environ["OPENAI_API_KEY"] = st.secrets["OPENAI_API_KEY"]

# Path to the SQLite database
db_path = 'data.sqlite'  # Ensure this path is correct and the file is in the repo

# Initialize the language model
llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0)

# Create few-shot examples
examples = [
    {
        "input": "List all customers in France with a credit limit over 20,000.",
        "query": "SELECT * FROM customers WHERE country = 'France' AND creditLimit > 20000;"
    },
    {
        "input": "Get the highest payment amount made by any customer.",
        "query": "SELECT MAX(amount) FROM payments;"
    },
    {
        "input": "Show product details for products in the 'Motorcycles' product line.",
        "query": "SELECT * FROM products WHERE productLine = 'Motorcycles';"
    },
    {
        "input": "Retrieve the names of employees who report to employee number 1002.",
        "query": "SELECT firstName, lastName FROM employees WHERE reportsTo = 1002;"
    },
    {
        "input": "List all products with a stock quantity less than 7000.",
        "query": "SELECT productName, quantityInStock FROM products WHERE quantityInStock < 7000;"
    },
    {
        'input': "what is price of `1968 Ford Mustang`",
        "query": "SELECT `buyPrice`, `MSRP` FROM products WHERE `productName` = '1968 Ford Mustang' LIMIT 1;"
    }
]

# Create the prompt templates
sql_template = """
Translate the following natural language query to a SQL query:
{query}
"""

answer_template = """
Given the following user question, corresponding SQL query, and SQL result, answer the user question.
Question: {question}
SQL Query: {query}
SQL Result: {result}
Answer:
"""

# Creating few-shot example prompt
example_prompt = ChatPromptTemplate.from_messages(
    [
        ("human", "{input}\nSQLQuery:"),
        ("ai", "{query}"),
    ]
)
few_shot_prompt = FewShotChatMessagePromptTemplate(
    example_prompt=example_prompt,
    examples=examples,
    input_variables=["input"],
)

# Get table information from the SQLite database using Pandas
def get_table_info(db_path):
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
    tables = cursor.fetchall()
    table_info = ""
    for table in tables:
        table_name = table[0]
        cursor.execute(f"PRAGMA table_info({table_name});")
        columns = cursor.fetchall()
        column_info = ", ".join([f"{col[1]} ({col[2]})" for col in columns])
        table_info += f"Table {table_name}: {column_info}\n"
    conn.close()
    return table_info

table_info = get_table_info(db_path)

# Create final prompt with few-shot examples and system message
final_prompt = ChatPromptTemplate.from_messages(
    [
        ("system",
         f"You are a MySQL expert. Given an input question, create a syntactically correct MySQL query to run. Unless otherwise specified.\n\nHere is the relevant table info: {table_info}\n\nBelow are a number of examples of questions and their corresponding SQL queries."),
        few_shot_prompt,
        MessagesPlaceholder(variable_name="messages"),
        ("human", "{input}"),
    ]
)

# Initialize history
history = ChatMessageHistory()

# Create chains
generate_query = LLMChain(llm=llm, prompt=final_prompt)
rephrase_answer = LLMChain(llm=llm, prompt=PromptTemplate.from_template(answer_template))

chain = (
        generate_query |
        rephrase_answer
)

# Function to handle natural language queries
def handle_nl_query(nl_query, db_path, history):
    try:
        # Add user query to history
        history.add_user_message(nl_query)

        # Generate SQL query from natural language query
        sql_query = generate_query.run({"input": nl_query, "messages": history.messages})
        print(f"Generated SQL query: {sql_query}")

        # Check if the generated query is valid SQL
        if not sql_query.strip().lower().startswith("select"):
            raise ValueError("Generated text is not a valid SQL query")

        # Execute the SQL query using Pandas
        conn = sqlite3.connect(db_path)
        df_result = pd.read_sql_query(sql_query, conn)
        conn.close()

        # Convert DataFrame to string
        result_str = df_result.to_string(index=False)

        # Add SQL query to history
        history.add_ai_message(sql_query)

        # Rephrase the SQL result into a natural language answer
        answer = rephrase_answer.run({"question": nl_query, "query": sql_query, "result": result_str})
        print(f"Answer: {answer}")

        # Add the final answer to history
        history.add_ai_message(answer)

        return answer, df_result

    except Exception as e:
        error_message = f"An error occurred: {str(e)}"
        print(error_message)

        # Add the error message to history
        history.add_ai_message(error_message)

        return error_message, pd.DataFrame()

# Streamlit app
def main():
    st.title("Data Chatbot")
    st.write("Ask your data-related questions below:")

    user_input = st.text_area("You:", key="input")

    if st.button("Send"):
        if user_input:
            st.write(f"You: {user_input}")
            answer, _ = handle_nl_query(user_input, db_path, history)
            st.write(f"Bot: {answer}")

if __name__ == "__main__":
    main()
