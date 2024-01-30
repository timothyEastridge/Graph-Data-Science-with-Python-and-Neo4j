import streamlit as st
import yaml
import os
import sys  
import io
import keyring
from neo4j import GraphDatabase
from langchain.chains import GraphCypherQAChain
from langchain.chat_models import ChatOpenAI
from langchain.prompts import PromptTemplate
from langchain.graphs import Neo4jGraph

# PARAMETERS
# Load Neo4j connection parameters
with open('config.yaml', 'r') as file:
    cfg = yaml.safe_load(file)
user, uri = cfg['neo4j_user'], cfg['neo4j_db_uri']
pw = keyring.get_password('eastridge', user)

# Neo4j Driver
driver = GraphDatabase.driver(uri, auth=(user, pw))

# OpenAI and Neo4j Setup
os.environ['OPENAI_API_KEY'] = keyring.get_password('eastridge', 'openai')
graph = Neo4jGraph(url=uri, username=user, password=pw)
llm = ChatOpenAI()

# Cypher template and instantiation
cypher_template = "You are an expert in Cypher queries for patents and you ALWAYS respond in fewer than 4000 tokens.\nSchema: {schema}\nQuestion: {question}"
cypher_prompt = PromptTemplate(template=cypher_template, input_variables=["schema", "question"])
cypher_chain = GraphCypherQAChain.from_llm(llm, graph=graph, cypher_prompt=cypher_prompt, verbose=True)

def get_patents_with_verbose_output(description: str):
    old_stdout = sys.stdout
    sys.stdout = buffer = io.StringIO()

    try:
        # Run the chain and capture verbose output
        result = cypher_chain.run(description)
        verbose_output = buffer.getvalue()
    except Exception as e:
        st.error(f"Error: {e}")
        result = ""
        verbose_output = ""
    finally:
        # Reset stdout
        sys.stdout = old_stdout

    return result, verbose_output

def main():
    st.title("Patent Search")
    st.sidebar.write("\n\n\n\n\n\n")
    with st.sidebar:
        description = st.text_area("Describe the patent", key="desc_input")
        show_cypher = st.checkbox("Show Cypher Code", value=False)

        if st.button('Search Patents'):
            st.session_state['patents'], st.session_state['verbose_output'] = get_patents_with_verbose_output(description)

    if 'patents' in st.session_state:
        # st.subheader('Patents')
        st.sidebar.write("\n\n\n\n\n\n")
        st.write(st.session_state['patents'])

    if show_cypher and 'verbose_output' in st.session_state:
        
        st.sidebar.write("\n\n\n\n\n\n") 
        st.sidebar.subheader("Generated Cypher Query")
        st.sidebar.write("\n\n") 
        # st.sidebar.empty()
        st.sidebar.text(st.session_state['verbose_output'])



if __name__ == '__main__':
    main()


# streamlit run streamlit_for_patents_v2.py