__import__('pysqlite3')
import sys
sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')

import streamlit as st
import openai
from langchain.document_loaders import JSONLoader, DirectoryLoader
from langchain.vectorstores import Chroma
from langchain.embeddings.openai import OpenAIEmbeddings


if 'started' not in st.session_state.keys():
    st.session_state['started'] = False


def pretty_print_docs(docs):
    return(f"\n{'-' * 100}\n".join(["Pocast title:\n" + \
                                    d.metadata['podcast_title'] + \
                                          f"\n\nDocument {i+1}:\n\n" + d.page_content for i, d in enumerate(docs)]))

def set_bg():
    '''
    A function to unpack an image from url and set as bg.
    Returns
    -------
    The background.
    '''
        
    st.markdown(
         f"""
         <style>
         .stApp {{
             background: url("https://img.freepik.com/free-vector/cloud-background-vector-cute-desktop-wallpaper_53876-136885.jpg?w=1800&t=st=1692448859~exp=1692449459~hmac=70658c3526a7bf6f5b4cc250a1c017152c2d47ccd4541b7c33153a997030f4f5");
             background-size: cover
         }}
         </style>
         """,
         unsafe_allow_html=True
     )

def get_documents(folder_path):
    def metadata_func(record: dict, metadata: dict) -> dict:

        metadata["episode_title"] = record['podcast_details'].get("episode_title")
        metadata["podcast_title"] = record['podcast_details'].get("podcast_title")

        return metadata

    loader_kwargs = {
        'jq_schema':'',
        'content_key':"podcast_highlights",
        'metadata_func':metadata_func
    }
    loader = DirectoryLoader(folder_path, glob="**/*.json", loader_cls=JSONLoader, loader_kwargs=loader_kwargs)

    data = loader.load()
    return data
    
def launch_ai_chat():

    st.title("Free Advice")

    openai.api_key = st.secrets["OPENAI_API_KEY"]
    documents = get_documents('processed_podcasts')
    embedding = OpenAIEmbeddings()
    vectordb = Chroma.from_documents(
        documents=documents,
        embedding=embedding
    )
    system_message = """
You are a friendly advisor who wants to give people the answers to their burning life questions.
You give answers from multiple viewpoints and leave it up to your advisee to choose which answer to listen to.
Use the given pieces of context separated by #### to answer questions. Only answer if you can back up your answer using stories from the context.
Each part of your answer must reference a piece of information from the context and provide details on the stories that back up your answer.
Embed the podcast title you got the information from in each explanation.
If you don't know the answer, just say that you don't know, don't try to make up an answer. 
"""
    if "openai_model" not in st.session_state:
        st.session_state["openai_model"] = "gpt-3.5-turbo"

    if "messages" not in st.session_state:
        st.session_state.messages = [
            {"role": "system", "content": system_message}
        ]

    if prompt := st.chat_input("Ask your podcasts for advice"):
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)
        docs_mmr = vectordb.max_marginal_relevance_search(prompt,k=3)

        question_and_context = f"""
####
Context: {pretty_print_docs(docs_mmr)}
####
Question: {prompt}
"""
        with st.chat_message("assistant"):
            message_placeholder = st.empty()
            full_response = ""
            for response in openai.ChatCompletion.create(
                model=st.session_state["openai_model"],
                messages=[
                    {"role": "system", "content": system_message},
                    {"role": "user", "content": question_and_context}
                ],
                stream=True,
            ):
                full_response += response.choices[0].delta.get("content", "")
                message_placeholder.markdown(full_response + "▌")
            message_placeholder.markdown(full_response)
        st.session_state.messages.append({"role": "assistant", "content": full_response})

if __name__ == '__main__':
    set_bg()
    if not st.session_state['started']:
        app_title_html = '<center><p style="color:Purple; font-size: 100px;">Cast Clarity</p></center>'
        st.markdown(app_title_html, unsafe_allow_html=True)
        cols = st.columns(3)
        with cols[1]:
            if st.button(':violet[Click to Begin]'):
                st.session_state['started'] = True
                st.experimental_rerun()
    else:
        launch_ai_chat()
