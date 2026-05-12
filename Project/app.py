import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
os.environ["TRANSFORMERS_VERBOSITY"] = "error"
import streamlit as st
from rag_backend import process_document, ask_question

#---------------------------------
# Page Setup
#---------------------------------


st.set_page_config(page_title="Simple RAG Chatbot", page_icon="🧠")

st.title("📄 Simple Document Q&A Bot")

#---------------------------------
# session state (MEMORY STORAGE)
#---------------------------------

if "vectorstore" not in st.session_state:
    st.session_state.vectorstore = None

if "messages" not in st.session_state:
    st.session_state.messages = []



#---------------------------------
# Upload file
#---------------------------------
file = st.file_uploader("Upload PDF or TXT", type=["pdf", "txt"])


#---------------------------------
# process document once
#---------------------------------
if file and st.session_state.vectorstore is None:
    with st.spinner("Processing document..."):
        st.session_state.vectorstore = process_document(file)
    st.success("Document ready!")


#---------------------------------
# chat display
#---------------------------------

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.write(msg["content"])


#---------------------------------
# input
#---------------------------------        
question = st.chat_input("Ask something from your document")


#---------------------------------
# process usr Q
#---------------------------------  

if question:
    if not st.session_state.vectorstore:
        st.error("Upload document first")
        st.stop()

    # user msg
    st.session_state.messages.append({"role": "user", "content": question})

    with st.chat_message("user"):
        st.write(question)


#---------------------------------
# AI answer
#---------------------------------  
    answer = ask_question(st.session_state.vectorstore, question)

    st.session_state.messages.append({"role": "assistant", "content": answer})

    with st.chat_message("assistant"):
        st.write(answer)