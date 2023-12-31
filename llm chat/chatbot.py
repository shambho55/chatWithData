
import streamlit as st
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from transformers import pipeline
import torch
import base64
import textwrap
from langchain.embeddings import SentenceTransformerEmbeddings
from langchain.vectorstores import chroma
from langchain.llms import huggingface_pipeline
from langchain.chains import RetrievalQA
from constants import CHROMA_SETTINGS
from streamlit_chat import message

checkpoint = "LaMini-T5-738M"

tokenizer = AutoTokenizer.from_pretrained(checkpoint)
base_model = AutoModelForSeq2SeqLM.from_pretrained(
    checkpoint,
    device_map="auto",
    torch_dtype = torch.float32
)

@st.cache_resource
def llm_pipeline():
    pipe = pipeline(
        'text2text-generation',
        model = base_model,
        tokenizer=tokenizer,
        max_length = 256,
        do_sample = True,
        temperature=0.3,
        top_p = 0.95,
    )
    local_llm = huggingface_pipeline(pipeline=pipe)
    return local_llm

@st.cache_resource
def qa_llm():
    llm = llm_pipeline()
    embeddings = SentenceTransformerEmbeddings(model_name="all-MiniLM-L6-v2")
    db = chroma(persist_directory="db",embedding_function = embeddings,client_settings = CHROMA_SETTINGS)
    retriever = db.as_retriever()
    qa = RetrievalQA.from_chain_type(
        llm = llm,
        chain_type="stuff",
        retriever=retriever,
        return_source_document = True
    )
    return qa

def process_answer(instruction):
    response = ''
    instruction = instruction
    qa = qa_llm()
    generated_text = qa(instruction)
    answer = generated_text['result']
    return answer, generated_text

#Display conversation history using streamlit messages
def display_conversation(history):
    for i in range(len(history["generated"])):
        message(history["past"][i], is_user=True, key=str(i) + "_user")
        message(history["generated"][i],key=str(i))

def main():
    st.title("Chat With Your PDF files 🐱 📕")
    with st.expander("About the App"):
        st.markdown(
            '''
            This is a Chatbot App build using Generative AI 
            which responds to your questions about the PDF.
            '''
        )

    user_input = st.text_input("",key="input")

    #Initialize session state for generated responses and past messages
    if "generated" not in st.session_state:
        st.session_state["generated"] = ["Kudos! I am here for your help."]
    if "past" not in st.session_state:
        st.session_state["past"] = ["Hi! there."]

    # Search the database for a response based on user input and update session state
    if user_input:
        answer = process_answer({"query" : user_input})
        st.session_state["past"].append(user_input)
        response = answer
        st.session_state["generated"].append(response)

    #Display conversation history using Streamlit messages
    if st.session_state["generated"]:
        display_conversation(st.session_state)


if __name__ == "__main__":
    main()