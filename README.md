# chatWithData
--> Extracted data from text_segments.csv file using pandas and PyPDF by converting data extracted based on 3 pdfs and merged them.
--> This app is based on Question Answerings based on your knowledge about the data.
--> Backend
    --> Open Source LaMini model on HuggingFace converting text embeddings received from chrome Database to inference which is used by langchain
    --> Embeddings Sentence Transformers used from Open Source MiniLM-v6
    --> LaMini Model is text 2 text inference Model

--> Frontend
    --> Streamlit messages and streamlit extras for displaying of messages and prompt generated based on your questions

Structure WorkFlow

PDF Files ---> LangChain (Using Loaders and text splitters) --> Embeddings generated using sentence Transformers MiniLM-v6 --> these embeddings stored on chroma database in format of paraquet --> which is then used by LaMini Model on HuggingFace generated inferences which is again used by langChain with chaining.

