import streamlit as st
import os
from langchain_community.document_loaders import DirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain.prompts import PromptTemplate

# Define the Streamlit app
st.title("Lyric Generator App")

# Load band names based on subdirectories in lyrics_data
band_names = [d for d in os.listdir("lyrics_data") if os.path.isdir(os.path.join("lyrics_data", d))]
band_name = st.selectbox("Select Band Name", band_names)

# Language option
language = st.selectbox("Select Language", ["fr", "en"], index=0)

# User-defined theme
user_theme = st.text_input("Enter Theme")

# Load lyrics data based on selected band name
loader = DirectoryLoader(f"lyrics_data/{band_name}/", glob="**/*.txt")
docs = loader.load()

# Text splitting
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000, chunk_overlap=200, add_start_index=True
)
all_splits = text_splitter.split_documents(docs)

# Vectorstore and retriever
vectorstore = FAISS.from_documents(documents=all_splits, embedding=OpenAIEmbeddings())
retriever = vectorstore.as_retriever(search_type="similarity", search_kwargs={"k": 2})

# Language model
llm = ChatOpenAI(model_name='gpt-3.5-turbo', temperature=0.1)


def generate_lyrics():
    prompt_language = "fr" if language == "fr" else "en"
    prompt_template = PromptTemplate.from_template(
        f"""You are a songwriter.
            Your task is to create lyrics for a new song on the given theme.
            The style should be similar to the lyrics of {band_name} whose lyrics are provided in the context.
            Do not copy the lyrics of {band_name}, but only draw inspiration from their style.
            Begin the lyrics with a verse that captures the essence of the theme in the style of {band_name}. Feel free to explore the emotions and narrative elements characteristic of their songs.
            Theme: {user_theme}
            """
    ) if prompt_language == "en" else PromptTemplate.from_template(
        f"""Vous êtes un parolier.
            Votre tâche est de créer des paroles pour une nouvelle chanson sur le thème donné dans la question.
            Le style doit être similaire à celui des paroles de {band_name} dont les paroles sont données dans le contexte.
            Il ne faut en aucun cas copier les paroles de {band_name} mais uniquement s'inspirer de leur style.
            Commencez les paroles par un couplet qui capture l'essence du thème dans le style de {band_name}. N'hésitez pas à explorer les émotions et les éléments narratifs caractéristiques de leurs chansons.
            Thème: {user_theme}
            """
    )

    rag_chain = (
        {"context": retriever | format_docs, "question": RunnablePassthrough()}
        | prompt_template
        | llm
        | StrOutputParser()
    )
    return rag_chain.invoke(user_theme)

def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)

# Generate and display lyrics on button click
if st.button("Generate Lyrics"):
    generated_lyrics = generate_lyrics()
    st.text(generated_lyrics)
