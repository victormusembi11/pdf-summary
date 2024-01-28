import os
import gradio as gr
from langchain_openai import OpenAI
from langchain.prompts import PromptTemplate
from langchain.chains.summarize import load_summarize_chain
from langchain_community.document_loaders import PyPDFLoader

os.environ["OPENAI_API_KEY"] = "your_api_key_here"

llm = OpenAI(temperature=0)


def summarize_pdf(pdf_file_path):
    loader = PyPDFLoader(pdf_file_path)
    docs = loader.load_and_split()
    chain = load_summarize_chain(
        llm, chain_type="map_reduce"
    )  # checkout other chain types
    summary = chain.invoke(docs)
    return summary


summarize = summarize_pdf("bitcoin.pdf")
print(summarize)
