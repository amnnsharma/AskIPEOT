from langchain.prompts import PromptTemplate
from langchain.document_loaders import PyPDFLoader, PyPDFDirectoryLoader
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.text_splitter import CharacterTextSplitter
from langchain.vectorstores import Chroma
#from langchain.chains.question_answering import load_qa_chain
#from langchain.chains import RetrievalQA
#from langchain.chat_models import ChatOpenAI
#from langchain.llms import HuggingFacePipeline
#from langchain.llms import HuggingFaceHub
from pathlib import Path
from langchain.llms import OpenAI
from langchain.chains import LLMChain
#import numpy as np
#import faiss
import os 



llm = OpenAI(openai_api_key="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")

embeddings = HuggingFaceEmbeddings()


def load_documents():
    loader = PyPDFDirectoryLoader(str(Path.cwd())+"/reference_reports")
    reports = loader.load()  

    text_splitter = CharacterTextSplitter(chunk_size=500, chunk_overlap=10, separator='\n')
    docs = text_splitter.split_documents(reports)
    vectorstore = Chroma.from_documents(docs, embeddings, persist_directory="./chroma_report_sample_500",
                                         collection_metadata={"hnsw:space": "cosine"})

#load_documents()


#d=64
#index = faiss.IndexFlatIP(d)




'''
use_vectorstore=Chroma(persist_directory="./chroma_report_sample_500", embedding_function=embeddings)
print(use_vectorstore)
context = use_vectorstore.similarity_search_with_relevance_scores(query="""IOGPT Report about"""+
                                                                  """What is a christmas tree in the oil and gas industry?""",k=2)
context =sorted(context,key=lambda x:x[1])
relevant_context=[i[0] for i in context if i[1]>=0.5]
relevant_reports=list(set([doc.metadata["source"][-8:-4] for doc in relevant_context]))
print(relevant_reports)
sources = list(set(
            [doc.metadata["source"] for doc in context]
        ))
'''
#print(sources[0][-8:-4])

def run_llm(query,chat_history):
    if len(chat_history)!=0:
        standalone_template = """Given the following conversation and a follow up question, rephrase the follow up question
        to be a standalone question, in context of oil and gas industry.
        Chat History:
        {chat_history}
        Follow Up Input: {question}
        Standalone question:"""
        standalone_prompt=PromptTemplate(template=standalone_template, input_variables=["chat_history","question"])
        llm_chain_question = LLMChain(prompt=standalone_prompt, llm=llm)
        response_question=llm_chain_question.run({"chat_history":chat_history[-4:],"question":query})
    else:
        response_question=query
    use_vectorstore=Chroma(persist_directory="./chroma_report_sample_500", embedding_function=embeddings)
    context = use_vectorstore.similarity_search_with_relevance_scores(query="IOGPT Report about "+response_question,k=2)
    context =sorted(context,key=lambda x:x[1],reverse=True)
    relevant_context=[i[0] for i in context if i[1]>=0.7]
    relevant_reports=list(set([doc.metadata["source"][-8:-4] for doc in relevant_context]))
    print("Context:",context)
    print("Standalone question:"+str(response_question),relevant_reports)
    template="""
    Act as an expert in oil and gas sector industry and answer the question in detail. 
    If you don't know the answer, just say that you don't know, don't try to make up an answer.
    Question: {question}
    Helpful Answer:"""
    prompt=PromptTemplate(template=template, input_variables=["question"])
    llm_chain = LLMChain(prompt=prompt, llm=llm)
    response=llm_chain.run({"question":response_question})

    if len(relevant_reports)==0:
        return response
    else:
        sources_string=""
        for i, source in enumerate(relevant_reports):
            sources_string += f"{i+1}. {source}\n"
        response= response+'\n\n'+ 'Reference Reports of IPEOT: \n'+ sources_string
        return response



#print(run_llm(["What is christmas tree?"],[""]))