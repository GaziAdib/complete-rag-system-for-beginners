from flask import Flask, request, jsonify
from dotenv import load_dotenv
import os

load_dotenv()

# all lanchain related imports

from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma

# LLm 
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate, PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough


app = Flask(__name__)

# Configure paths
UPLOAD_FOLDER = "uploads"
PDF_PATH = os.path.join(UPLOAD_FOLDER, "ultimate_resume.pdf")


# load pdf and read

    # loader = PyPDFLoader(PDF_PATH)

    # documents = loader.load()

    # print(documents)

    # result = [doc.page_content for doc in documents]

    # return jsonify({"response": result})


#fix LLM
llm = ChatGroq( 
        api_key=os.getenv('GROQ_API_KEY'), 
        model="llama-3.1-8b-instant", 
        temperature=0.0, 
        max_tokens=100, 
        max_retries=2 
    )

@app.route('/pdf-read', methods=['POST'])
def read_pdf():
    
    loader = PyPDFLoader(PDF_PATH)

    #documents = loader.load()
    documents = loader.load_and_split()

    print(documents)

    result = [doc.page_content for doc in documents]

    text_splitter = RecursiveCharacterTextSplitter( 
        chunk_size=500,
        chunk_overlap=200,
        length_function=len,
        is_separator_regex=False
    )

    chunks = text_splitter.split_documents(documents)

    chunks[0].page_content  # The chunked text 
    chunks[0].metadata    # Page number etc.  

    # get each chunk details from the pdf
    #return jsonify({"response": chunks[2].page_content})

    # return jsonify({"response": result})


    # Fix an Embedding Model
    embedding_model = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")  # Correct

    # add chunks to embedding 
    db = Chroma.from_documents(
        documents=chunks,
        embedding=embedding_model,
        persist_directory="chroma_db"
    )

    db.persist()


    # use vector store to retrrive

    vectorstore = Chroma(
        persist_directory="chroma_db",
        embedding_function=embedding_model
    )

    retriever = vectorstore.as_retriever(search_kwargs={'k': 3})
    
    # retriever = vectorstore.as_retriever( 
    #     search_type="similarity_score_threshold", 
    #     search_kwargs={"score_threshold": 0.5} 
    # )

    #retriever = vectorstore.as_retriever(search_type="mmr")

    query = "What is the project name ?" 

    docs = retriever.get_relevant_documents(query)

    for d in docs:
        print(d.page_content[:300])  # Preview the most relevant chunks

    # Serialize documents for JSON response
    # serialized_docs = []
    # for doc in docs:
    #     serialized_docs.append({
    #         "page_content": doc.page_content,
    #         "metadata": doc.metadata  # Includes page numbers, source file, etc.
    #     })
    
    # return jsonify({
    #     "query": query,
    #     "results": serialized_docs,
    #     "count": len(docs)
    # }) 

    # define Promt Template

    request_data = request.get_json()
    user_query = request_data.get('query', '')

    if not user_query:
        return jsonify({"error": "No query provided"}), 400

    # 3. Optimized prompt template
    prompt_template = """Answer the question based only on this context:
        {context}

        Question: {question}

        Guidelines:
        - add Markdown formate
        - Be concise (2-5 sentences)
        - If unsure, say "I couldn't find relevant information"
        - Never hallucinate facts"""
    
    prompt = PromptTemplate.from_template(prompt_template)

    chain = (
        {'context': RunnablePassthrough(), 'question': RunnablePassthrough()}
        | prompt
        | llm 
        | StrOutputParser()
    )


    # 5. Streamlined invocation
    answer = chain.invoke({
            "context": "\n\n".join(d.page_content for d in docs),
            "question": user_query
        })

    return jsonify({
        "query": user_query,
        "answer": answer
    })




  



if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8000, debug=True)
