from flask import Flask, request, jsonify
from dotenv import load_dotenv
import os

load_dotenv()

# all lanchain related imports

from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import Chroma


app = Flask(__name__)

# Configure paths
UPLOAD_FOLDER = "uploads"
PDF_PATH = os.path.join(UPLOAD_FOLDER, "ultimate_resume.pdf")


# load pdf and read 

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8000, debug=True)
