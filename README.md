# Building a Conversational Web Application for PDF Documents using Mistral-7B-v0.1

This project is a web application that allows users to interact with and query text extracted from multiple PDF documents using a conversational AI interface. Built with Streamlit and various NLP tools, this application provides an intuitive way to search through PDF content and receive relevant answers.

## Features

- **PDF Upload**: Upload multiple PDF files.
- **Text Extraction**: Extract text from the uploaded PDFs.
- **Text Chunking**: Break text into manageable chunks.
- **Vector Store**: Store and retrieve text chunks using FAISS.
- **Conversational AI**: Interact with a conversational AI model to query the extracted text.
- **Streamlit Interface**: User-friendly web interface for interaction.

## Technology Stack

- **Streamlit**: For creating interactive web applications.
- **LangChain**: For managing language model interactions and vector stores.
- **HuggingFace Transformers**: For language model embeddings and inference.
- **FAISS**: For efficient similarity search.
- **PyPDF2**: For PDF text extraction.

## Installation

To set up and run this project, follow these steps:

### 1. Clone the Repository

```bash
git clone https://github.com/your-username/pdf-chatbot-web-app.git
cd pdf-chatbot-web-app
```

### 2. Set Up a Virtual Environment

Create and activate a virtual environment:

```bash
python -m venv venv
source venv/bin/activate  # On Windows use `venv\Scripts\activate`
```

### 3. Install Dependencies

Install the required Python packages:

```bash
pip install -r requirements.txt
```

### 4. Set Up Environment Variables

Create a `.streamlit/secrets.toml` file with your HuggingFace API key:

```toml
[HUGGINGFACEHUB_API_KEY]
api_key = "your_huggingface_api_key"
```

Replace `"your_huggingface_api_key"` with your actual HuggingFace API key.

## Usage

### 1. Run the Application

Start the Streamlit application:

```bash
streamlit run app.py
```

### 2. Interact with the Application

- **Upload PDFs**: Use the file uploader in the sidebar to upload your PDF documents.
- **Ask Questions**: Enter your queries in the text input field and click on the "Submit" button to get answers based on the uploaded PDFs.
- **Processing**: Click on the "Process" button to process the uploaded PDFs and prepare the application for querying.

## Code Overview

- **`main.py`**: Main application file containing the Streamlit interface and core functionality.
- **`htmlTemplates.py`**: Contains HTML templates used for styling the chat interface.
- **`requirements.txt`**: Lists the Python packages required for the project.

## Example Code

Here’s a brief overview of the key functions in `main.py`:

### PDF Processing

```python
def get_pdf_text(pdf_docs):
    text = ""
    for pdf in pdf_docs:
        pdf_reader = PdfReader(pdf)
        for page in pdf_reader.pages:
            text += page.extract_text()
    return text
```

### Text Chunking

```python
def get_text_chunks(text):
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
        length_function=len
    )
    chunks = text_splitter.split_text(text)
    return chunks
```

### Vector Store Creation

```python
def get_vectorstore(text_chunks):
    model_id = "sentence-transformers/all-MiniLM-L6-v2"
    model_kwargs = {'device': 'cpu'}
    encode_kwargs = {'normalize_embeddings': False}
    embeddings = HuggingFaceEmbeddings(
        model_name=model_id,
        model_kwargs=model_kwargs,
        encode_kwargs=encode_kwargs
    )
    vectorstore = FAISS.from_texts(texts=text_chunks, embedding=embeddings)
    return vectorstore
```

### Conversational AI

```python
def get_conversation_chain(vectorstore):
    memory = ConversationBufferMemory(memory_key='chat_history', return_messages=True)
    llm = HuggingFaceHub(repo_id="mistralai/Mistral-7B-v0.1", model_kwargs={"temperature":0.5, "max_length":512})
    conversation_chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        chain_type="stuff",
        retriever=vectorstore.as_retriever(search_kwargs={"k": 3}),
        memory=memory
    )
    return conversation_chain
```

## Contributing

Contributions are welcome! If you have suggestions, improvements, or bug fixes, please create a pull request or open an issue.

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

## Contact

For questions or feedback, please contact [rafsunsheikh116@gmail.com](mailto:rafsunsheikh116@gmail.com).
