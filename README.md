# DocMind: Your Personal AI Document Assistant

DocMind is a user-friendly, web-based application built with Streamlit, designed to simplify how you interact with your
documents. Powered by Cohere's cutting-edge language models, DocMind offers a seamless experience for extracting
insights and information from your files.

## Key Features

- **Document Chat:** Upload your documents (currently supports PDF format) and engage in natural language conversations
  to find specific information. DocMind intelligently analyzes your questions and retrieves relevant answers directly
  from the content.
- **ChatBot:** Interact with a powerful language model for general conversations and knowledge exploration.
- **Cohere Web Search Integration:** Enhance the depth and scope of your answers by seamlessly integrating web search
  results, providing a broader perspective on your queries.
- **User Authentication and Data Management:** Securely manage your documents and interactions with user login and data
  export capabilities.

## Getting Started

1. **Clone the Repository:**
   ```bash
   git clone https://github.com/abdalrohman/DocMind.git
   cd DocMind
   ```

2. **Install Dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

3. **Set Up Your Cohere API Key:**
    - Create a `.env` file in the userdata directory inside your username dir.
    - Add your Cohere API key to the `.env` file:
      ```
      COHERE_API_KEY='YOUR_COHERE_API_KEY'
      ```

4. **Run the Application:**

   ```bash
   streamlit run app.py
   ```

## Usage

- **Uploading Documents:**  Navigate to the "DocumentsChat" page and upload your PDF documents. The application will
  process and index them for efficient retrieval.
- **Chatting with Documents:** Ask questions about your uploaded documents. DocMind will analyze your input, extract
  relevant information, and provide you with concise and cited answers.
- **Using the ChatBot:**  Visit the "ChatBot" page for natural language conversations powered by Cohere's language
  models.
- **Web Search Integration:** Enable the web search feature to broaden the scope of your answers with relevant
  information from the internet.
- **Exporting Your Data:** Download your uploaded documents and chat history from the sidebar for easy access and
  backup.

## Technologies Used

- Streamlit
- Langchain
- Cohere
- ChromaDB

## Contributing

Contributions are welcome! If you have any suggestions, bug reports, or feature requests, please feel free to open an
issue or submit a pull request.

## License

This project is licensed under the MIT License.
