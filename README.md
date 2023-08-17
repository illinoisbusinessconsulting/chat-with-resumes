# Resume Chatbot

This project is a chatbot that answers questions about resumes of various individuals. It uses the LangChain library to process natural language and retrieve relevant information from a database of resumes. The chatbot is built with Gradio for the user interface.

## Key Files 

- app.py: This is the main application file where the chatbot is configured and launched.
- callbacks.py: This file contains the callback handler for streaming responses from the chatbot.
- requirements.txt: This file lists the Python packages that need to be installed for the chatbot to run.  
- /docs: This directory contains the documents (resumes) that the chatbot uses to answer questions.
- .env: This file contains environment variables such as the OpenAI API key.

## Setup

1. Install the required Python packages:

    ```
    pip install -r requirements.txt
    ```

2. Setup your environment variables in the .env file. For example:

    ```
    OPENAI_API_KEY = 'your-openai-api-key'
    ```

3. Run the application:

    ```	
    python app.py
    ```

    The chatbot will be accessible at http://localhost:7860.

## How It Works

The chatbot uses the LangChain library to process natural language and retrieve relevant information from a database of resumes. The resumes are stored as documents in the /docs directory. When a user asks a question, the chatbot uses an OpenAI model to generate a response. The response is streamed back to the user in real-time. The StreamingGradioCallbackHandler class in callbacks.py handles the streaming of responses. It puts each new token generated by the model into a queue, which is then read by the main application in app.py. The app.py file sets up the chatbot and launches the Gradio interface. It creates an instance of the StreamingGradioCallbackHandler and passes it to the AgentExecutor, which executes the chatbot's actions. The AgentExecutor runs in a separate thread and sends user inputs to the chatbot. When the chatbot generates a new token, the StreamingGradioCallbackHandler puts it into the queue. The main application reads from the queue and updates the chatbot's response in the Gradio interface.

## Customization

You can customize the chatbot by modifying the app.py file. For example, you can change the OpenAI model used by the chatbot, or you can add more tools to the tools list to enable the chatbot to perform additional tasks. You can also customize the callback handler by subclassing StreamingGradioCallbackHandler and overriding its methods. For example, you could add additional logging or error handling.

## Note

This chatbot is designed to work with resumes stored as PDF files in the /docs directory. If you want to use a different format or source of data, you will need to modify the configure_retriever function in app.py.