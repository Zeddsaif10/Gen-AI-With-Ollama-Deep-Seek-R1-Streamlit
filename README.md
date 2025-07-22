# 🧠 DeepSeek Code Companion

DeepSeek Code Companion is a powerful, locally hosted AI coding assistant designed to supercharge your development workflow. Built using Streamlit, LangChain, and Ollama, this tool enables natural, multi-turn conversations with state-of-the-art open-source LLMs such as deepseek-r1. Whether you’re writing new code, debugging existing logic, or simply looking for documentation help — DeepSeek is your always-on pair programmer.

🔍 Key Features
	•	💬 Conversational AI Chat Interface
Engage in interactive, real-time conversations with an expert-level coding assistant that remembers prior messages for contextual understanding.
	•	🧠 Local LLM Integration (via Ollama)
Runs fully offline using Ollama’s local inference engine. Choose between lightweight and larger variants of the deepseek-r1 model for optimized performance.
	•	🎨 Modern Dark-Themed UI
Includes a sleek, glassmorphic design with custom CSS for an elegant and intuitive user experience — styled for developers.
	•	🛠️ Debugging-First Prompts
Configured with system prompts that emphasize clean solutions, strategic print statements, and robust debugging practices.
	•	📦 Lightweight & Extendable Architecture
Modular design makes it easy to swap models, update prompts, or extend the chat logic using LangChain’s composability.

🧰 Tech Stack
	•	Frontend: Streamlit with custom CSS styling
	•	Backend: LangChain Prompt Templates + Ollama-hosted LLMs
	•	Models: deepseek-r1:1.5b, deepseek-r1:3b (via Ollama)

🚀 How It Works
	1.	On app launch, the sidebar allows the user to select the preferred DeepSeek model variant.
	2.	A system prompt initializes the assistant with role-specific behavior: concise, correct, and focused on debugging.
	3.	Each user message is appended to a session-based message log, maintaining conversation memory for multi-turn queries.
	4.	The prompt chain is dynamically constructed using LangChain and processed by ChatOllama, then streamed back into the UI.

📦 Installation & Setup
	1.	Install dependencies:
 
    pip install streamlit langchain langchain-ollama
      
  2. Run Ollama locally:
 
    ollama serve
    ollama pull deepseek-coder:1.5b
     
  4. Launch the app:

    streamlit run app.py
    
  
   🔐 No API keys required. All inference is done locally through Ollama.

🌟 Use Cases
	•	Explaining and refactoring Python code
	•	Generating code documentation
	•	Identifying bugs with debug print suggestions
	•	Designing solution logic for new features
	•	Answering programming-related questions

    	
     
