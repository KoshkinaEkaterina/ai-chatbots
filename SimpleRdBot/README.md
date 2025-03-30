# 🤖 RD-Bot: Educational Assistant for Robot Dreams Students

**RD-Bot** is an AI-powered assistant built with FastAPI and OpenAI to support students of Robot Dreams courses. It understands user questions using intent classification, maintains a modular architecture, and provides both hardcoded and LLM-generated responses.

---

## ✨ Features

- ✅ Intent classification using simple rules
- 💬 Natural answers for educational queries
- 📚 GPT-powered explanations of AI/tech terms
- 🧠 Modular architecture (handlers, state, classification)
- ⚡ FastAPI backend with Swagger UI support

---

## 🧠 Supported Intents

| Intent                 | Description                                      |
|------------------------|--------------------------------------------------|
| `ASK_SCHEDULE`         | Shows date/time of upcoming class               |
| `ASK_PAYMENT`          | Explains how to pay for a course                |
| `REQUEST_CERTIFICATE`  | Informs student when they’ll receive certificate |
| `EXPLAIN_TERM`         | Uses GPT to explain a technical term            |
| `RECOMMEND_COURSE`     | Recommends next course based on current one     |
| `ASK_HOMEWORK_STATUS`  | Gives update on submitted or upcoming homework  |
| `TROUBLE_LOGIN`        | Helps with login issues                         |
| `UNKNOWN`              | Fallback for unrecognized questions             |

---

## 🚀 How to Run
1. Create .env file with your OpenAI API key
OPENAI_API_KEY=sk-...

2. Start the server
uvicorn SimpleRdBot.main:app --reload

3. Test in your browser
   Go to http://localhost:8000/docs to open Swagger UI and test /chat endpoint.



## 🛠 Project Structure
SimpleRdBot/
├── main.py         # FastAPI entry point
├── agent.py        # Coordinates intent + state
├── intents.py      # Classifies user intent
├── handlers.py     # Logic for each intent
├── state.py        # Manages user state in memory
├── .env            # API key config

## 📘 Notes
- The bot is designed for educational support but can be extended for other domains.
- Explanations are powered by OpenAI GPT-4 using LangChain's ChatOpenAI.
- For homework, responses are a mix of fixed and AI-generated content.