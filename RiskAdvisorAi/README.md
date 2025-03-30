# 🧠 Risk Advisor AI — Final Project

This project is a proof-of-concept for an intelligent assistant that supports Risk Management and Trading teams. 
It demonstrates advanced chatbot orchestration, retrieval-augmented generation (RAG), and contextual responses based 
on user authorization.

---

## 🛡 Data Ethics
This project does not use any real company or client data.
All data, user profiles, and procedures are fictional and created solely for educational purposes.

## 🎯 Project Goal

To simulate how an AI chatbot could assist teams in:
- Validating trading limits and product permissions
- Retrieving internal risk procedures and escalation rules
- Answering general financial questions using GPT-4
- Providing contextual and role-specific responses (Trader / Risk / Compliance)

---

## 🧱 Architecture Overview

The bot is built as a modular, agent-based architecture:
User → Orchestrator → SQLAgent / DocAgent / GPTAgent → Confidence Evaluator → Response


### Components:
- **SQLAgent** — Queries a fake SQL database (e.g. VaR, PnL, limits, allowed products)
- **DocAgent** — Retrieves answers from an internal PDF using Pinecone vector store
- **GPTAgent** — Handles fallback and general knowledge using GPT-4
- **Evaluator** — Scores each answer by confidence (high / medium / low)
- **OAuth layer** — (Simulated) Authenticates user and identifies access rights

---

## 🗂 Tech Stack

- Python (FastAPI)
- OpenAI GPT-4 + Embeddings (`text-embedding-ada-002`)
- Pinecone Vector Store (document retriever)
- SQLite (for mocked SQL risk data)
- LangChain (agent orchestration)
- Swagger (API testing UI)
- dotenv (.env secrets)

---

## 🔐 Authentication & Authorization

The system uses simulated user profiles to:
- Limit what data each trader can access
- Customize VaR, PnL, product limits
- Personalize answers based on role (`user_id`, `desk`)

---

## 🧪 Example User Prompts

| Prompt | Source |
|--------|--------|
| "What is my trading limit for Desk A?" | SQLAgent |
| "Explain the approval process for a new product." | DocAgent (via PDF) |
| "What is convexity adjustment?" | GPTAgent |
| "Show me the VaR chart for the last month." | SQL + Plot |

---

## 📄 Document Retrieval (RAG)

- Document used: `Risk_Procedures.pdf`
- Content includes escalation rules, approval workflows, internal reporting
- Embedded into Pinecone using OpenAI embeddings (1536-dim)

---

## 🧠 Confidence Evaluation

Each answer is scored:
- Based on data source
- Whether key fields were filled (e.g. date, value)
- Shown in the response as `high`, `medium`, or `low` confidence

---

## 🧪 Testing

- Run backend:
  ```bash
  uvicorn main:app --reload
