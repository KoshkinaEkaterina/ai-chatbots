from typing import List
from langchain_core.messages import SystemMessage
from langchain_openai import AzureChatOpenAI
import os
from dotenv import load_dotenv
from random import random, choice

class TeacherBot:
    def __init__(self):
        # Load environment variables
        load_dotenv()
        
        # Initialize Azure OpenAI
        self.model = AzureChatOpenAI(
            azure_deployment=os.getenv("AZURE_OPENAI_DEPLOYMENT"),
            api_key=os.getenv("AZURE_OPENAI_API_KEY"),
            azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
            openai_api_version=os.getenv("AZURE_OPENAI_API_VERSION"),
            temperature=0.2
        )

    def respond(self, question: str) -> str:
        """Generate a contextually appropriate teacher response."""
        
        # First check if the question is about difficult situations
        difficult_situation_keywords = [
            "problém", "náročn", "konflikt", "incident", "chování", "řešit", "situac",
            "zasáhnout", "krize", "těžk", "šikan", "násilí", "agres"
        ]
        
        is_about_difficulties = any(keyword in question.lower() for keyword in difficult_situation_keywords)
        
        # If the question is about difficult situations, 80% chance to share a serious experience
        if is_about_difficulties and random() < 0.8:
            prompt = f"""Generate a very brief teacher's response (max 20 words) in Czech about a difficult classroom situation.
            Question asked: {question}
            
            IMPORTANT:
            - Keep it under 20 words
            - Make it feel natural and spontaneous
            - Focus on one specific moment or detail
            - Include emotional impact briefly
            
            Response should be concise but meaningful."""
            
            response = self.model.invoke([SystemMessage(content=prompt)])
            return response.content
        
        # 30% chance of giving a non-standard response (these can be throw-offs)
        if random() < 0.3:
            response_types = [
                # Questioning the interview process
                {
                    "type": "meta",
                    "responses": [
                        "Můžete mi vysvětlit, proč se ptáte zrovna na tohle?",
                        "Nejsem si jistá, jestli je vhodné o tomhle mluvit. Jaký je účel těchto otázek?",
                        "Než odpovím, chtěla bych vědět, jak s těmito informacemi naložíte.",
                        "Tohle je docela osobní téma. Můžete mi říct více o tom, proč vás to zajímá?",
                    ]
                },
                # Confusion about question
                {
                    "type": "confusion",
                    "responses": [
                        "Promiňte, ale není mi úplně jasné, na co se ptáte. Můžete to formulovat jinak?",
                        "Ta otázka je dost složitá. Můžete ji nějak zjednodušit?",
                        "Nevím, jestli správně chápu, co chcete vědět...",
                        "Tohle je hodně komplexní téma. Můžeme to rozebrat po částech?",
                    ]
                },
                # Process questions
                {
                    "type": "process",
                    "responses": [
                        "Jak dlouho tento rozhovor ještě potrvá?",
                        "Kolik takových rozhovorů už jste dělali?",
                        "Kdo všechno bude mít přístup k těmto informacím?",
                        "Můžeme si udělat krátkou přestávku? Je toho na mě hodně.",
                    ]
                }
            ]
            
            response_type = choice(response_types)
            return choice(response_type["responses"])
        
        # For normal responses, generate contextually appropriate content
        prompt = f"""Generate a very brief teacher's response (max 20 words) in Czech to this question: {question}
        
        Make it:
        - Maximum 20 words
        - Natural and specific
        - Focus on one concrete example
        - Use conversational Czech"""
        
        response = self.model.invoke([SystemMessage(content=prompt)])
        return response.content 