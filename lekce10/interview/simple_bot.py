from typing import Dict, Optional
from langgraph.graph import StateGraph, START, END
from langchain_openai import AzureChatOpenAI
from langchain_core.messages import SystemMessage
import os
from dotenv import load_dotenv
import logging

class SimpleBot:
    def __init__(self):
        # Setup logging
        logging.basicConfig(level=logging.DEBUG)
        self.logger = logging.getLogger("SimpleBot")
        
        # Load environment variables
        load_dotenv()
        
        # Initialize OpenAI
        self.model = AzureChatOpenAI(
            deployment_name=os.getenv("AZURE_OPENAI_DEPLOYMENT"),
            openai_api_key=os.getenv("AZURE_OPENAI_API_KEY"),
            azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
            api_version=os.getenv("AZURE_OPENAI_API_VERSION"),
            temperature=0.7
        )
        
        # Initialize state
        self.state = self.get_initial_state()
        
        # Create graph
        self.graph = self.create_graph()
    
    def get_initial_state(self) -> Dict:
        return {
            "current_step": "start",
            "user_input": None,
            "context": {},
            "response": None,
            "is_complete": False
        }
    
    def create_graph(self) -> StateGraph:
        """Create the conversation flow graph."""
        # Create graph with proper state type annotation
        workflow = StateGraph(Dict)
        
        # Add nodes
        workflow.add_node("greet", self.greet_step)
        workflow.add_node("understand", self.understand_step)
        workflow.add_node("analyze", self.analyze_step)
        workflow.add_node("respond", self.respond_step)
        workflow.add_node("check_done", self.check_done_step)
        
        # Define conditional routing
        def should_process(state: Dict) -> bool:
            """Determine if we should process input or end."""
            if not state.get("user_input"):  # First run
                return False
            return True

        def should_end(state: Dict) -> bool:
            """Check if we should end the conversation."""
            return state.get("is_complete", False)
        
        # Create branching flow
        workflow.add_edge(START, "greet")
        
        # Add conditional edges from greet
        workflow.add_conditional_edges(
            "greet",
            should_process,
            {
                True: "understand",   # Process user input
                False: "check_done"   # Just return greeting
            }
        )
        
        # Linear processing flow
        workflow.add_edge("understand", "analyze")
        workflow.add_edge("analyze", "respond")
        workflow.add_edge("respond", "check_done")
        
        # Add conditional edges from check_done
        workflow.add_conditional_edges(
            "check_done",
            should_end,
            {
                True: END,           # End conversation
                False: END           # Return current response
            }
        )
        
        return workflow.compile()
    
    def greet_step(self, state: Dict) -> Dict:
        """Initial greeting."""
        self.logger.debug("Executing greet step")
        if not state.get("response"):
            state["response"] = "Dobrý den, jak vám mohu pomoci?"
        return state
    
    def understand_step(self, state: Dict) -> Dict:
        """Process user input."""
        self.logger.debug("Executing understand step")
        if not state.get("user_input"):
            return state
            
        prompt = f"""Analyze this user input and extract key points:
        INPUT: {state['user_input']}
        
        Return a brief summary."""
        
        response = self.model.invoke([SystemMessage(content=prompt)])
        state["context"]["understanding"] = response.content
        return state
    
    def analyze_step(self, state: Dict) -> Dict:
        """Analyze the conversation context."""
        self.logger.debug("Executing analyze step")
        if not state.get("context", {}).get("understanding"):
            return state
            
        prompt = f"""Based on this understanding:
        {state['context']['understanding']}
        
        What should we focus on in our response?
        Return a brief strategy."""
        
        response = self.model.invoke([SystemMessage(content=prompt)])
        state["context"]["analysis"] = response.content
        return state
    
    def respond_step(self, state: Dict) -> Dict:
        """Generate response."""
        self.logger.debug("Executing respond step")
        if not state.get("context", {}).get("analysis"):
            return state
            
        prompt = f"""Generate a helpful response in Czech:
        Understanding: {state['context']['understanding']}
        Strategy: {state['context']['analysis']}
        
        Return ONLY the response in Czech."""
        
        response = self.model.invoke([SystemMessage(content=prompt)])
        state["response"] = response.content
        return state
    
    def check_done_step(self, state: Dict) -> Dict:
        """Check if conversation should end."""
        self.logger.debug("Executing check_done step")
        
        # Handle None input safely
        user_input = state.get("user_input")
        if user_input and isinstance(user_input, str):
            if user_input.lower() in ["konec", "nashledanou", "děkuji"]:
                state["is_complete"] = True
                state["response"] = "Děkuji za rozhovor. Na shledanou!"
                return state
        
        # If we got here, ensure we have a response
        if not state.get("response"):
            state["response"] = "Omlouvám se, nerozumím. Můžete to říct jinak?"
        
        return state
    
    def chat(self, message: Optional[str] = None) -> Dict:
        """Process a message and return response."""
        try:
            # Update state with user input
            self.state["user_input"] = message
            
            # Process through graph
            self.logger.debug(f"Processing message: {message}")
            result = self.graph.invoke(self.state)
            
            # Update state for next iteration
            self.state = {
                "current_step": "start",
                "user_input": None,
                "context": {},
                "response": None,
                "is_complete": result.get("is_complete", False)
            }
            
            # Return response
            return {
                "response": result["response"],
                "is_complete": result.get("is_complete", False)
            }
            
        except Exception as e:
            self.logger.error(f"Error in chat: {str(e)}", exc_info=True)
            return {
                "response": "Omlouvám se, došlo k chybě.",
                "is_complete": True
            } 