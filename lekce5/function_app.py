import azure.functions as func
import logging
import json
from interview_bot import InterviewBot

# Initialize bot and sessions
bot = InterviewBot()
sessions = {}

def get_default_state():
    """Initialize a new session state"""
    state = {
        "topics": bot.load_topics(bot.topics_path),
        "current_topic_id": "T1",
        "current_question": None,
        "user_message": None,
        "conversation_history": [],
        "interview_complete": False
    }
    # Generate first question
    state = bot.generate_question(state)
    return state

app = func.FunctionApp()

@app.route(route="chat", auth_level=func.AuthLevel.ANONYMOUS)
def chat_endpoint(req: func.HttpRequest) -> func.HttpResponse:
    # Add CORS headers
    headers = {
        "Access-Control-Allow-Origin": "*",
        "Access-Control-Allow-Methods": "POST, OPTIONS",
        "Access-Control-Allow-Headers": "Content-Type",
    }
    
    # Handle CORS preflight
    if req.method == "OPTIONS":
        return func.HttpResponse(status_code=200, headers=headers)

    try:
        logging.info('Python HTTP trigger function processed a request.')
        
        # Log request details
        body = req.get_json()
        logging.info(f'Request body: {body}')
        
        session_id = body.get('session_id')
        message = body.get('message', '')
        
        # No session_id means new session
        if not session_id:
            logging.info('Starting new session')
            state = get_default_state()
            
            # Generate session ID
            import uuid
            session_id = str(uuid.uuid4())
            sessions[session_id] = state
            
            response_data = {
                "session_id": session_id,
                "question": state["current_question"]
            }
            logging.info(f'New session response: {response_data}')
            
            return func.HttpResponse(
                json.dumps(response_data),
                headers=headers
            )

        # Existing session
        logging.info(f'Processing message for session {session_id}')
        state = sessions.get(session_id)
        if not state:
            logging.error(f'Invalid session ID: {session_id}')
            return func.HttpResponse(
                json.dumps({"error": "Invalid session"}),
                headers=headers,
                status_code=400
            )
            
        # Process message through bot
        try:
            state["user_message"] = message
            old_state = state.copy()
            state = bot.graph.invoke(state)
            sessions[session_id] = state
            
            response_data = {
                "question": state["current_question"],
                "complete": state.get("interview_complete", False)
            }
            logging.info(f'Response data: {response_data}')
            
            return func.HttpResponse(
                json.dumps(response_data),
                headers=headers
            )
            
        except Exception as e:
            logging.error(f'Error processing message: {str(e)}', exc_info=True)
            return func.HttpResponse(
                json.dumps({"error": f"Error processing message: {str(e)}"}),
                headers=headers,
                status_code=500
            )
        
    except Exception as e:
        logging.error(f'Unexpected error: {str(e)}', exc_info=True)
        return func.HttpResponse(
            json.dumps({"error": str(e)}),
            headers=headers,
            status_code=500
        ) 