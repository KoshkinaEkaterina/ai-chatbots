import logging
import json
import azure.functions as func
from urllib.parse import parse_qs

def main(req: func.HttpRequest, twilioMessage: func.Out[str]) -> func.HttpResponse:
    """Test endpoint that sends SMS via Twilio binding"""
    try:
        logging.debug("Test endpoint called")
        logging.debug(f"Headers: {dict(req.headers)}")
        
        # Get raw body and parse it
        body = req.get_body().decode('utf-8')
        logging.debug(f"Raw body: {body}")
        
        # Parse form data
        form_data = parse_qs(body)
        message = form_data.get('Body', ['No message'])[0].strip()
        from_number = form_data.get('From', ['Unknown'])[0]
        
        logging.debug(f"Received message: '{message}' from {from_number}")
        
        # Prepare Twilio message
        value = {
            "body": f"👋 Got your message!\nFrom: {from_number}\nMessage: {message}",
            "to": from_number
        }
        
        # Send via Twilio binding
        twilioMessage.set(json.dumps(value))
        
        return func.HttpResponse(
            "Message sent",
            status_code=200
        )
        
    except Exception as e:
        logging.error(f"Error in test endpoint: {str(e)}", exc_info=True)
        return func.HttpResponse(
            "Error in test endpoint",
            status_code=500
        ) 


