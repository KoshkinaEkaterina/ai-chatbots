import azure.functions as func
import logging
from ..interview_bot import InterviewBot
from twilio.twiml.messaging_response import MessagingResponse
from urllib.parse import parse_qs
from dotenv import load_dotenv

# Initialize environment and bot
load_dotenv()
bot = InterviewBot()

def main(req: func.HttpRequest) -> func.HttpResponse:
    """Handle incoming WhatsApp messages"""
    try:
        # Debug logging
        logging.debug("Received webhook request")
        logging.debug(f"Headers: {dict(req.headers)}")
        
        # Get raw body and parse it
        body = req.get_body().decode('utf-8')
        logging.debug(f"Raw body: {body}")
        
        # Parse form data manually
        form_data = parse_qs(body)
        logging.debug(f"Parsed form data: {form_data}")
        
        # Extract message and phone number
        message = form_data.get('Body', [''])[0].strip()
        from_number = form_data.get('From', [''])[0].replace('whatsapp:', '')
        
        logging.debug(f"Message from {from_number}: {message}")
        
        # Get ChatGPT response
        result = bot.chat(message)
        
        # Create TwiML response
        twiml = MessagingResponse()
        twiml.message(result["response"])
        
        return func.HttpResponse(
            str(twiml),
            mimetype="application/xml"
        )
        
    except Exception as e:
        logging.error(f"Error processing WhatsApp message: {str(e)}", exc_info=True)
        twiml = MessagingResponse()
        twiml.message("🤖 Beep boop... I encountered an error. Please try again!")
        return func.HttpResponse(
            str(twiml),
            mimetype="application/xml",
            status_code=500
        ) 