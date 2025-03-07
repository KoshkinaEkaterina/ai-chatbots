from typing import List, Dict, Optional, TypedDict
import os
from dotenv import load_dotenv
import logging
from langchain_openai import ChatOpenAI
from langchain_core.messages import SystemMessage
from langgraph.graph import StateGraph, START, END
import requests
from datetime import datetime, timedelta, time
import json
from pydantic import BaseModel, Field, EmailStr
import pytz
from zoneinfo import ZoneInfo
import parsedatetime
from dateutil import parser
from dateutil.relativedelta import relativedelta

class WorldState:
    """Tool for providing current world context"""
    def __init__(self):
        self.cal = parsedatetime.Calendar()
        self.local_tz = ZoneInfo("Europe/Prague")
    
    def get_current_context(self) -> Dict:
        """Get current date/time context"""
        now = datetime.now(self.local_tz)
        return {
            "current_time": now.isoformat(),
            "current_date": now.date().isoformat(),
            "day_of_week": now.strftime("%A"),
            "is_weekend": now.weekday() >= 5,
            "timezone": "Europe/Prague",
        }
    
    def parse_relative_datetime(self, text: str) -> Dict:
        """Parse relative date/time expressions"""
        now = datetime.now(self.local_tz)
        
        # Try parsedatetime first
        struct, status = self.cal.parse(text)
        if status > 0:
            dt = datetime(*struct[:6], tzinfo=self.local_tz)
            return {
                "datetime": dt.isoformat(),
                "is_relative": True,
                "original_text": text
            }
        
        # Fallback to dateutil parser
        try:
            dt = parser.parse(text, default=now)
            if dt.tzinfo is None:
                dt = dt.replace(tzinfo=self.local_tz)
            return {
                "datetime": dt.isoformat(),
                "is_relative": False,
                "original_text": text
            }
        except:
            return None

class ZapierState(TypedDict):
    """State for the Zapier workflow"""
    original_request: str
    world_context: Dict
    parsed_datetime: Optional[Dict]
    calendar_data: Dict
    action_results: List[Dict]
    final_response: str
    error: Optional[str]

class CalendarEventTime(BaseModel):
    """Model for event time parsing"""
    date: datetime  # Changed from datetime to handle both date and time
    duration_minutes: int = Field(default=60, ge=30, le=480)
    is_all_day: bool = False

class CalendarEventData(BaseModel):
    """Model for calendar event data"""
    summary: str = Field(..., min_length=1, max_length=200)
    description: Optional[str] = Field(default="", max_length=1000)
    location: Optional[str] = Field(default=None, max_length=200)
    start_time: datetime  # Changed to use datetime directly
    duration_minutes: int = Field(default=60, ge=30, le=480)
    attendees: List[str] = Field(default_factory=list)

    def to_google_format(self, timezone: str = "Europe/Prague") -> dict:
        """Convert to Google Calendar API format"""
        # Calculate end time
        end_time = self.start_time + timedelta(minutes=self.duration_minutes)

        event = {
            "summary": self.summary or "New Meeting",
            "start": {
                "dateTime": self.start_time.isoformat(),
                "timeZone": timezone
            },
            "end": {
                "dateTime": end_time.isoformat(),
                "timeZone": timezone
            }
        }

        if self.description:
            event["description"] = self.description
        if self.location:
            event["location"] = self.location
        if self.attendees:
            event["attendees"] = [{"email": email} for email in self.attendees]

        return event

class ZapierBot:
    def __init__(self):
        load_dotenv()
        
        self.calendar_webhook = os.getenv("ZAPIER_CALENDAR_WEBHOOK")
        if not self.calendar_webhook:
            raise ValueError("ZAPIER_CALENDAR_WEBHOOK not set in environment")
        
        self.llm = ChatOpenAI(
            model="gpt-4",
            temperature=0.2,
            api_key=os.getenv("OPENAI_API_KEY")
        )
        
        self.world = WorldState()
        self.logger = logging.getLogger(__name__)
        self.graph = self._create_workflow_graph()

    def _create_workflow_graph(self) -> StateGraph:
        """Create the calendar workflow graph"""
        workflow = StateGraph(ZapierState)

        # Add nodes
        workflow.add_node("get_context", self._get_world_context)
        workflow.add_node("parse_datetime", self._parse_datetime)
        workflow.add_node("extract_calendar_data", self._extract_calendar_data)
        workflow.add_node("create_calendar_event", self._create_calendar_event)
        workflow.add_node("compose_response", self._compose_response)

        # Add edges
        workflow.add_edge("get_context", "parse_datetime")
        workflow.add_edge("parse_datetime", "extract_calendar_data")
        workflow.add_edge("extract_calendar_data", "create_calendar_event")
        workflow.add_edge("create_calendar_event", "compose_response")

        workflow.set_entry_point("get_context")
        workflow.set_finish_point("compose_response")

        return workflow.compile()

    def _get_world_context(self, state: ZapierState) -> ZapierState:
        """Get current world context"""
        try:
            context = self.world.get_current_context()
            return {**state, "world_context": context}
        except Exception as e:
            self.logger.error(f"Context error: {str(e)}")
            return {**state, "error": f"Context error: {str(e)}"}

    def _parse_datetime(self, state: ZapierState) -> ZapierState:
        """Parse date/time from request"""
        if state.get("error"):
            return state

        try:
            prompt = f"""Extract the time-related information from this request.
            Current context: {state['world_context']}
            
            REQUEST: {state['original_request']}
            
            Extract:
            1. When the event starts
            2. How long it lasts (if specified)
            3. Any recurring pattern (if specified)

            Return as natural language description."""

            response = self.llm.invoke([SystemMessage(content=prompt)])
            datetime_text = response.content.strip()
            
            parsed = self.world.parse_relative_datetime(datetime_text)
            if not parsed:
                return {**state, "error": "Could not parse date/time from request"}

            return {**state, "parsed_datetime": parsed}

        except Exception as e:
            self.logger.error(f"DateTime parsing error: {str(e)}")
            return {**state, "error": f"DateTime parsing error: {str(e)}"}

    def _extract_calendar_data(self, state: ZapierState) -> ZapierState:
        """Extract calendar event details"""
        if state.get("error"):
            return state

        try:
            prompt = f"""Extract calendar event details from this request. Current time context: {state['world_context']}

            REQUEST: {state['original_request']}

            Parse and return EXACTLY this JSON structure:
            {{
                "summary": "Meeting title",
                "description": "Meeting description",
                "time_info": {{
                    "date": "2024-03-10",  # Use YYYY-MM-DD format
                    "time": "14:00",       # Use 24-hour HH:MM format
                    "duration_minutes": 180,
                    "is_all_day": false
                }},
                "location": "",
                "attendees": ["example@email.com"]
            }}
            """

            # Debug the LLM response
            response = self.llm.invoke([SystemMessage(content=prompt)])
            print("\n🔍 DEBUG: LLM Response")
            print("-" * 60)
            print(response.content)
            print("-" * 60)

            extracted = json.loads(response.content)

            # Parse time information
            time_info = extracted.get("time_info", {})
            date_str = time_info["date"]
            time_str = time_info["time"]
            
            # Combine date and time into datetime
            start_datetime = datetime.strptime(
                f"{date_str} {time_str}", 
                "%Y-%m-%d %H:%M"
            ).replace(tzinfo=ZoneInfo("Europe/Prague"))

            # Debug parsed values
            print("\n🔍 DEBUG: Parsed Time Values")
            print(f"Start DateTime: {start_datetime}")
            print(f"Duration: {time_info.get('duration_minutes')} minutes")

            # Create event data model
            event_data = CalendarEventData(
                summary=str(extracted.get("summary", "New Meeting")),
                description=str(extracted.get("description", "")),
                location=str(extracted.get("location", "")),
                start_time=start_datetime,
                duration_minutes=int(time_info.get("duration_minutes", 60)),
                attendees=list(extracted.get("attendees", []))
            )

            # Convert to Google Calendar format
            calendar_data = event_data.to_google_format()

            # Debug final output
            print("\n🔍 DEBUG: Final Calendar Data")
            print("-" * 60)
            print(json.dumps(calendar_data, indent=2))
            print("-" * 60)

            return {**state, "calendar_data": calendar_data}

        except Exception as e:
            self.logger.error(f"Calendar data extraction error: {str(e)}")
            print(f"\n🔍 DEBUG: Full error details")
            import traceback
            print(traceback.format_exc())
            return {**state, "error": f"Calendar data error: {str(e)}"}

    def _create_calendar_event(self, state: ZapierState) -> ZapierState:
        """Create calendar event via Zapier"""
        if state.get("error"):
            return state

        try:
            calendar_data = state["calendar_data"]
            
            # Debug: Print request data
            print("\n🔍 DEBUG: Calendar Event Request")
            print("-" * 60)
            print(f"Webhook URL: {self.calendar_webhook}")
            print("\nPayload:")
            print(json.dumps(calendar_data, indent=2))
            print("-" * 60)

            response = requests.post(
                self.calendar_webhook,
                json=calendar_data,
                headers={"Content-Type": "application/json"}
            )
            
            # Debug: Print response
            print("\n📡 DEBUG: Zapier Response")
            print("-" * 60)
            print(f"Status Code: {response.status_code}")
            print("\nResponse Headers:")
            print(json.dumps(dict(response.headers), indent=2))
            print("\nResponse Body:")
            try:
                print(json.dumps(response.json(), indent=2))
            except:
                print(response.text)
            print("-" * 60)
            
            if response.status_code != 200:
                error_msg = f"Calendar creation failed (Status {response.status_code}): {response.text}"
                self.logger.error(error_msg)
                return {**state, "error": error_msg}

            return {
                **state,
                "action_results": [{
                    "action": "create_calendar_event",
                    "status": "success",
                    "response": response.json() if response.text else {}
                }]
            }

        except Exception as e:
            error_msg = f"Calendar creation error: {str(e)}"
            self.logger.error(error_msg, exc_info=True)
            return {**state, "error": error_msg}

    def _compose_response(self, state: ZapierState) -> ZapierState:
        """Compose user-friendly response"""
        if state.get("error"):
            return {**state, "final_response": f"Error: {state['error']}"}

        try:
            data = state["calendar_data"]
            start = parser.parse(data["start"]["dateTime"])
            end = parser.parse(data["end"]["dateTime"])
            
            response = (
                f"✅ Vytvořil jsem událost v kalendáři:\n\n"
                f"📋 {data['summary']}\n"
                f"🕒 {start.strftime('%d.%m.%Y %H:%M')} - {end.strftime('%H:%M')}\n"
            )
            
            if data.get("location"):
                response += f"📍 {data['location']}\n"
            if data.get("description"):
                response += f"📝 {data['description']}\n"
            if data.get("attendees"):
                response += f"👥 Účastníci: {', '.join([attendee['email'] for attendee in data['attendees']])}\n"

            return {**state, "final_response": response}

        except Exception as e:
            self.logger.error(f"Response composition error: {str(e)}")
            return {**state, "error": f"Response error: {str(e)}"}

    def process_request(self, request: str) -> Dict:
        """Process calendar event request"""
        try:
            initial_state: ZapierState = {
                "original_request": request,
                "world_context": {},
                "parsed_datetime": None,
                "calendar_data": {},
                "action_results": [],
                "final_response": "",
                "error": None
            }

            final_state = self.graph.invoke(initial_state)

            if final_state.get("error"):
                return {
                    "success": False,
                    "error": final_state["error"]
                }

            return {
                "success": True,
                "response": final_state["final_response"],
                "details": final_state["calendar_data"]
            }

        except Exception as e:
            self.logger.error(f"Request processing error: {str(e)}")
            return {
                "success": False,
                "error": str(e)
            }

def interactive_zapier():
    """Run interactive calendar assistant"""
    zapier_bot = ZapierBot()
    print("\n=== Kalendářní Asistent ===")
    print("Pro ukončení napište 'exit'\n")
    print("Příklady příkazů:")
    print("- Vytvoř schůzku s Janem zítra v 14:00 na hodinu")
    print("- Naplánuj týmový call příští středu od 10:00 do 11:30")
    print("- Přidej do kalendáře oběd s klientem v pátek ve 12:00\n")

    while True:
        request = input("\nCo si přejete naplánovat? ").strip()
        if request.lower() == 'exit':
            break

        print("\nZpracovávám požadavek... ⚙️\n")
        
        result = zapier_bot.process_request(request)

        if result["success"]:
            print(result["response"])
        else:
            print("\n❌ Chyba:")
            print(result["error"])

        print("\n" + "=" * 80 + "\n")

if __name__ == "__main__":
    interactive_zapier() 