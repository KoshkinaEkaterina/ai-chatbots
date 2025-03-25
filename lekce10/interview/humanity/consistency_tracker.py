from typing import Dict, List, Optional
from state import State, ConsistencyState
import re
from datetime import datetime

class ConsistencyTracker:
    def __init__(self):
        self.contradiction_markers = [
            (r"předtím jsem říkal/a", r"ale teď"),
            (r"vlastně ne", r"spíš"),
            (r"když se zamyslím", r"bylo to jinak")
        ]

    def track_statement(self, topic: str, statement: str, state: State) -> None:
        if topic not in state.humanity.consistency.memory_summary:
            state.humanity.consistency.memory_summary[topic] = []
        
        state.humanity.consistency.memory_summary[topic].append({
            "statement": statement,
            "timestamp": datetime.now(),
            "context": state.current_question
        })

    def check_contradictions(self, response: str, state: State) -> List[Dict]:
        contradictions = []
        
        # Check current response against previous statements
        for topic, statements in state.humanity.consistency.memory_summary.items():
            for statement in statements:
                if self._is_contradicting(response, statement["statement"]):
                    contradictions.append({
                        "topic": topic,
                        "original": statement["statement"],
                        "new": response,
                        "timestamp": datetime.now()
                    })
                    
        return contradictions

    def _is_contradicting(self, new_text: str, old_text: str) -> bool:
        # Add contradiction detection logic
        for start, end in self.contradiction_markers:
            if re.search(f"{start}.*{end}", new_text.lower()):
                return True
        return False

    def generate_clarification_prompt(self, contradiction: Dict) -> str:
        return (
            f"Zmiňoval/a jste dříve, že {contradiction['original']}. "
            f"Teď jste uvedl/a něco jiného. Můžete mi to prosím více objasnit?"
        )

    def update_state(self, state: State, response: str) -> State:
        # Track the new statement
        self.track_statement(
            state.current_topic_id,
            response,
            state
        )
        
        # Check for contradictions
        contradictions = self.check_contradictions(response, state)
        if contradictions:
            state.humanity.consistency.contradictions.extend(contradictions)
            
        return state 