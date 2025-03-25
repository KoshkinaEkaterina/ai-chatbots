from typing import Dict, List, Set, Optional
from state import State, ReferenceState
from datetime import datetime
import random
from langchain_core.messages import SystemMessage

class MicroReferencesModule:
    def __init__(self):
        # Initialize with reference types
        self.reference_types = {
            "people": set(),
            "places": set(),
            "events": set(),
            "emotions": set(),
            "actions": set()
        }

    def extract_references(self, text: str, model) -> Dict[str, Set[str]]:
        """Extract references using LLM instead of spaCy"""
        prompt = f"""Analyze this Czech text and extract key references into categories.
        
        TEXT: {text}
        
        Return a JSON object with these categories:
        {{
            "people": ["list of person names mentioned"],
            "places": ["list of locations mentioned"],
            "events": ["list of events or situations described"],
            "emotions": ["list of emotions expressed or described"],
            "actions": ["list of key actions or verbs mentioned"]
        }}
        
        Extract ONLY items that are explicitly mentioned in the text.
        For Czech names, preserve diacritics.
        """
        
        try:
            result = model.invoke([SystemMessage(content=prompt)])
            # Parse the JSON response
            import json
            references = json.loads(result.content)
            
            # Convert lists to sets
            return {
                "people": set(references.get("people", [])),
                "places": set(references.get("places", [])),
                "events": set(references.get("events", [])),
                "emotions": set(references.get("emotions", [])),
                "actions": set(references.get("actions", []))
            }
        except Exception as e:
            # Fallback to empty sets if parsing fails
            return {key: set() for key in self.reference_types}

    def add_to_reference_bank(self, state: State, references: Dict[str, Set[str]]) -> None:
        for ref_type, items in references.items():
            if ref_type not in state.humanity.reference.reference_bank:
                state.humanity.reference.reference_bank[ref_type] = []
            
            for item in items:
                entry = {
                    "text": item,
                    "timestamp": datetime.now(),
                    "context": state.current_question,
                    "type": ref_type
                }
                state.humanity.reference.reference_bank[ref_type].append(entry)

    def generate_reference(self, state: State) -> Optional[str]:
        """Generate a contextual reference to use in the next question."""
        if not state.humanity.reference.reference_bank:
            return None
            
        # Prioritize recent and relevant references
        recent_refs = []
        for ref_type, refs in state.humanity.reference.reference_bank.items():
            if refs:
                recent_refs.extend(refs[-3:])  # Get last 3 references of each type
                
        if not recent_refs:
            return None
            
        # Select a reference that fits the current context
        selected_ref = random.choice(recent_refs)
        return selected_ref["text"]

    def update_state(self, state: State, response: str) -> State:
        # Get the model from the state (assuming it's available)
        model = state.get("model")  # You'll need to ensure this is available
        if not model:
            return state
            
        # Extract new references using LLM
        references = self.extract_references(response, model)
        
        # Update named entities
        for entity_type, entities in references.items():
            if entity_type not in state.humanity.reference.named_entities:
                state.humanity.reference.named_entities[entity_type] = []
            state.humanity.reference.named_entities[entity_type].extend(entities)
        
        # Add to reference bank
        self.add_to_reference_bank(state, references)
        
        # Update key moments if significant
        if state.humanity.emotional.emotional_weight > 0.7:
            state.humanity.reference.key_moments.append({
                "text": response,
                "timestamp": datetime.now(),
                "emotional_weight": state.humanity.emotional.emotional_weight
            })
        
        return state 