from typing import Dict
import logging

class TeacherBot:
    def __init__(self):
        # Set up logging
        logging.basicConfig(level=logging.DEBUG)
        self.logger = logging.getLogger('TeacherBot')
        
        # Core knowledge from the dialogue
        self.context = {
            "student": {
                "gender": "female",
                "behavior": "self-harm",
                "signs": [
                    "wears long sleeves almost always",
                    "avoids PE class",
                    "claims to be sick often",
                    "quiet personality",
                    "doesn't seek contact",
                    "visible cuts on arms (trying to hide them)"
                ],
                "social": [
                    "generally quiet",
                    "doesn't talk much to anyone",
                    "other students don't pay much attention to her",
                    "no conflicts with others"
                ]
            },
            "teacher": {
                "approach": [
                    "discussing with colleagues",
                    "planning to contact parents",
                    "unsure about proper response",
                    "emphasizes being teacher, not psychologist"
                ],
                "planned_actions": [
                    "contact parents",
                    "discuss self-harm with parents",
                    "find ways to help effectively"
                ],
                "class_impact": "minimal, isolated issue with one student"
            }
        }

    def generate_response(self, question: str) -> str:
        """Generate contextually appropriate response based on the teacher's perspective."""
        
        # Convert question to lowercase for easier matching
        question_lower = question.lower()
        
        # Initial response about the situation
        if "situaci" in question_lower or "problém" in question_lower:
            return "Aktuálně řešíme, že se jedna s dívek sebepoškozuje, snaží se to skrývat, ale občas je vidět, že má pořezané ruce.."
        
        # Define response patterns based on the dialogue
        if "spouštěč" in question_lower or "proč" in question_lower:
            return "To vůbec netuším"
            
        if "tělocvik" in question_lower or "sport" in question_lower:
            return "Tělocviku se většinou vyhývá s tím, že je nachlazená"
            
        if "konflikty" in question_lower or "problémy" in question_lower:
            return "Nene, nic takového, celkově je spíš tišší a úplně kontakty nevyhledává"
            
        if "ostatní" in question_lower or "spolužáci" in question_lower:
            return "Ostatní si ji spíš taky nevšímají"
            
        if "atmosféra" in question_lower or "třída" in question_lower:
            return "Ne, to se spíš neděje"
            
        if "reagovat" in question_lower or "řešit" in question_lower:
            return "Nevím, jak na to mám reagovat, nejsem psycholog ale učitel"
            
        if "postup" in question_lower or "plán" in question_lower:
            return "Probírám to s kolegy a snažím se sbírat jejich zkušenosti. Aktuálně se shodujeme, že se máma spojit s rodiči"
            
        if "rodiče" in question_lower:
            if "plán" in question_lower:
                return "Chceme s nimi probrat, že se jejich dcera sebepoškozuje a jak je možné jejich dceři efektivně pomoci"
            else:
                return "Zatím nijak, teprve to máme v plánu"

        if "rukávy" in question_lower or "oblečení" in question_lower:
            return "To nedokážu říct, obecně nosí dlouhé rukávy téměř pořád.."

        if "mluví" in question_lower or "komunikuje" in question_lower:
            return "Celkově moc s nikým nemluví, je těžké to hodnotit"

        if "vliv" in question_lower or "dopad" in question_lower:
            return "Nijak doufám"

        # Default response for unmatched questions
        return "To nedokážu říct, celkově je těžké to hodnotit"

    def chat(self, message: str) -> Dict:
        """Process chat message and return response."""
        try:
            response = self.generate_response(message)
            self.logger.debug(f"Generated response for: {message}")
            return {
                "response": response
            }
        except Exception as e:
            self.logger.error(f"Error generating response: {str(e)}")
            return {
                "response": "Omlouvám se, ale teď nedokážu odpovědět."
            } 