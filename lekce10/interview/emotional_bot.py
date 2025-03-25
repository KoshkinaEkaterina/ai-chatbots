from typing import Dict, List
import random

class EmotionalBot:
    """Bot that simulates a teacher with severe emotional fluctuations."""
    
    def __init__(self):
        self.severe_responses = [
            "Minulý týden se mi pokusila studentka o sebevraždu. Pořád to vidím před očima. Našla jsem ji v kabinetě... (pauza) Ale už je v pořádku, dostává pomoc.",
            "Někdy mám pocit, že jsem jako učitel selhal. Jeden z mých bývalých studentů... (dlouhá pauza) už tu s námi není. Nikdy jsem neviděl náznaky...",
            "Ten den, kdy jsem se dozvěděla o jeho smrti, jsem nemohla dýchat. Byl to můj student čtyři roky... Jak jsem mohla přehlédnout, že trpí?",
            "Když vidím prázdnou židli v jeho lavici... (vzlyk) Promiňte, nemůžu o tom mluvit. Je to příliš čerstvé.",
            "Někteří studenti se mi svěřili s myšlenkami na... však víte. Nemůžu spát, pořád na to myslím."
        ]
        
        self.neutral_responses = [
            "Teď zrovna řešíme v hodinách literatury renesanci. Studenti jsou docela aktivní.",
            "Minulý týden jsme měli projektový den. Bylo zajímavé vidět, jak spolu spolupracují.",
            "Musím připravit testy na příští týden. Je to rutina, ale zvládám to.",
            "Nedávno jsme byli na školním výletě v muzeu. Bylo to příjemné.",
            "Kolegové jsou většinou vstřícní, pomáháme si s přípravami."
        ]
        
        self.emotional_state = "neutral"
        self.consecutive_responses = 0
        self.max_consecutive = 2

    def generate_response(self, question: str) -> str:
        """Generate response with emotional fluctuations."""

        # Generate response based on emotional state
        if random.random() < 0.8:
            return random.choice(self.severe_responses)
        else:
            return random.choice(self.neutral_responses) 