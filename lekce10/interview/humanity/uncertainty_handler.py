from typing import Dict, Optional
from state import State
import random

class UncertaintyHandler:
    def __init__(self):
        self.uncertainty_patterns = [
            r"nevím",
            r"netuším",
            r"nejsem si jistý/á",
            r"těžko říct",
            r"nejsem psycholog",
            r"neumím posoudit"
        ]
        
        self.reframing_strategies = {
            "personal_experience": [
                "Chápu, že je to těžké posoudit. Můžete mi říct, jak jste to vnímal/a vy osobně?",
                "Co jste si v té situaci myslel/a vy?",
                "Jak jste se v té situaci cítil/a?"
            ],
            "concrete_examples": [
                "Můžete mi třeba popsat konkrétní situaci, kterou si pamatujete?",
                "Vzpomenete si na nějaký příklad z vaší zkušenosti?"
            ],
            "different_perspective": [
                "Zkusme se na to podívat z jiné strany - co vám v té situaci pomohlo?",
                "Co bylo pro vás v té situaci nejdůležitější?"
            ]
        }

    def handle_uncertainty(self, response: str, state: State) -> Optional[str]:
        if any(re.search(pattern, response.lower()) for pattern in self.uncertainty_patterns):
            # Choose appropriate reframing strategy based on context
            if "nejsem psycholog" in response.lower():
                return random.choice(self.reframing_strategies["personal_experience"])
            elif "nevím" in response.lower():
                return random.choice(self.reframing_strategies["concrete_examples"])
            else:
                return random.choice(self.reframing_strategies["different_perspective"])
        return None 