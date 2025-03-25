    def print_topic_status(self, topic: Topic):
        """Print detailed status of topic coverage and insights."""
        print(f"\n{'='*100}")
        print(f"DETAILNÍ ANALÝZA ODPOVĚDI PRO TÉMA: {topic.question}")
        print(f"{'='*100}")
        
        for factor, description in topic.factors.items():
            print(f"\n{'='*50}")
            print(f"FAKTOR: {factor}")
            print(f"POPIS: {description}")
            print(f"{'='*50}")
            
            if factor in topic.factor_insights and topic.factor_insights[factor]:
                print("\nNALEZENÉ INFORMACE:")
                for insight in topic.factor_insights[factor]:
                    print(f"\n• DETAIL: {insight['key_info']}")
                    if 'evidence' in insight:
                        print(f"  DŮKAZ: {insight['evidence']}")
                    if 'quote' in insight:
                        print(f"  CITACE: \"{insight['quote']}\"")
                    print(f"  RELEVANCE: {insight.get('score', 0.0):.2f}")
                
                print(f"\nCELKOVÉ POKRYTÍ: {topic.covered_factors.get(factor, 0.0):.2f}")
            else:
                print("\nŽÁDNÉ INFORMACE NEBYLY NALEZENY")
                print(f"POKRYTÍ: {topic.covered_factors.get(factor, 0.0):.2f}")
            
            print(f"\n{'-'*50}")
        
        print(f"\n{'='*100}")


    def print_topic_summary(self, topic: Topic, file_path: str = "interview_analysis.txt"):
        """Create a detailed summary of all insights gathered for each factor in the topic."""
        # First print to console
        print(f"\n{'='*100}")
        print(f"SOUHRNNÁ ANALÝZA TÉMATU: {topic.question}")
        print(f"{'='*100}\n")
        
        for factor, description in topic.factors.items():
            print(f"\nFAKTOR: {factor}")
            print(f"POPIS: {description}")
            print(f"CELKOVÉ POKRYTÍ: {topic.covered_factors.get(factor, 0.0):.2f}")
            
            if factor in topic.factor_insights and topic.factor_insights[factor]:
                print("\nVŠECHNA ZJIŠTĚNÍ:")
                for insight in topic.factor_insights[factor]:
                    print(f"\n• INFORMACE: {insight['key_info']}")
                    print(f"  CITACE: \"{insight['quote']}\"")
                    print(f"  RELEVANCE: {insight.get('score', 0.0):.2f}")
                
                # Generate an overall summary
                summary_prompt = f"""Create a concise summary of these insights about {factor}:
                {chr(10).join(f'- {i["key_info"]}' for i in topic.factor_insights[factor])}
                
                Return a 2-3 sentence summary in Czech."""
                
                summary = self.model.invoke([SystemMessage(content=summary_prompt)])
                print(f"\nSOUHRN FAKTORU:\n{summary.content}")
            else:
                print("\nŽÁDNÉ INFORMACE NEBYLY ZÍSKÁNY")
            
            print(f"\n{'-'*50}")


def print_brief_status(self, old_state: State, answer: str, next_question: str):
        """Print status with emotional awareness."""
        current_topic = old_state["topics"][old_state["current_topic_id"]]
        
        # Check for emotional content
        emotional_analysis = self.analyze_emotional_content(answer)
        
        # If the response was emotionally significant, acknowledge before analysis
        if emotional_analysis["emotional_weight"] > 0.6:
            print("\n" + "-"*50)
            print("EMOČNÍ KONTEXT:")
            print("Učitel sdílel velmi citlivou zkušenost. Dejme prostor pro zpracování...")
            print("-"*50 + "\n")
        
        # Only print analysis and next question (removed answer printing)
        print("\nANALÝZA:")
        covered = {f: s for f, s in current_topic.covered_factors.items() if s > 0}
        if covered:
            for factor, score in covered.items():
                print(f"✓ {factor}: {score:.2f}")
        else:
            print("❌ Odpověď neposkytla žádné relevantní informace k tématu.")
        
        print("\nDALŠÍ OTÁZKA:")
        print(next_question)
        print("\nZDŮVODNĚNÍ:")
        if not covered:
            print("Předchozí odpověď byla mimo téma. Zkusíme otázku položit jinak.")
        else:
            uncovered = [f for f, s in current_topic.covered_factors.items() if s < 0.7]
            if uncovered:
                print(f"Potřebujeme více informací o: {', '.join(uncovered)}")
            else:
                print("Přecházíme k dalšímu tématu.")