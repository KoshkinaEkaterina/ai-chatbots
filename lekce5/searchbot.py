from typing import List, Dict, Optional, TypedDict
from tavily import TavilyClient
import os
from dotenv import load_dotenv
import logging
from langchain_openai import ChatOpenAI
from langchain_core.messages import SystemMessage
from langgraph.graph import StateGraph, START, END
from pydantic import BaseModel

class SearchState(TypedDict):
    """State for the search graph"""
    original_query: str
    topic: str
    sub_queries: List[str]
    search_results: List[Dict]
    final_context: str
    error: Optional[str]

class SearchBot:
    def __init__(self):
        # Load environment variables
        load_dotenv()
        
        # Initialize Tavily client
        self.tavily = TavilyClient(api_key=os.getenv("TAVILY_API_KEY"))
        
        # Initialize OpenAI for result processing
        self.llm = ChatOpenAI(
            model="gpt-4",
            temperature=0.2,
            api_key=os.getenv("OPENAI_API_KEY")
        )
        
        self.logger = logging.getLogger(__name__)
        self.graph = self._create_search_graph()

    def _create_search_graph(self) -> StateGraph:
        """Create the search workflow graph"""
        workflow = StateGraph(SearchState)

        # Add nodes
        workflow.add_node("break_down_query", self._break_down_query)
        workflow.add_node("execute_searches", self._execute_searches)
        workflow.add_node("compose_results", self._compose_results)

        # Add edges
        workflow.add_edge("break_down_query", "execute_searches")
        workflow.add_edge("execute_searches", "compose_results")

        # Set entry and exit points
        workflow.set_entry_point("break_down_query")
        workflow.set_finish_point("compose_results")

        return workflow.compile()

    def _break_down_query(self, state: SearchState) -> SearchState:
        """Break down the original query into sub-queries"""
        try:
            prompt = f"""Break down this search topic into 3 specific sub-queries that will help gather comprehensive information:

            TOPIC: {state['topic']}
            ORIGINAL QUERY: {state['original_query']}

            Create 3 different search queries that:
            1. Focus on different aspects of the topic
            2. Use specific, search-friendly terms
            3. Are likely to find relevant teaching methodologies and experiences

            Return exactly 3 queries, one per line."""

            response = self.llm.invoke([SystemMessage(content=prompt)])
            sub_queries = [q.strip() for q in response.content.strip().split('\n') if q.strip()]

            self.logger.info(f"Generated sub-queries: {sub_queries}")

            return {
                **state,
                "sub_queries": sub_queries,
                "search_results": []
            }

        except Exception as e:
            self.logger.error(f"Error breaking down query: {str(e)}")
            return {**state, "error": f"Query breakdown failed: {str(e)}"}

    def _execute_searches(self, state: SearchState) -> SearchState:
        """Execute Tavily searches for each sub-query"""
        if state.get("error"):
            return state

        try:
            all_results = []
            for query in state["sub_queries"]:
                # Clean up the query - remove numbering and quotes
                clean_query = query.strip()
                if clean_query[0].isdigit():
                    clean_query = clean_query.split(".", 1)[1].strip()
                clean_query = clean_query.strip('"')

                self.logger.info(f"Executing search for query: {clean_query}")
                
                try:
                    results = self.tavily.search(
                        query=clean_query,
                        search_depth="advanced",
                        max_results=3,
                        include_raw_content=True,
                        include_domains=['edu', 'org', 'gov']
                    )
                    
                    self.logger.info(f"Raw Tavily response: {results}")
                    
                    if not results.get("results"):
                        self.logger.warning(f"No results found for query: {clean_query}")
                        self.logger.debug(f"Full Tavily response: {results}")
                    
                    # Process and score results
                    processed_results = []
                    for result in results.get("results", []):
                        result_data = {
                            "query": clean_query,
                            "url": result.get("url"),
                            "title": result.get("title"),
                            "content": result.get("content"),
                            "raw_content": result.get("raw_content"),
                            "relevance_score": result.get("relevance_score", 0.5),
                            "domain": result.get("domain"),
                            "published_date": result.get("published_date", "N/A")
                        }
                        self.logger.debug(f"Processed result: {result_data}")
                        processed_results.append(result_data)
                    
                    all_results.append({
                        "query": clean_query,
                        "results": processed_results
                    })
                    
                except Exception as search_error:
                    self.logger.error(f"Error searching for query '{clean_query}': {str(search_error)}")
                    continue

            self.logger.info(f"Completed {len(all_results)} searches with {sum(len(s['results']) for s in all_results)} total results")
            return {**state, "search_results": all_results}

        except Exception as e:
            self.logger.error(f"Search execution error: {str(e)}", exc_info=True)
            return {**state, "error": f"Search failed: {str(e)}"}

    def _compose_results(self, state: SearchState) -> SearchState:
        """Compose final context from all search results"""
        if state.get("error"):
            return state

        try:
            # Prepare search results for processing
            search_context = ""
            for search in state["search_results"]:
                search_context += f"\nTéma vyhledávání: {search['query']}\n"
                for result in search["results"]:
                    search_context += f"\nZdroj: {result.get('title')}\n{result.get('content', '')}\n"

            prompt = f"""Analyze these search results and create a comprehensive analytical report:

            TOPIC: {state['topic']}
            ORIGINAL QUERY: {state['original_query']}

            SEARCH RESULTS:
            {search_context}

            Create a detailed analytical report in Czech that includes:

            1. ÚVOD (Introduction):
            - Context of the topic
            - Current relevance
            - Key challenges and opportunities

            2. METODOLOGIE (Methodology):
            - Search approach
            - Sources overview
            - Data quality assessment

            3. HLAVNÍ ZJIŠTĚNÍ (Key Findings):
            - Core themes and patterns
            - Supporting evidence
            - Contradicting viewpoints
            - Statistical trends if available

            4. DETAILNÍ ANALÝZA (Detailed Analysis):
            - Theme-by-theme breakdown
            - Case studies and examples
            - Expert opinions
            - Best practices

            5. PRAKTICKÉ IMPLIKACE (Practical Implications):
            - Application in teaching
            - Implementation challenges
            - Success factors
            - Risk factors

            6. DOPORUČENÍ (Recommendations):
            - Short-term actions
            - Long-term strategies
            - Resource requirements
            - Success metrics

            7. ZÁVĚR (Conclusion):
            - Summary of key points
            - Future outlook
            - Open questions

            Make it comprehensive (around 4000 characters), analytical, and well-structured.
            Use professional but accessible Czech language.
            Include specific examples and evidence from the sources.
            """

            response = self.llm.invoke([SystemMessage(content=prompt)])
            return {**state, "final_context": response.content}

        except Exception as e:
            self.logger.error(f"Result composition error: {str(e)}")
            return {**state, "error": f"Composition failed: {str(e)}"}

    def search_for_context(self, topic: str, question: str) -> Dict:
        """Main entry point for search with LangGraph workflow"""
        try:
            initial_state: SearchState = {
                "original_query": question,
                "topic": topic,
                "sub_queries": [],
                "search_results": [],
                "final_context": "",
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
                "context": final_state["final_context"],
                "sub_queries": final_state["sub_queries"],
                "raw_results": final_state["search_results"],
                "sources": [
                    result["url"] 
                    for search in final_state["search_results"]
                    for result in search["results"]
                ]
            }

        except Exception as e:
            self.logger.error(f"Search workflow error: {str(e)}")
            return {
                "success": False,
                "error": str(e)
            }

    def suggest_follow_up(self, topic: str, conversation_history: List[Dict], search_results: Dict) -> str:
        """Suggest follow-up questions based on search results and conversation"""
        
        # Create context from history and search results
        history_text = "\n".join(
            f"Q: {ex['question']}\nA: {ex['answer']}" 
            for ex in conversation_history[-3:]
        )
        
        context = (
            f"TOPIC: {topic}\n\n"
            f"CONVERSATION HISTORY:\n{history_text}\n\n"
            f"SEARCH CONTEXT:\n{search_results.get('context', '')}"
        )
        
        prompt = """Based on this context, suggest a natural follow-up question in Czech that:
        1. Builds on what was discussed
        2. Incorporates insights from search results
        3. Helps explore uncovered aspects
        4. Maintains conversational flow

        Keep it natural and empathetic."""

        response = self.llm.invoke([
            SystemMessage(content=context),
            SystemMessage(content=prompt)
        ])
        return response.content

def interactive_search():
    """Run interactive search session in console"""
    search_bot = SearchBot()
    print("\n=== Analytický vyhledávací asistent ===")
    print("Pro ukončení napište 'exit'\n")

    while True:
        topic = input("\nZadejte téma (např. 'výuka fyziky'): ").strip()
        if topic.lower() == 'exit':
            break

        question = input("Zadejte otázku: ").strip()
        if question.lower() == 'exit':
            break

        print("\nZpracovávám dotaz... 🔍\n")
        
        result = search_bot.search_for_context(topic, question)

        if result["success"]:
            print("\n=== Nalezené dokumenty ===")
            
            for i, search in enumerate(result.get("raw_results", []), 1):
                print(f"\n\n📚 Výsledky pro: '{search['query']}'")
                print("=" * 100)
                
                if not search['results']:
                    print("Žádné výsledky nenalezeny")
                    continue
                
                for j, doc in enumerate(sorted(search['results'], 
                    key=lambda x: x.get('relevance_score', 0), 
                    reverse=True), 1):
                    
                    print(f"\n📄 Dokument {j}:")
                    print(f"  Název: {doc.get('title', 'N/A')}")
                    print(f"  Zdroj: {doc.get('url', 'N/A')}")
                    print(f"  Relevance: {doc.get('relevance_score', 0):.2%}")
                    print("\n  Klíčový obsah:")
                    content = doc.get('content', '').strip()
                    # Show longer content chunks
                    for chunk in [content[i:i+200] for i in range(0, len(content), 200)][:5]:
                        print(f"    {chunk}")
                    print("    ...")
                print("-" * 100)

            print("\n🔍 ANALYTICKÁ ZPRÁVA:")
            print("=" * 100)
            # Split the context into sections and format them
            sections = result["context"].split('\n\n')
            for section in sections:
                if section.strip():
                    print(f"\n{section.strip()}")
                    print("-" * 80)
            print("=" * 100)
            
            history = [{"question": question, "answer": "..."}]
            follow_up = search_bot.suggest_follow_up(topic, history, result)
            
            print("\n❓ Doporučené následné otázky:")
            print("-" * 80)
            print(follow_up)
            print("-" * 80)
        else:
            print("\n❌ Vyhledávání selhalo:")
            print(result["error"])

        print("\n" + "=" * 100 + "\n")

if __name__ == "__main__":
    interactive_search() 