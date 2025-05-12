from langsmith import traceable
from langchain_core.messages import HumanMessage, AIMessage
from langchain_core.tools import tool
from langchain_openai import ChatOpenAI
from typing import List, Dict, Any
import os

# Setup LangSmith tracing
os.environ["LANGCHAIN_TRACING_V2"] = "true"
os.environ["LANGCHAIN_ENDPOINT"] = "https://api.smith.langchain.com"
os.environ["LANGCHAIN_API_KEY"] = ""  
os.environ["LANGCHAIN_PROJECT"] = "reflexion-agent"

class ReflexionAgent:
    def __init__(self, max_iterations=3):
        """Initialize the Reflexion agent with LLM and tools."""
        self.llm = ChatOpenAI(model="gpt-4", temperature=0)
        self.max_iterations = max_iterations
        
        # Register tools
        self.tools = {
            "web_search": self.web_search,
        }
    
    @traceable(name="initial_response")
    def generate_initial_response(self, query: str) -> Dict[str, Any]:
        """Generate the initial response with self-critique and tool suggestions."""
        prompt = f"""
        You are a helpful assistant. Given the user query, provide:
        1. Your initial response to help the user
        2. A self-critique of your response, identifying potential weaknesses or knowledge gaps
        3. Suggested tool queries that could provide more information to improve your response
        
        Format your response as a JSON with the following keys:
        - "answer": Your initial response
        - "reflection": Your self-critique
        - "tool_queries": A list of dictionaries with "tool" and "query" keys
        
        User Query: {query}
        """
        
        response = self.llm.invoke([HumanMessage(content=prompt)])
        
        try:
            # Parse response as JSON (in production, add proper error handling)
            import json
            return json.loads(response.content)
        except:
            # Fallback in case of parsing issues
            return {
                "answer": response.content,
                "reflection": "Unable to generate structured reflection.",
                "tool_queries": []
            }
    
    @tool
    @traceable(name="web_search")
    def web_search(self, query: str) -> str:
        """Simulate a web search tool."""
        # In a real implementation, this would connect to a search API
        return f"Simulated web search results for query: '{query}'"
    
    @traceable(name="execute_tool_queries")
    def execute_tool_queries(self, tool_queries: List[Dict[str, str]]) -> Dict[str, str]:
        """Execute the suggested tool queries and return results."""
        tool_results = {}
        
        for tool_query in tool_queries:
            tool_name = tool_query.get("tool")
            query = tool_query.get("query")
            
            if tool_name in self.tools:
                tool_results[f"{tool_name}:{query}"] = self.tools[tool_name](query)
            
        return tool_results
    
    @traceable(name="revise_response")
    def revise_response(self, 
                        original_query: str, 
                        previous_response: Dict[str, Any], 
                        tool_results: Dict[str, str]) -> Dict[str, Any]:
        """Revise the response based on tool results."""
        prompt = f"""
        You are helping the user with this query: "{original_query}"
        
        Your previous response was:
        {previous_response["answer"]}
        
        Your reflection was:
        {previous_response["reflection"]}
        
        New information from tools:
        {tool_results}
        
        Please provide:
        1. An updated response that incorporates this new information
        2. A new self-reflection on remaining limitations or areas for improvement
        3. Additional tool queries that could further improve your response
        
        Format your response as a JSON with the following keys:
        - "answer": Your revised response
        - "reflection": Your new self-critique
        - "tool_queries": A list of dictionaries with "tool" and "query" keys
        """
        
        response = self.llm.invoke([HumanMessage(content=prompt)])
        
        try:
            # Parse response as JSON
            import json
            return json.loads(response.content)
        except:
            # Fallback in case of parsing issues
            return {
                "answer": response.content,
                "reflection": "Unable to generate structured reflection.",
                "tool_queries": []
            }
    
    @traceable(run_name="reflexion_agent_run")
    def run(self, query: str) -> str:
        """Run the full Reflexion agent loop."""
        # Step 1-2: Generate initial response with self-critique and tool suggestions
        current_response = self.generate_initial_response(query)
        
        iteration = 0
        while iteration < self.max_iterations:
            # Step 3: Execute tool queries
            tool_results = self.execute_tool_queries(current_response["tool_queries"])
            
            if not tool_results:
                # No tools were executed or no new information gathered
                break
                
            # Step 4-5: Revise response with new context
            current_response = self.revise_response(
                query, 
                current_response, 
                tool_results
            )
            
            # If no new tool queries are suggested, we're done
            if not current_response["tool_queries"]:
                break
                
            iteration += 1
        
        # Return the final answer
        return current_response["answer"]


# Example usage
if __name__ == "__main__":
    agent = ReflexionAgent(max_iterations=3)
    
    user_query = "What are the latest developments in fusion energy research?"
    final_answer = agent.run(user_query)
    
    print(f"Query: {user_query}")
    print(f"Final Answer: {final_answer}")