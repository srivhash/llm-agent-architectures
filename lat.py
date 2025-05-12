from langsmith import traceable
from langchain_core.messages import HumanMessage, AIMessage
from langchain_core.tools import tool
from langchain_openai import ChatOpenAI
from typing import List, Dict, Any, Optional, Tuple
import os
import math
import json
import numpy as np
from dataclasses import dataclass, field

# Setup LangSmith tracing
os.environ["LANGCHAIN_TRACING_V2"] = "true"
os.environ["LANGCHAIN_ENDPOINT"] = "https://api.smith.langchain.com"
os.environ["LANGCHAIN_API_KEY"] = "" 
os.environ["LANGCHAIN_PROJECT"] = "llm-mcts-agent"

@dataclass
class MCTSNode:
    """A node in the Monte Carlo Tree Search."""
    id: str
    parent_id: Optional[str] = None
    content: str = ""  # The actual response content
    reflection: str = ""  # Reflection on the content
    score: float = 0.0  # Score from the reflection
    is_solution: bool = False  # Whether this node is considered a solution
    children: List[str] = field(default_factory=list)  # Child node IDs
    visits: int = 0  # Number of times this node has been visited
    type: str = "response"  # Either "response" or "tool_execution"
    
    def is_leaf(self) -> bool:
        """Check if the node is a leaf node."""
        return len(self.children) == 0
    
    def uct_score(self, parent_visits: int, exploration_weight: float = 1.0) -> float:
        """Calculate the UCT score for node selection."""
        if self.visits == 0:
            return float('inf')
        
        # Exploitation term: normalized score (assuming scores are between 0 and 1)
        exploitation = self.score
        
        # Exploration term: UCB1 formula
        exploration = exploration_weight * math.sqrt(math.log(parent_visits) / self.visits)
        
        return exploitation + exploration


class LLMMCTSAgent:
    def __init__(self, max_depth=5, num_candidates=3, score_threshold=0.8):
        """Initialize the MCTS agent with LLMs."""
        self.llm = ChatOpenAI(model="gpt-4", temperature=0.7)
        self.max_depth = max_depth  # Maximum search depth
        self.num_candidates = num_candidates  # Number of candidates to generate at each node
        self.score_threshold = score_threshold  # Score threshold to consider a solution
        self.nodes = {}  # Dictionary mapping node IDs to MCTSNode objects
        self.node_counter = 0  # Counter for generating unique node IDs
        
        # Register tools that can be used by the agent
        self.tools = {
            "web_search": self.web_search,
            "calculator": self.calculator,
        }
    
    def _generate_node_id(self) -> str:
        """Generate a unique node ID."""
        node_id = f"node_{self.node_counter}"
        self.node_counter += 1
        return node_id
    
    @tool
    @traceable(name="web_search")
    def web_search(self, query: str) -> str:
        """Simulate a web search tool."""
        # In a real implementation, this would connect to a search API
        return f"Simulated web search results for query: '{query}'"
    
    @tool
    @traceable(name="calculator")
    def calculator(self, expression: str) -> str:
        """Simple calculator tool."""
        try:
            result = eval(expression)
            return f"Calculator result for '{expression}': {result}"
        except:
            return f"Error evaluating '{expression}'"
    
    @traceable(name="generate_initial_response")
    def generate_initial_response(self, query: str) -> MCTSNode:
        """Generate the initial response as the root node."""
        prompt = f"""
        You are helping a user with the following query: "{query}"
        
        Provide a helpful response OR suggest a tool execution that would help answer the query.
        
        If you can answer directly, provide a clear and informative response.
        If you need more information, suggest executing a specific tool with a specific input.
        
        Format your response as a JSON object with either:
        1. For a direct answer:
           {{"type": "response", "content": "your detailed answer here"}}
           
        2. For a tool execution:
           {{"type": "tool_execution", "tool": "tool_name", "input": "tool input"}}
        """
        
        response = self.llm.invoke([HumanMessage(content=prompt)])
        
        try:
            response_data = json.loads(response.content)
            node_id = self._generate_node_id()
            
            if response_data["type"] == "response":
                root_node = MCTSNode(
                    id=node_id,
                    content=response_data["content"],
                    type="response"
                )
            else:  # tool_execution
                tool_output = self.execute_tool(response_data["tool"], response_data["input"])
                root_node = MCTSNode(
                    id=node_id,
                    content=f"Tool: {response_data['tool']}\nInput: {response_data['input']}\nOutput: {tool_output}",
                    type="tool_execution"
                )
            
            self.nodes[node_id] = root_node
            return root_node
            
        except Exception as e:
            # Fallback in case of parsing issues
            node_id = self._generate_node_id()
            root_node = MCTSNode(
                id=node_id,
                content=f"I'll help you with: '{query}'\n\n{response.content}",
                type="response"
            )
            self.nodes[node_id] = root_node
            return root_node
    
    def execute_tool(self, tool_name: str, tool_input: str) -> str:
        """Execute a specific tool with the given input."""
        if tool_name in self.tools:
            return self.tools[tool_name](tool_input)
        return f"Error: Tool '{tool_name}' not found"
    
    @traceable(name="evaluate_node")
    def evaluate_node(self, node: MCTSNode, query: str, path_to_node: List[MCTSNode]) -> MCTSNode:
        """Evaluate a node, generating a reflection, score, and solution determination."""
        # Construct the context from the path to this node
        context = ""
        for i, path_node in enumerate(path_to_node):
            if i > 0:  # Skip the root node in reflection
                context += f"Step {i}: {path_node.content}\n"
                if path_node.reflection:
                    context += f"Reflection: {path_node.reflection}\n"
        
        prompt = f"""
        User query: "{query}"
        
        Previous steps:
        {context}
        
        Current output:
        {node.content}
        
        Please evaluate this output by providing:
        1. A detailed reflection on the quality, relevance, and completeness of the output
        2. A score between 0.0 and 1.0 (where 1.0 is perfect)
        3. Whether this output can be considered a complete solution to the user's query
        
        Format your response as a JSON with the following keys:
        - "reflection": Your detailed evaluation of the output
        - "score": A float between 0.0 and 1.0
        - "is_solution": true/false
        """
        
        response = self.llm.invoke([HumanMessage(content=prompt)])
        
        try:
            evaluation = json.loads(response.content)
            node.reflection = evaluation["reflection"]
            node.score = float(evaluation["score"])
            node.is_solution = bool(evaluation["is_solution"])
        except Exception as e:
            # Fallback in case of parsing issues
            node.reflection = f"Error parsing evaluation: {str(e)}"
            node.score = 0.0
            node.is_solution = False
        
        node.visits += 1
        return node
    
    @traceable(name="generate_candidates")
    def generate_candidates(self, 
                           query: str, 
                           parent_node: MCTSNode, 
                           path_to_parent: List[MCTSNode]) -> List[MCTSNode]:
        """Generate candidate child nodes based on the parent node."""
        # Construct context from the path
        context = ""
        for i, path_node in enumerate(path_to_parent):
            context += f"Step {i+1}: {path_node.content}\n"
            if path_node.reflection:
                context += f"Reflection: {path_node.reflection}\n"
        
        candidates = []
        
        prompt = f"""
        User query: "{query}"
        
        Context so far:
        {context}
        
        The last step had the following reflection:
        {parent_node.reflection}
        
        Generate a new, improved next step to better answer the user's query. 
        This can be either a direct response or a tool execution.
        
        Format your response as a JSON object with either:
        1. For a direct answer:
           {{"type": "response", "content": "your detailed answer here"}}
           
        2. For a tool execution:
           {{"type": "tool_execution", "tool": "tool_name", "input": "tool input"}}
        
        Your output should be different from previous steps and address the limitations identified in the reflection.
        """
        
        # Generate multiple candidates
        for _ in range(self.num_candidates):
            response = self.llm.invoke([HumanMessage(content=prompt)])
            
            try:
                response_data = json.loads(response.content)
                node_id = self._generate_node_id()
                
                if response_data["type"] == "response":
                    candidate = MCTSNode(
                        id=node_id,
                        parent_id=parent_node.id,
                        content=response_data["content"],
                        type="response"
                    )
                else:  # tool_execution
                    tool_output = self.execute_tool(response_data["tool"], response_data["input"])
                    candidate = MCTSNode(
                        id=node_id,
                        parent_id=parent_node.id,
                        content=f"Tool: {response_data['tool']}\nInput: {response_data['input']}\nOutput: {tool_output}",
                        type="tool_execution"
                    )
                
                self.nodes[node_id] = candidate
                parent_node.children.append(node_id)
                candidates.append(candidate)
                
            except Exception as e:
                # Handle parsing errors
                continue
        
        return candidates
    
    @traceable(name="select_best_node")
    def select_best_node(self, parent_node: MCTSNode) -> Optional[MCTSNode]:
        """Select the best child node based on UCT score."""
        if not parent_node.children:
            return None
        
        best_score = float('-inf')
        best_node = None
        
        for child_id in parent_node.children:
            child_node = self.nodes[child_id]
            uct_score = child_node.uct_score(parent_node.visits)
            
            if uct_score > best_score:
                best_score = uct_score
                best_node = child_node
        
        return best_node
    
    def _get_path_to_node(self, node: MCTSNode) -> List[MCTSNode]:
        """Get the path from root to the given node."""
        path = [node]
        current = node
        
        while current.parent_id is not None:
            current = self.nodes[current.parent_id]
            path.insert(0, current)
        
        return path
    
    @traceable(run_name="mcts_search")
    def search(self, query: str) -> Tuple[MCTSNode, List[MCTSNode]]:
        """Run the MCTS search algorithm."""
        # Generate the root node
        root_node = self.generate_initial_response(query)
        current_depth = 0
        
        # Evaluate the root node
        root_node = self.evaluate_node(root_node, query, [root_node])
        
        # Early termination if root is a solution
        if root_node.is_solution and root_node.score >= self.score_threshold:
            return root_node, [root_node]
        
        best_solution = None
        best_solution_score = float('-inf')
        
        # Main MCTS loop
        while current_depth < self.max_depth:
            current_depth += 1
            
            # 1. Selection: Find the current best node to expand from
            current_node = root_node
            path_to_current = [current_node]
            
            # Navigate down the tree using UCT until we reach a leaf
            while not current_node.is_leaf() and len(current_node.children) > 0:
                next_node = self.select_best_node(current_node)
                if next_node is None:
                    break
                current_node = next_node
                path_to_current.append(current_node)
            
            # 2. Expansion: Generate candidates from the selected node
            candidates = self.generate_candidates(query, current_node, path_to_current)
            
            # 3. Simulation and Backpropagation: Evaluate each candidate
            for candidate in candidates:
                candidate_path = path_to_current + [candidate]
                candidate = self.evaluate_node(candidate, query, candidate_path)
                
                # Track the best solution found so far
                if candidate.is_solution and candidate.score > best_solution_score:
                    best_solution = candidate
                    best_solution_score = candidate.score
                    
                    # Early termination if we found a good enough solution
                    if best_solution_score >= self.score_threshold:
                        return best_solution, self._get_path_to_node(best_solution)
        
        # If we've exhausted our search or reached max depth, return the best solution found
        if best_solution is not None:
            return best_solution, self._get_path_to_node(best_solution)
        
        # If no solution was found, return the highest-scoring leaf node
        best_leaf = None
        best_leaf_score = float('-inf')
        
        for node_id, node in self.nodes.items():
            if node.is_leaf() and node.score > best_leaf_score:
                best_leaf = node
                best_leaf_score = node.score
        
        if best_leaf is not None:
            return best_leaf, self._get_path_to_node(best_leaf)
        
        # Fallback to the root node if no better node was found
        return root_node, [root_node]
    
    @traceable(run_name="mcts_agent_run")
    def run(self, query: str) -> str:
        """Run the full MCTS agent and return the best solution."""
        best_node, path = self.search(query)
        
        # Format the solution for human readability
        response = f"Final answer for: {query}\n\n"
        
        # If the path is non-trivial, show the reasoning process
        if len(path) > 1:
            response += "Reasoning steps:\n"
            for i, node in enumerate(path):
                if i > 0:  # Skip the root in the explanation
                    response += f"\nStep {i}:\n{node.content}\n"
            
            response += f"\n---\n\nFinal answer:\n{best_node.content}\n"
            response += f"\nConfidence score: {best_node.score:.2f}"
        else:
            # Just the answer for a single-node path
            response += f"{best_node.content}\n"
            response += f"\nConfidence score: {best_node.score:.2f}"
        
        return response


# Example usage
if __name__ == "__main__":
    agent = LLMMCTSAgent(max_depth=3, num_candidates=3, score_threshold=0.85)
    
    user_query = "What would be the economic impact if we discovered room temperature superconductors tomorrow?"
    final_answer = agent.run(user_query)
    
    print(f"Query: {user_query}")
    print(f"Final Answer: {final_answer}")