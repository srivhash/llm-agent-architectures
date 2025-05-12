from langsmith import traceable
from langchain_core.messages import HumanMessage, AIMessage
from langchain_core.tools import tool
from langchain_openai import ChatOpenAI
from typing import List, Dict, Any, Optional, Tuple, Set
import os
import json
import re
from dataclasses import dataclass, field

# Setup LangSmith tracing
os.environ["LANGCHAIN_TRACING_V2"] = "true"
os.environ["LANGCHAIN_ENDPOINT"] = "https://api.smith.langchain.com"
os.environ["LANGCHAIN_API_KEY"] = ""  
os.environ["LANGCHAIN_PROJECT"] = "rewoo-agent"  

@dataclass
class PlanStep:
    """A step in the Rewoo execution plan."""
    id: str  # Unique identifier for the step
    description: str  # Human-readable description
    tool: str  # Tool to use for this step
    input_template: str  # Input template with variable placeholders
    output_var: str  # Variable name to store the output
    is_executed: bool = False  # Whether this step has been executed
    output: str = ""  # Output from executing this step


@dataclass
class RewooExecutionPlan:
    """The overall execution plan for solving a query using Rewoo."""
    steps: List[PlanStep]  # Ordered list of steps
    evidence: Dict[str, str] = field(default_factory=dict)  # Variable substitutions


class RewooAgent:
    def __init__(self):
        """Initialize the Rewoo agent with LLMs."""
        self.planner_llm = ChatOpenAI(model="gpt-4", temperature=0.2)  # Low temperature for planning
        self.execution_llm = ChatOpenAI(model="gpt-4", temperature=0.5)  # Mid temperature for execution
        self.solver_llm = ChatOpenAI(model="gpt-4", temperature=0.7)  # Higher temperature for synthesis
        
        # Register tools that can be used by the agent
        self.tools = {
            "web_search": self.web_search,
            "calculator": self.calculator,
            "code_interpreter": self.code_interpreter,
            "read_file": self.read_file,
            "data_analysis": self.data_analysis,
        }
        
        # Regex for variable substitution
        self.var_pattern = re.compile(r'\${([^}]+)}')
    
    @tool
    @traceable(name="web_search")
    def web_search(self, query: str) -> str:
        """Search the web for information."""
        # In a real implementation, this would connect to a search API
        return f"Simulated web search results for: '{query}'"
    
    @tool
    @traceable(name="calculator")
    def calculator(self, expression: str) -> str:
        """Perform mathematical calculations."""
        try:
            result = eval(expression)
            return f"Calculator result for '{expression}': {result}"
        except Exception as e:
            return f"Error evaluating '{expression}': {str(e)}"
    
    @tool
    @traceable(name="code_interpreter")
    def code_interpreter(self, code: str) -> str:
        """Execute code and return the result."""
        return f"Simulated code execution for:\n{code}\n\nResult: [Execution output would appear here]"
    
    @tool
    @traceable(name="read_file")
    def read_file(self, file_path: str) -> str:
        """Read the contents of a file."""
        return f"Simulated file content for: '{file_path}'"
    
    @tool
    @traceable(name="data_analysis")
    def data_analysis(self, data_and_query: str) -> str:
        """Analyze data and answer questions about it."""
        return f"Simulated data analysis for: '{data_and_query}'"
    
    def _extract_variables(self, text: str) -> Set[str]:
        """Extract variable references from a text string."""
        return set(self.var_pattern.findall(text))
    
    def _substitute_variables(self, template: str, variables: Dict[str, str]) -> str:
        """Substitute variables in a template string."""
        result = template
        for var_name, var_value in variables.items():
            result = result.replace(f"${{{var_name}}}", var_value)
        return result
    
    @traceable(name="generate_plan")
    def generate_plan(self, query: str) -> RewooExecutionPlan:
        """Generate a structured execution plan with variable substitution."""
        prompt = f"""
        You are an expert planner for a Rewoo agent that uses tool chaining with variable substitution.
        Given a user query, create a detailed plan with steps that use available tools.
        
        Use the format where each step's output is stored in a variable that can be used in subsequent steps.
        
        Available tools:
        - web_search: Search the web for information
        - calculator: Perform mathematical calculations
        - code_interpreter: Execute code to process data
        - read_file: Read content from a file
        - data_analysis: Analyze data and answer questions about it
        
        FORMAT YOUR RESPONSE AS JSON with the following structure:
        {{
            "steps": [
                {{
                    "id": "step1",
                    "description": "Human-readable description of this step",
                    "tool": "name_of_tool",
                    "input_template": "Input to the tool with ${variable} references if needed",
                    "output_var": "variable_name_to_store_result"
                }},
                ...more steps...
            ]
        }}
        
        Use variable substitution by referencing previous step outputs with ${{variable_name}}.
        Create a thorough plan that breaks down the problem into logical steps.
        
        User Query: {query}
        """
        
        response = self.planner_llm.invoke([HumanMessage(content=prompt)])
        
        try:
            # Parse the plan from the response
            plan_data = json.loads(response.content)
            
            # Convert to our plan objects
            steps = []
            for step_data in plan_data["steps"]:
                steps.append(PlanStep(
                    id=step_data["id"],
                    description=step_data["description"],
                    tool=step_data["tool"],
                    input_template=step_data["input_template"],
                    output_var=step_data["output_var"]
                ))
            
            return RewooExecutionPlan(steps=steps)
            
        except Exception as e:
            # Fallback in case of parsing issues - create a simple plan
            error_msg = f"Error parsing plan: {str(e)}"
            print(error_msg)
            
            # Create a minimal fallback plan
            steps = [
                PlanStep(
                    id="step1",
                    description=f"Search for information about: {query}",
                    tool="web_search",
                    input_template=query,
                    output_var="search_result"
                ),
                PlanStep(
                    id="step2",
                    description="Analyze the search results",
                    tool="code_interpreter",
                    input_template="Analyze this information and extract key points:\n${search_result}",
                    output_var="analysis"
                )
            ]
            
            return RewooExecutionPlan(steps=steps)
    
    @traceable(name="execute_step")
    def execute_step(self, step: PlanStep, variables: Dict[str, str]) -> str:
        """Execute a single step with variable substitution."""
        # Substitute variables in the input template
        input_with_vars = self._substitute_variables(step.input_template, variables)
        
        # Check if the requested tool exists
        if step.tool not in self.tools:
            return f"Error: Tool '{step.tool}' not found. Available tools: {', '.join(self.tools.keys())}"
        
        # Execute the tool
        try:
            result = self.tools[step.tool](input_with_vars)
            return result
        except Exception as e:
            return f"Error executing {step.tool}: {str(e)}"
    
    @traceable(name="execute_plan")
    def execute_plan(self, plan: RewooExecutionPlan) -> RewooExecutionPlan:
        """Execute all steps in the plan with variable substitution."""
        for step in plan.steps:
            # Execute the step
            output = self.execute_step(step, plan.evidence)
            
            # Store the step output in the evidence dict
            plan.evidence[step.output_var] = output
            step.output = output
            step.is_executed = True
        
        return plan
    
    @traceable(name="solve")
    def solve(self, query: str, plan: RewooExecutionPlan) -> str:
        """Generate a final answer based on the executed plan and evidence."""
        # Format the evidence
        evidence_str = ""
        for step in plan.steps:
            evidence_str += f"\nStep {step.id}: {step.description}\n"
            evidence_str += f"Tool: {step.tool}\n"
            evidence_str += f"Input: {self._substitute_variables(step.input_template, plan.evidence)}\n"
            evidence_str += f"Output ({step.output_var}): {step.output}\n"
            evidence_str += "-" * 40 + "\n"
        
        prompt = f"""
        You are answering the following user query: "{query}"
        
        The following steps were executed to gather information:
        {evidence_str}
        
        Based on all the information gathered, please provide a comprehensive answer to the user's query.
        Your answer should:
        1. Directly address what the user asked
        2. Synthesize the information from all steps
        3. Present a coherent, well-structured response
        4. Acknowledge any limitations or uncertainties in the available information
        
        Your answer:
        """
        
        response = self.solver_llm.invoke([HumanMessage(content=prompt)])
        return response.content
    
    @traceable(run_name="rewoo_run")
    def run(self, query: str) -> Dict[str, Any]:
        """Run the full Rewoo agent pipeline and return the result with tracing info."""
        # Step 1: Generate the execution plan
        plan = self.generate_plan(query)
        
        # Step 2: Execute the plan with variable substitution
        executed_plan = self.execute_plan(plan)
        
        # Step 3: Generate the final answer based on evidence
        answer = self.solve(query, executed_plan)
        
        # Return the complete result with plan and evidence
        return {
            "query": query,
            "answer": answer,
            "plan": [
                {
                    "id": step.id,
                    "description": step.description,
                    "tool": step.tool,
                    "input": self._substitute_variables(step.input_template, executed_plan.evidence),
                    "output_var": step.output_var,
                    "output": step.output
                }
                for step in executed_plan.steps
            ],
            "evidence": executed_plan.evidence
        }


# Helper function to format the result for display
def format_result(result: Dict[str, Any]) -> str:
    """Format the result for display."""
    output = f"Query: {result['query']}\n\n"
    output += f"Answer: {result['answer']}\n\n"
    
    output += "Plan Execution:\n"
    for step in result["plan"]:
        output += f"\nStep {step['id']}: {step['description']}\n"
        output += f"Tool: {step['tool']}\n"
        output += f"Input: {step['input']}\n"
        output += f"Output: {step['output']}\n"
    
    return output


# Example usage
if __name__ == "__main__":
    agent = RewooAgent()
    
    user_query = "Compare the growth rates of renewable energy adoption in Germany and the United States over the past decade."
    result = agent.run(user_query)
    
    print(format_result(result))