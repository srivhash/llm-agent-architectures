from langsmith import traceable
from langchain_core.messages import HumanMessage, AIMessage
from langchain_core.tools import tool
from langchain_openai import ChatOpenAI
from typing import List, Dict, Any, Optional, Tuple
import os
import json
from dataclasses import dataclass

# Setup LangSmith tracing
os.environ["LANGCHAIN_TRACING_V2"] = "true"
os.environ["LANGCHAIN_ENDPOINT"] = "https://api.smith.langchain.com"
os.environ["LANGCHAIN_API_KEY"] = ""
os.environ["LANGCHAIN_PROJECT"] = "plan-execute-agent"

@dataclass
class Step:
    """A step in the execution plan."""
    description: str
    is_complete: bool = False
    output: str = ""


@dataclass
class ExecutionPlan:
    """The overall execution plan for solving a query."""
    steps: List[Step]
    current_step_index: int = 0
    is_complete: bool = False
    final_answer: str = ""


class PlanExecuteAgent:
    def __init__(self, max_iterations=10):
        """Initialize the Plan and Execute agent with LLMs."""
        self.planner_llm = ChatOpenAI(model="gpt-4", temperature=0.2)  # Low temperature for planning
        self.execution_llm = ChatOpenAI(model="gpt-4", temperature=0.7)  # Higher temperature for diverse execution
        self.max_iterations = max_iterations
        
        # Register tools that can be used by the agent
        self.tools = {
            "web_search": self.web_search,
            "calculator": self.calculator,
            "code_interpreter": self.code_interpreter,
        }
    
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
    
    @tool
    @traceable(name="code_interpreter")
    def code_interpreter(self, code: str) -> str:
        """Simulate code execution."""
        return f"Simulated execution of code:\n{code}\n\nSimulated output: [Output would appear here]"
    
    @traceable(name="create_initial_plan")
    def create_initial_plan(self, query: str) -> ExecutionPlan:
        """Create an initial step-by-step plan for solving the query."""
        prompt = f"""
        You are a strategic planner. Given a user query, create a detailed step-by-step plan to solve it.
        
        For each step, consider:
        1. What specific information needs to be gathered
        2. What calculations or analyses need to be performed
        3. What tools might be useful (web_search, calculator, code_interpreter)
        
        Make your plan specific, actionable, and comprehensive. Break complex steps into simpler ones.
        
        Format your response as a JSON array of steps, where each step is an object with a "description" key.
        
        User Query: {query}
        """
        
        response = self.planner_llm.invoke([HumanMessage(content=prompt)])
        
        try:
            # Parse the steps from the response
            steps_data = json.loads(response.content)
            steps = [Step(description=step["description"]) for step in steps_data]
            return ExecutionPlan(steps=steps)
        except Exception as e:
            # Fallback in case of parsing issues
            # Extract potential steps using simple text parsing
            content = response.content
            lines = content.split('\n')
            steps = []
            
            for line in lines:
                line = line.strip()
                if line and (line.startswith("- ") or line.startswith("Step ") or line[0].isdigit() and line[1] == '.'):
                    # Remove leading markers and numbers
                    description = line
                    for prefix in ["- ", "Step ", "1. ", "2. ", "3. ", "4. ", "5. ", "6. ", "7. ", "8. ", "9. "]:
                        if description.startswith(prefix):
                            description = description[len(prefix):]
                            break
                    steps.append(Step(description=description))
            
            if not steps:
                # If we couldn't parse steps, create a single generic step
                steps = [Step(description="Solve the query: " + query)]
                
            return ExecutionPlan(steps=steps)
    
    @traceable(name="execute_step")
    def execute_step(self, query: str, plan: ExecutionPlan, step_index: int) -> str:
        """Execute a specific step in the plan."""
        # Get the current step description
        current_step = plan.steps[step_index]
        
        # Construct context from previous steps
        previous_steps_context = ""
        for i in range(step_index):
            if plan.steps[i].is_complete:
                previous_steps_context += f"Step {i+1}: {plan.steps[i].description}\n"
                previous_steps_context += f"Output: {plan.steps[i].output}\n\n"
        
        prompt = f"""
        You are helping with the query: "{query}"
        
        The overall plan is:
        {self._format_plan_for_prompt(plan)}
        
        Previous steps completed:
        {previous_steps_context if previous_steps_context else "No steps completed yet."}
        
        Your current task is to execute Step {step_index + 1}:
        {current_step.description}
        
        Available tools:
        - web_search: Search the web for information
        - calculator: Perform mathematical calculations
        - code_interpreter: Execute code to process data or perform analyses
        
        You can either:
        1. Execute a tool by responding with a JSON object:
           {{"tool": "tool_name", "input": "tool_input"}}
           
        2. Generate a direct response if no tool is needed:
           {{"response": "your detailed response here"}}
        
        Focus only on this specific step. Be thorough and detailed.
        """
        
        response = self.execution_llm.invoke([HumanMessage(content=prompt)])
        
        try:
            # Parse the response
            response_data = json.loads(response.content)
            
            if "tool" in response_data:
                # Execute the specified tool
                tool_name = response_data["tool"]
                tool_input = response_data["input"]
                
                if tool_name in self.tools:
                    tool_output = self.tools[tool_name](tool_input)
                    return f"Used tool '{tool_name}' with input '{tool_input}'.\nResult: {tool_output}"
                else:
                    return f"Error: Tool '{tool_name}' not found. Proceeded with direct reasoning instead.\n{response.content}"
            else:
                # Direct response without tool
                return response_data["response"]
                
        except Exception as e:
            # If parsing fails, treat the whole response as a direct answer
            return f"Direct response: {response.content}"
    
    def _format_plan_for_prompt(self, plan: ExecutionPlan) -> str:
        """Format the execution plan for inclusion in prompts."""
        formatted_plan = ""
        for i, step in enumerate(plan.steps):
            status = "✓" if step.is_complete else "○"
            formatted_plan += f"{status} Step {i+1}: {step.description}\n"
        return formatted_plan
    
    @traceable(name="replan")
    def replan(self, query: str, plan: ExecutionPlan) -> ExecutionPlan:
        """Re-evaluate the plan based on the execution results so far."""
        # Construct context from all steps executed so far
        steps_context = ""
        for i, step in enumerate(plan.steps):
            if step.is_complete:
                steps_context += f"Step {i+1}: {step.description}\n"
                steps_context += f"Output: {step.output}\n\n"
        
        current_step_index = plan.current_step_index
        
        prompt = f"""
        You are evaluating and updating a plan to solve the following query: "{query}"
        
        The current plan is:
        {self._format_plan_for_prompt(plan)}
        
        Progress so far:
        {steps_context if steps_context else "No steps completed yet."}
        
        Current step being worked on: Step {current_step_index + 1}
        
        Based on the progress so far, please:
        1. Evaluate if the plan needs to be modified
        2. Decide if the query has been adequately answered
        
        Format your response as a JSON object with the following structure:
        {{
            "plan_evaluation": "Your assessment of the current plan",
            "is_complete": true/false (whether the query has been adequately answered),
            "final_answer": "The final answer to provide to the user if complete, otherwise empty string",
            "updated_steps": [
                {{"description": "Step 1 description", "is_complete": true/false}},
                {{"description": "Step 2 description", "is_complete": true/false}},
                ...
            ],
            "next_step_index": 0 (the index of the next step to execute, or same as current if plan continues)
        }}
        """
        
        response = self.planner_llm.invoke([HumanMessage(content=prompt)])
        
        try:
            # Parse the replanning response
            replan_data = json.loads(response.content)
            
            # Update the plan with the new steps
            updated_steps = []
            for i, step_data in enumerate(replan_data["updated_steps"]):
                if i < len(plan.steps) and plan.steps[i].is_complete:
                    # Preserve completed steps and their outputs
                    updated_steps.append(plan.steps[i])
                else:
                    # Add new or modified steps
                    updated_steps.append(Step(
                        description=step_data["description"],
                        is_complete=step_data.get("is_complete", False)
                    ))
            
            # Create updated plan
            updated_plan = ExecutionPlan(
                steps=updated_steps,
                current_step_index=replan_data["next_step_index"],
                is_complete=replan_data["is_complete"],
                final_answer=replan_data.get("final_answer", "")
            )
            
            return updated_plan
            
        except Exception as e:
            # If parsing fails, keep the original plan but move to the next step
            next_step = min(plan.current_step_index + 1, len(plan.steps) - 1)
            plan.current_step_index = next_step
            
            # Check if we've reached the end of the plan
            if next_step == len(plan.steps) - 1:
                # If we're at the last step, mark as complete and extract a final answer from the response
                final_answer_lines = []
                for line in response.content.split("\n"):
                    if "final answer" in line.lower() or "conclusion" in line.lower():
                        final_answer_lines.append(line)
                
                if final_answer_lines:
                    plan.final_answer = "\n".join(final_answer_lines)
                    plan.is_complete = True
                    
            return plan
    
    @traceable(run_name="plan_execute_run")
    def run(self, query: str) -> str:
        """Run the full Plan and Execute agent loop."""
        # Step 1-2: Create initial plan
        plan = self.create_initial_plan(query)
        
        iteration = 0
        while not plan.is_complete and iteration < self.max_iterations:
            # Check if we're within the plan's bounds
            if plan.current_step_index >= len(plan.steps):
                break
                
            # Step 3: Execute the current step
            step_output = self.execute_step(query, plan, plan.current_step_index)
            
            # Update the step's output and mark it as complete
            plan.steps[plan.current_step_index].output = step_output
            plan.steps[plan.current_step_index].is_complete = True
            
            # Step 4-5: Replan based on execution results
            plan = self.replan(query, plan)
            
            iteration += 1
        
        # Format the final response
        if plan.is_complete and plan.final_answer:
            return self._format_final_answer(query, plan)
        else:
            # If we ran out of iterations or something went wrong, return what we have
            return self._format_incomplete_answer(query, plan)
    
    def _format_final_answer(self, query: str, plan: ExecutionPlan) -> str:
        """Format the final answer when the plan is complete."""
        response = f"Answer to: {query}\n\n"
        response += f"{plan.final_answer}\n\n"
        
        response += "The solution was reached through the following steps:\n"
        for i, step in enumerate(plan.steps):
            if step.is_complete:
                response += f"\nStep {i+1}: {step.description}\n"
                response += f"Result: {step.output}\n"
        
        return response
    
    def _format_incomplete_answer(self, query: str, plan: ExecutionPlan) -> str:
        """Format a response when the plan couldn't be completed."""
        response = f"Partial answer to: {query}\n\n"
        
        # Include progress so far
        response += "Progress made:\n"
        for i, step in enumerate(plan.steps):
            status = "✓" if step.is_complete else "○"
            response += f"\n{status} Step {i+1}: {step.description}\n"
            if step.is_complete:
                response += f"Result: {step.output}\n"
        
        # Extract what we've learned so far
        response += "\nBased on the steps completed, here's what we know:\n"
        
        # Attempt to synthesize a partial answer
        partial_answer_prompt = f"""
        Based on the following incomplete execution of a plan to answer "{query}":
        
        {self._format_plan_for_prompt(plan)}
        
        Progress details:
        """
        
        for i, step in enumerate(plan.steps):
            if step.is_complete:
                partial_answer_prompt += f"\nStep {i+1}: {step.description}\n"
                partial_answer_prompt += f"Result: {step.output}\n"
        
        partial_answer_prompt += "\nPlease provide the best partial answer possible based on the information gathered so far."
        
        try:
            partial_answer = self.planner_llm.invoke([HumanMessage(content=partial_answer_prompt)])
            response += partial_answer.content
        except:
            response += "Unable to synthesize a partial answer based on the information gathered."
        
        return response


# Example usage
if __name__ == "__main__":
    agent = PlanExecuteAgent(max_iterations=5)
    
    user_query = "Analyze the potential impacts of quantum computing on modern cryptography systems."
    final_answer = agent.run(user_query)
    
    print(f"Query: {user_query}")
    print(f"Final Answer: {final_answer}")