# LLM Agent Architectures

This repository contains implementations of various agent architectures powered by Large Language Models (LLMs). It includes tools, strategies, and examples for building intelligent agents capable of solving complex tasks.

## Features

- **Rewoo Agent**: A tool-chaining agent with variable substitution and execution plans.
- **Plan-Execute Agent**: A step-by-step planner and executor for solving queries.
- **LLM-MCTS Agent**: A Monte Carlo Tree Search-based agent for iterative reasoning.
- **Reflexion Agent**: An agent that iteratively improves its responses through self-reflection.

## Repository Structure

- `rewoo.py`: Implementation of the Rewoo agent.
- `plan_execute.py`: Implementation of the Plan-Execute agent.
- `lat.py`: Implementation of the LLM-MCTS agent.
- `reflexion.py`: Implementation of the Reflexion agent.
- `README.md`: Documentation for the repository.

## Getting Started

1. **Clone the Repository**:
   ```bash
   git clone https://github.com/your-username/llm-agent-architectures.git
   cd llm-agent-architectures
   ```

2. **Install Dependencies**:
   Ensure you have Python 3.8+ installed, then run:
   ```bash
   pip install -r requirements.txt
   ```

3. **Run an Example**:
   Execute one of the agent scripts to see it in action:
   ```bash
   python rewoo.py
   ```

## Contributing

Contributions are welcome! Feel free to open issues or submit pull requests to improve the codebase.

## License

This project is licensed under the MIT License. See the `LICENSE` file for details.
