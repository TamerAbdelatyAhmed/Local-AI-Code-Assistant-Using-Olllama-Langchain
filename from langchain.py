from langchain.tools import tool
from langchain_ollama import OllamaLLM
from langchain.agents import Agent, Task
import subprocess

# 1. Configuration and Tools
llm = OllamaLLM(model="codeLlama")

# Define the CLITool class with the @tool decorator
class CLITool:
    @tool("Executor")
    def execute_cli_command(self, command: str):
        """Execute a CLI command using subprocess."""
        try:
            result = subprocess.run(command, shell=True, capture_output=True, text=True)
            return result.stdout if result.returncode == 0 else result.stderr
        except Exception as e:
            return str(e)

# 2. Creating an Agent for CLI tasks
cli_tool = CLITool()  # Create an instance of the tool
cli_agent = Agent(
    role='Software Engineer',
    goal='Always use Executor Tool. Ability to perform CLI operations, write programs and execute using Executor Tool',
    backstory='Expert in command line operations, creating and executing code.',
    tools=[cli_tool.execute_cli_command],  # Pass the method as a tool
    verbose=True,
    llm=llm 
)

# 3. Defining a Task for CLI operations
cli_task = Task(
    description='Identify the OS and then empty my recycle bin',
    expected_output='The result of the CLI operation',
    agent=cli_agent,
    tools=[cli_tool.execute_cli_command]  # Use the method from the tool instance
)

# 4. Creating the Crew (unchanged)
cli_crew = Crew(
    agents=[cli_agent],
    tasks=[cli_task],
    process=Process.sequential,
    manager_llm=llm
)

# 5. Run the Crew
import gradio as gr

def cli_interface(command):
    cli_task.description = command  
    result = cli_crew.kickoff()
    return result

iface = gr.Interface(
    fn=cli_interface, 
    inputs=gr.Textbox(lines=2, placeholder="What action to take?"), 
    outputs="text",
    title="CLI Command Executor",
    description="Execute CLI commands via a natural language interface."
)

iface.launch()
