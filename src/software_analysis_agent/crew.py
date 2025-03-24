from crewai import Agent, Crew, Process, Task, LLM
from crewai.project import CrewBase, agent, crew, task
import os
from loguru import logger

from software_analysis_agent.tools.tavily_tool import TavilySearchTool
from dotenv import load_dotenv
load_dotenv(override=True)

# If you want to run a snippet of code before or after the crew starts, 
# you can use the @before_kickoff and @after_kickoff decorators
# https://docs.crewai.com/concepts/crews#example-crew-class-with-decorators

@CrewBase
class SoftwareAnalysisAgent():
	"""SoftwareAnalysisAgent crew"""

	# Learn more about YAML configuration files here:
	# Agents: https://docs.crewai.com/concepts/agents#yaml-configuration-recommended
	# Tasks: https://docs.crewai.com/concepts/tasks#yaml-configuration-recommended
	agents_config = 'config/agents.yaml'
	tasks_config = 'config/tasks.yaml'

	def __init__(self, result_file: str = None):
		self.llm = self.load_llm_model()
		self.MAX_ITER = int(os.getenv('MAX_ITER', 5))
		self.VERBOSE = os.getenv('VERBOSE', 'true').lower() == 'true'
		self.TAVILY_API_KEY = os.getenv('TAVILY_API_KEY', None)
		# Initialize the tool
		# self.code_interpreter = CodeInterpreterTool()
		self.tavily_search_tool = TavilySearchTool(api_key=self.TAVILY_API_KEY)
		self.result_file = result_file

	def load_llm_model(self):
		MODEL = os.getenv("MODEL", "openai/gpt-4o-mini")
		MODEL_TEMPERATURE = float(os.getenv("MODEL_TEMPERATURE", 0.0))
		CONTEXT_WINDOW_SIZE = int(os.getenv("CONTEXT_WINDOW_SIZE", 0))
		model = f"{MODEL}"
		logger.info(f"Loading LLM model: {model} {MODEL_TEMPERATURE} {CONTEXT_WINDOW_SIZE}")
		if CONTEXT_WINDOW_SIZE>0:
			llm = LLM(
				model=model,
				temperature=MODEL_TEMPERATURE,
				max_tokens=CONTEXT_WINDOW_SIZE,                
			)
		else:
			llm = LLM(
				model=model,
				temperature=MODEL_TEMPERATURE,					
			)
		
		return llm

	# If you would like to add tools to your agents, you can learn more about it here:
	# https://docs.crewai.com/concepts/agents#agent-tools

	@agent
	def project_manager(self) -> Agent:
		return Agent(
			config=self.agents_config['project_manager'],
			llm=self.llm,
			max_iter=self.MAX_ITER,
			verbose=self.VERBOSE
		)

	@agent
	def business_analyst(self) -> Agent:
		return Agent(
			config=self.agents_config['business_analyst'],
			llm=self.llm,
			max_iter=self.MAX_ITER,
			verbose=self.VERBOSE
		)

	@agent
	def software_architect(self) -> Agent:
		return Agent(
			config=self.agents_config['software_architect'],
			llm=self.llm,
			max_iter=self.MAX_ITER,
			verbose=self.VERBOSE
		)

	@agent
	def technical_designer(self) -> Agent:
		return Agent(
			config=self.agents_config['technical_designer'],
			llm=self.llm,
			max_iter=self.MAX_ITER,
			verbose=self.VERBOSE
		)


	@task
	def business_requirement_analysis_task(self) -> Task:
		return Task(
			config=self.tasks_config['business_requirement_analysis_task'],
			output_file='result/business_requirements.md',
		)

	@task
	def functional_design_task(self) -> Task:
		return Task(
			config=self.tasks_config['functional_design_task'],
			output_file='result/functional_design.md',
		)

	@task
	def software_architecture_design_task(self) -> Task:
		return Task(
			config=self.tasks_config['software_architecture_design_task'],
			output_file='result/software_architecture.md',
		)

	@task
	def technical_design_task(self) -> Task:
		return Task(
			config=self.tasks_config['technical_design_task'],
			output_file='result/technical_design.md',
		)
	
	@task
	def project_initiation_task(self) -> Task:
		return Task(
			config=self.tasks_config['project_initiation_task'],
			output_file='result/project_plan.md',
		)

	@crew
	def crew(self) -> Crew:
		"""Creates the SoftwareAnalysisAgent crew"""
		return Crew(
			agents=self.agents,
			tasks=self.tasks,
			process=Process.hierarchical,
			manager_llm=self.llm,
			verbose=self.VERBOSE,
		)
	
	
