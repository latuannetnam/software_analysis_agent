#!/usr/bin/env python
import os
import sys
import warnings

from datetime import datetime

from software_analysis_agent.crew import SoftwareAnalysisAgent

warnings.filterwarnings("ignore", category=SyntaxWarning, module="pysbd")

# This main file is intended to be a way for you to run your
# crew locally, so refrain from adding unnecessary logic into this file.
# Replace with inputs you want to test with, it will automatically
# interpolate any tasks and agents information

def run():
    """
    Run the crew.
    """
    INPUT_FILE  = os.getenv('INPUT_FILE', 'data/inputs.md')
    RESULT_FILE = os.getenv('RESULT_FILE', 'data/results.md')
    use_case = ""
    if not os.path.exists(INPUT_FILE):
        raise FileNotFoundError(f"Input file {INPUT_FILE} does not exist.")
    else:
        with open(INPUT_FILE, 'r') as f:
            use_case = f.read()
    inputs = {
        'use_case': use_case,
    }
    
    try:
        agent = SoftwareAnalysisAgent(result_file=RESULT_FILE).crew()
        result = agent.kickoff(inputs=inputs)
        print(f"Final result:\n {result}")
    except Exception as e:
        raise Exception(f"An error occurred while running the crew: {e}")


def train():
    """
    Train the crew for a given number of iterations.
    """
    inputs = {
        "topic": "AI LLMs"
    }
    try:
        SoftwareAnalysisAgent().crew().train(n_iterations=int(sys.argv[1]), filename=sys.argv[2], inputs=inputs)

    except Exception as e:
        raise Exception(f"An error occurred while training the crew: {e}")

def replay():
    """
    Replay the crew execution from a specific task.
    """
    try:
        SoftwareAnalysisAgent().crew().replay(task_id=sys.argv[1])

    except Exception as e:
        raise Exception(f"An error occurred while replaying the crew: {e}")

def test():
    """
    Test the crew execution and returns the results.
    """
    inputs = {
        "topic": "AI LLMs",
        "current_year": str(datetime.now().year)
    }
    try:
        SoftwareAnalysisAgent().crew().test(n_iterations=int(sys.argv[1]), openai_model_name=sys.argv[2], inputs=inputs)

    except Exception as e:
        raise Exception(f"An error occurred while testing the crew: {e}")
