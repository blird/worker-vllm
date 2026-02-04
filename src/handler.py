import os
import multiprocessing
import runpod
from utils import JobInput
from engine import vLLMEngine, OpenAIvLLMEngine

# Guard against vLLM V1 spawn multiprocessing re-importing this module.
# vLLM V1 uses spawn which re-executes module-level code in child processes.
# Only initialize engines in the main process to prevent recursive engine creation.
_is_main_process = (
    multiprocessing.current_process().name == 'MainProcess' or
    os.environ.get('RUNPOD_POD_ID')  # RunPod sets this in workers
)

if _is_main_process:
    vllm_engine = vLLMEngine()
    openai_engine = OpenAIvLLMEngine(vllm_engine)
else:
    vllm_engine = None
    openai_engine = None

async def handler(job):
    job_input = JobInput(job["input"])
    engine = openai_engine if job_input.openai_route else vllm_engine
    results_generator = engine.generate(job_input)
    async for batch in results_generator:
        yield batch

if __name__ == '__main__':
    runpod.serverless.start(
        {
            "handler": handler,
            "concurrency_modifier": lambda x: vllm_engine.max_concurrency,
            "return_aggregate_stream": True,
        }
    )
