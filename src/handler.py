import runpod
from utils import JobInput
from engine import vLLMEngine, OpenAIvLLMEngine

# Lazy initialization to prevent vLLM V1 multiprocessing spawn errors.
# vLLM V1 uses Python spawn multiprocessing, which re-imports this module.
# Eager initialization would cause recursive engine creation and crash.
vllm_engine = None
openai_engine = None

def get_engine():
    global vllm_engine
    if vllm_engine is None:
        vllm_engine = vLLMEngine()
    return vllm_engine

def get_openai_engine():
    global openai_engine
    if openai_engine is None:
        openai_engine = OpenAIvLLMEngine(get_engine())
    return openai_engine

async def handler(job):
    job_input = JobInput(job["input"])
    engine = get_openai_engine() if job_input.openai_route else get_engine()
    results_generator = engine.generate(job_input)
    async for batch in results_generator:
        yield batch

runpod.serverless.start(
    {
        "handler": handler,
        "concurrency_modifier": lambda x: get_engine().max_concurrency,
        "return_aggregate_stream": True,
    }
)