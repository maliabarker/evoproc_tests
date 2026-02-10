import json
from typing import Any, Dict, List, Optional

from vllm import LLM, SamplingParams
from vllm.sampling_params import StructuredOutputsParams

from evoproc_procedures.models import Procedure
from evoproc_procedures.prompts import create_procedure_prompt

# Init vLLM
def init_vllm(model: str) -> LLM:
    # NOTE: keep this LLM instance alive for the whole GA run (don’t re-init each generation).
    # vLLM supports batched offline inference via llm.generate(list_of_prompts, sampling_params). :contentReference[oaicite:1]{index=1}
    return LLM(
        model=model,
        enable_prefix_caching=True,
        gpu_memory_utilization=0.40,   # 0.3 is likely too low for 120B :contentReference[oaicite:3]{index=3}
        max_model_len=8192,            # avoid default 131072 for GA :contentReference[oaicite:4]{index=4}
        max_num_seqs=32,
        max_num_batched_tokens=4096,
        disable_log_stats=True,
    )

def generate_initial_population(
    llm: LLM,
    problem_text: str,
    schema_text: str,
    pop_size: int,
    *,
    temperature: float = 0.8,
    top_p: float = 0.95,
    max_tokens: int = 600,
    base_seed: Optional[int] = None,
) -> List[Dict[str, Any]]:
    prompt = create_procedure_prompt(problem_text)

    # One prompt, many samples:
    sampling = SamplingParams(
        n=pop_size,
        temperature=temperature,
        top_p=top_p,
        max_tokens=max_tokens,
        seed=base_seed,  # if you want reproducibility across runs (or vary seeds per run) :contentReference[oaicite:3]{index=3}
        stop=None,
        structured_outputs=schema_text
    )

    outputs = llm.generate([prompt], sampling)
    # outputs is a list of RequestOutput objects; each has .prompt and .outputs[i].text :contentReference[oaicite:4]{index=4}
    out = outputs[0]

    procedures: List[Dict[str, Any]] = []
    for cand in out.outputs:
        text = cand.text.strip()
        # If your model sometimes wraps JSON in backticks, you can add cleanup here.
        try:
            procedures.append(json.loads(text))
        except json.JSONDecodeError:
            # Keep raw text for repair loop
            procedures.append({"__invalid_json__": text})

    return procedures

if __name__ == "__main__":
    example_q = "Natalia sold clips to 48 of her friends in April, and then she sold half as many clips in May. How many clips did Natalia sell altogether in April and May?"
    # Start LLM
    init_vllm("openai/gpt-oss-20b")
    # Generate initial population
    pop = generate_initial_population()