import json
import sys
import os

# Add the current directory to sys.path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import common
from mgsm_eval import MGSMEval
from groq_sampler import GroqChatCompletionSampler

def main():
    debug = True
    samplers = {
        "groq_llama3.1_8b_instant": GroqChatCompletionSampler(),
    }

    def get_evals(eval_name):
        match eval_name:
            case "mgsm":
                return MGSMEval(num_examples_per_lang=10 if debug else 250)
            case _:
                raise Exception(f"Unrecognized eval type: {eval_name}")

    evals = {"mgsm": get_evals("mgsm")}
    print(evals)
    debug_suffix = "_DEBUG" if debug else ""
    mergekey2resultpath = {}
    for sampler_name, sampler in samplers.items():
        for eval_name, eval_obj in evals.items():
            result = eval_obj(sampler)
            file_stem = f"{eval_name}_{sampler_name}"
            report_filename = f"/tmp/{file_stem}{debug_suffix}.html"
            print(f"Writing report to {report_filename}")
            with open(report_filename, "w") as fh:
                fh.write(common.make_report(result))
            metrics = result.metrics | {"score": result.score}
            print(metrics)
            result_filename = f"/tmp/{file_stem}{debug_suffix}.json"
            with open(result_filename, "w") as f:
                f.write(json.dumps(metrics, indent=2))
            print(f"Writing results to {result_filename}")
            mergekey2resultpath[f"{file_stem}"] = result_filename
    
    # The rest of the function remains the same
    # ...

if __name__ == "__main__":
    main()
