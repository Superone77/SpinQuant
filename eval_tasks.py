import argparse
import torch
from transformers import LlamaTokenizerFast
from eval_utils.modeling_llama import LlamaForCausalLM
from lm_eval import evaluator


def load_model(model_path: str, base_model: str):
    config = LlamaForCausalLM.from_pretrained(base_model).config
    model = LlamaForCausalLM.from_pretrained(base_model, config=config)
    state_dict = torch.load(model_path, map_location="cpu")
    model.load_state_dict(state_dict)
    model.to("cuda")
    tokenizer = LlamaTokenizerFast.from_pretrained(base_model)
    tokenizer.pad_token = tokenizer.eos_token
    return model, tokenizer


def main():
    parser = argparse.ArgumentParser(description="Evaluate saved SpinQuant model")
    parser.add_argument("--model_path", type=str, required=True, help="Path to saved model state_dict")
    parser.add_argument("--base_model", type=str, required=True, help="Base model identifier")
    parser.add_argument(
        "--tasks",
        type=str,
        default="arc_challenge,arc_easy,boolq,hellaswag,openbookqa,piqa,winogrande,mmlu",
        help="Comma separated list of tasks to evaluate",
    )
    args = parser.parse_args()

    model, tokenizer = load_model(args.model_path, args.base_model)

    task_list = [t.strip() for t in args.tasks.split(",")]
    results = evaluator.simple_evaluate(model=model, tokenizer=tokenizer, tasks=task_list)
    print(results)


if __name__ == "__main__":
    main()
