# coding=utf-8
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import datetime
from logging import Logger

import torch
import torch.distributed as dist
from transformers import LlamaTokenizerFast
import transformers
from eval_utils.main import ptq_model
from eval_utils.modeling_llama import LlamaForCausalLM
from utils import data_utils, eval_utils, utils
from utils.process_args import process_args_ptq

log: Logger = utils.get_logger("spinquant")


# def train() -> None:
#     dist.init_process_group(backend="nccl", timeout=datetime.timedelta(hours=8))
#     model_args, training_args, ptq_args = process_args_ptq()
#     local_rank = utils.get_local_rank()

#     log.info("the rank is {}".format(local_rank))
#     torch.distributed.barrier()

#     config = transformers.AutoConfig.from_pretrained(
#         model_args.input_model, token=model_args.access_token
#     )
#     # Llama v3.2 specific: Spinquant is not compatiable with tie_word_embeddings, clone lm_head from embed_tokens
#     process_word_embeddings = False
#     if config.tie_word_embeddings:
#         config.tie_word_embeddings = False
#         process_word_embeddings = True
#     dtype = torch.bfloat16 if training_args.bf16 else torch.float16
#     model = LlamaForCausalLM.from_pretrained(
#         pretrained_model_name_or_path=model_args.input_model,
#         config=config,
#         torch_dtype=dtype,
#         token=model_args.access_token,
#     )
#     if process_word_embeddings:
#         model.lm_head.weight.data = model.model.embed_tokens.weight.data.clone()
#     model.cuda()

#     model = ptq_model(ptq_args, model, model_args)
#     model.seqlen = training_args.model_max_length
#     if local_rank == 0:
#         log.info("Model PTQ completed {}".format(model))
#         log.info("Start to load tokenizer...")
#     tokenizer = LlamaTokenizerFast.from_pretrained(
#         pretrained_model_name_or_path=model_args.input_model,
#         cache_dir=training_args.cache_dir,
#         model_max_length=training_args.model_max_length,
#         padding_side="right",
#         use_fast=True,
#         add_eos_token=False,
#         add_bos_token=False,
#         token=model_args.access_token,
#     )
#     log.info("Complete tokenizer loading...")
#     model.config.use_cache = False

#     testloader = data_utils.get_wikitext2(
#         seed=ptq_args.seed,
#         seqlen=2048,
#         tokenizer=tokenizer,
#         eval_mode=True,
#     )

#     dataset_ppl = eval_utils.evaluator(model, testloader, utils.DEV, ptq_args)
#     log.info("wiki2 ppl is: {}".format(dataset_ppl))
#     from lm_eval.evaluator import evaluate
#     from lm_eval import tasks
#     task_dict = tasks.get_task_dict(["arc_challenge","arc_easy"])


#     results = evaluate(
#         lm=model,
#         task_dict=task_dict,
#         num_fewshot=0
#     )
#     log.info(results)
#     dist.barrier()


def make_table(result_dict):
    """Generate table of results."""
    from pytablewriter import MarkdownTableWriter, LatexTableWriter

    md_writer = MarkdownTableWriter()
    latex_writer = LatexTableWriter()
    md_writer.headers = ["Task", "Version", "Metric", "Value", "", "Stderr"]
    latex_writer.headers = ["Task", "Version", "Metric", "Value", "", "Stderr"]

    values = []

    for k, dic in result_dict["results"].items():
        version = result_dict["versions"][k]
        for m, v in dic.items():
            if m.endswith("_stderr"):
                continue

            if m + "_stderr" in dic:
                se = dic[m + "_stderr"]
                values.append([k, version, m,  v, "±",  se])
            else:
                print()
                values.append([k, version, m,  v, "", ""])
            k = ""
            version = ""
    md_writer.value_matrix = values
    latex_writer.value_matrix = values

    # todo: make latex table look good
    # print(latex_writer.dumps())

    return md_writer.dumps()

def train() -> None:
    dist.init_process_group(backend="nccl", timeout=datetime.timedelta(hours=8))
    model_args, training_args, ptq_args = process_args_ptq()
    local_rank = utils.get_local_rank()

    log.info("the rank is {}".format(local_rank))
    torch.distributed.barrier()
    from lm_eval import models
    # lm = models.get_model("hf-causal-experimental").create_from_arg_string(
    #         "pretrained=/local/mnt/workspace/wanqi/llama/modelscope/Llama-2-7b-ms", {"batch_size": 1, "max_batch_size": 1, "device": "cuda"}
    #     )
    config = transformers.AutoConfig.from_pretrained(
        model_args.input_model, token=model_args.access_token
    )
    # Llama v3.2 specific: Spinquant is not compatiable with tie_word_embeddings, clone lm_head from embed_tokens
    process_word_embeddings = False
    if config.tie_word_embeddings:
        config.tie_word_embeddings = False
        process_word_embeddings = True
    dtype = torch.bfloat16 if training_args.bf16 else torch.float16
    model = LlamaForCausalLM.from_pretrained(
        pretrained_model_name_or_path=model_args.input_model,
        config=config,
        torch_dtype=dtype,
        token=model_args.access_token,
    )
    if process_word_embeddings:
        model.lm_head.weight.data = model.model.embed_tokens.weight.data.clone()
    model.cuda()

    from lm_eval import evaluator
    from lm_eval.models.huggingface import HFLM
    lm = HFLM(pretrained=model,batch_size=32,device="cuda")
    _ = ptq_model(ptq_args, lm.model, model_args)
    lm.model.seqlen = training_args.model_max_length
    if local_rank == 0:
        log.info("Model PTQ completed {}".format(lm.model))
        log.info("Start to load tokenizer...")
    tokenizer = LlamaTokenizerFast.from_pretrained(
        pretrained_model_name_or_path=model_args.input_model,
        cache_dir=training_args.cache_dir,
        model_max_length=training_args.model_max_length,
        padding_side="right",
        use_fast=True,
        add_eos_token=False,
        add_bos_token=False,
        token=model_args.access_token,
    )
    log.info("Complete tokenizer loading...")
    lm.model.config.use_cache = False
    # results_0 = evaluator.simple_evaluate(model=lm,tasks=["arc_challenge","arc_easy"],batch_size = 32,device="cuda")
    # # log.info(results)
    # # log.info(make_table(results))
    # results_1 = evaluator.simple_evaluate(model=lm,tasks=["boolq","hellaswag","openbookqa"],batch_size = 32,device="cuda")
    # # log.info(results)
    # # log.info(make_table(results))
    # results_2 = evaluator.simple_evaluate(model=lm,tasks=["piqa"],batch_size = 32,device="cuda")
    # # log.info(results)
    # # log.info(make_table(results))
    # results_3 = evaluator.simple_evaluate(model=lm,tasks=["winogrande"],batch_size = 32,device="cuda")
    # # log.info(results)
    # log.info(make_table(results_0))
    # log.info(make_table(results_1))
    # log.info(make_table(results_2))
    # log.info(make_table(results_3))
    

    testloader = data_utils.get_wikitext2(
        seed=ptq_args.seed,
        seqlen=2048,
        tokenizer=tokenizer,
        eval_mode=True,
    )

    dataset_ppl = eval_utils.evaluator(lm.model, testloader, utils.DEV, ptq_args)
    log.info("wiki2 ppl is: {}".format(dataset_ppl))
    # from lm_eval.evaluator import evaluate,make_table
    # from lm_eval import tasks
    # task_dict = tasks.get_task_dict(["arc_challenge","arc_easy"])

    # # lm.model.cpu()
    # results = evaluate(
    #     lm=lm,
    #     task_dict=task_dict,
    #     num_fewshot=0
    # )
    # log.info(make_table(results))
    dist.barrier()

if __name__ == "__main__":
    train()
