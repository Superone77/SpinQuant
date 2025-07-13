# coding=utf-8
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import datetime
from logging import Logger
import os, json, argparse, pathlib
from typing import List, Dict, Tuple

import torch
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.distributed as dist
from transformers import LlamaTokenizerFast
import transformers
from eval_utils.main import ptq_model
from eval_utils.modeling_llama import LlamaForCausalLM
from utils import data_utils, eval_utils, utils
from utils.process_args import process_args_ptq
from analysis_utils import *
from grouplayernorm import convert_rms_to_group
log: Logger = utils.get_logger("spinquant")
from utils import data_utils, eval_utils, utils











def train() -> None:
    # dist.init_process_group(backend="nccl", timeout=datetime.timedelta(hours=8))
    model_args, training_args, ptq_args = process_args_ptq()
    # local_rank = utils.get_local_rank()
    local_rank = 0
    compare_list = ["baseline", "grouprms","spinquant"]# "spinquant","baseline", "grouprms"
    wiki_ppl = False
    # log.info("the rank is {}".format(local_rank))
    # torch.distributed.barrier()
    from lm_eval import models
    # lm = models.get_model("hf-causal-experimental").create_from_arg_string(
    #         "pretrained=/local/mnt/workspace/wanqi/llama/modelscope/Llama-2-7b-ms", {"batch_size": 1, "max_batch_size": 1, "device": "cuda"}
    #     )
    config = transformers.AutoConfig.from_pretrained(
        model_args.input_model, token=model_args.access_token
    )
    config.num_hidden_layers = 6
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
    # Llama v3.2 specific: Spinquant is not compatiable with tie_word_embeddings, clone lm_head from embed_tokens
    process_word_embeddings = False
    if config.tie_word_embeddings:
        config.tie_word_embeddings = False
        process_word_embeddings = True
    dtype = torch.bfloat16 if training_args.bf16 else torch.float16
    if "spinquant" in compare_list:
        print("spinquant")
        model_rot = LlamaForCausalLM.from_pretrained(
            pretrained_model_name_or_path=model_args.input_model,
            config=config,
            torch_dtype=dtype,
            token=model_args.access_token,
        )
        if process_word_embeddings:
            model_rot.lm_head.weight.data = model_rot.model.embed_tokens.weight.data.clone()
        model_rot.cuda()
        from lm_eval import evaluator
        from lm_eval.models.huggingface import HFLM
        model_rot = ptq_model(ptq_args, model_rot, model_args)
        model_rot.seqlen = training_args.model_max_length
        model_rot.config.use_cache = False
        if local_rank == 0:
            log.info("Model PTQ completed {}".format(model_rot))
        if wiki_ppl:
            testloader = data_utils.get_wikitext2(
                seed=ptq_args.seed,
                seqlen=2048,
                tokenizer=tokenizer,
                eval_mode=True,
            )

            dataset_ppl = eval_utils.evaluator(model_rot, testloader, utils.DEV, ptq_args)
            log.info("wiki2 ppl is: {}".format(dataset_ppl))
        
            
    if "baseline" in compare_list:
        print("baseline")
        model_norot = LlamaForCausalLM.from_pretrained(
            pretrained_model_name_or_path=model_args.input_model,
            config=config,
            torch_dtype=dtype,
            token=model_args.access_token,
        )
        model_norot.cuda()
        from utils import fuse_norm_utils
        fuse_norm_utils.fuse_layer_norms(model_norot)
        
        model_norot.config.use_cache = False
        model_norot.seqlen = training_args.model_max_length
        if wiki_ppl:
            testloader = data_utils.get_wikitext2(
                seed=ptq_args.seed,
                seqlen=2048,
                tokenizer=tokenizer,
                eval_mode=True,
            )

            dataset_ppl = eval_utils.evaluator(model_norot, testloader, utils.DEV, ptq_args)
            log.info("wiki2 ppl is: {}".format(dataset_ppl))
        
    if "grouprms" in compare_list:
        print("grouprms")
        model_groupnrom = LlamaForCausalLM.from_pretrained(
            pretrained_model_name_or_path="/local/mnt/workspace/wanqi/8bit_optimizer/remote/SpinQuant/llama3_gln/checkpoint-500",
            config=config,
            torch_dtype=dtype,
            token=model_args.access_token,
        )
        convert_rms_to_group(model_groupnrom, 16)
        model_groupnrom.cuda()
        model_groupnrom.seqlen = training_args.model_max_length
        model_groupnrom.config.use_cache = False
        if wiki_ppl:
            testloader = data_utils.get_wikitext2(
                seed=ptq_args.seed,
                seqlen=2048,
                tokenizer=tokenizer,
                eval_mode=True,
            )

            dataset_ppl = eval_utils.evaluator(model_groupnrom, testloader, utils.DEV, ptq_args)
            log.info("wiki2 ppl is: {}".format(dataset_ppl))
        

    log.info("Start to load tokenizer...")
    
    
    batch_inputs = get_wikitext_batch(
        tokenizer,
        seq_len=model_norot.seqlen,
        batch_size=8,
        device="cuda",
    )
    layer_idx = 5
    target_layers = ["k_proj", "q_proj", "v_proj", "o_proj","gate_proj","up_proj","down_proj"]
    out_dir = "/local/mnt/workspace/wanqi/tmp/activation_compare_llama3_spinquant"
    npy_out_dir = out_dir+"/npy_data"
    out_dir = pathlib.Path(out_dir)
    npy_out_dir = pathlib.Path(npy_out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    npy_out_dir.mkdir(parents=True, exist_ok=True)
    for layer_sub in target_layers:
        print(f"\n>>> Processing target: {layer_sub}")
        means_all_act_in, kurts_all_act_in, labels = [], [], []
        means_all_weight, kurts_all_weight= [], []
        means_all_act_out, kurts_all_act_out= [], []
        mse_list_act_in, mse_list_act_out,mse_list_weight = [],[],[]
        for group_size in [1024,256,64,16]:
            if "baseline" in compare_list:
                label = f"baseline_groupsize{group_size}"
                in_act, out_act,weight = capture_nth_layer_act(model_norot, layer_sub, layer_idx, batch_inputs)
                np.save(npy_out_dir / f"in_act_{layer_sub}_{label}.npy",in_act.to(torch.float32).numpy())
                np.save(npy_out_dir / f"out_act_{layer_sub}_{label}.npy",out_act.to(torch.float32).numpy())
                np.save(npy_out_dir / f"weight_{layer_sub}_{label}.npy",out_act.to(torch.float32).numpy())
                
                if os.path.exists(npy_out_dir / f"mse_in_act_{layer_sub}_{label}.npy"):
                    mse_list_act_in.append(np.load(npy_out_dir / f"mse_in_act_{layer_sub}_{label}.npy"))
                    mse_list_act_out.append(np.load(npy_out_dir / f"mse_out_act_{layer_sub}_{label}.npy"))
                    mse_list_weight.append(np.load(npy_out_dir / f"mse_weight_{layer_sub}_{label}.npy"))
                else:
                    (q_in_act, metrics_in_act), (q_out_act, metrics_out_act), (q_weight, metrics_weight) = parallel_quantization(in_act, out_act, weight, group_size)
                    mse_list_act_in.append(np.array(metrics_in_act["mses"]))
                    mse_list_act_out.append(np.array(metrics_out_act["mses"]))
                    mse_list_weight.append(np.array(metrics_weight["mses"]))
                    np.save(npy_out_dir / f"mse_in_act_{layer_sub}_{label}.npy",metrics_in_act["mses"])
                    np.save(npy_out_dir / f"mse_out_act_{layer_sub}_{label}.npy",metrics_out_act["mses"])
                    np.save(npy_out_dir / f"mse_weight_{layer_sub}_{label}.npy",metrics_weight["mses"])
                
                means, kurts = analysis_within_group(in_act, group_size)
                means_all_act_in.append(means.to(torch.float32).numpy())
                kurts_all_act_in.append(kurts.to(torch.float32).numpy())
                np.save(npy_out_dir / f"kurts_in_act_{layer_sub}_{label}.npy",kurts.to(torch.float32).numpy())
                np.save(npy_out_dir / f"means_in_act_{layer_sub}_{label}.npy",means.to(torch.float32).numpy())

                means, kurts = analysis_within_group(out_act, group_size)
                means_all_act_out.append(means.to(torch.float32).numpy())
                kurts_all_act_out.append(kurts.to(torch.float32).numpy())
                np.save(npy_out_dir / f"kurts_out_act_{layer_sub}_{label}.npy",kurts.to(torch.float32).numpy())
                np.save(npy_out_dir / f"means_out_act_{layer_sub}_{label}.npy",means.to(torch.float32).numpy())
                means, kurts = analysis_within_group(weight, group_size)
                means_all_weight.append(means.to(torch.float32).numpy())
                kurts_all_weight.append(kurts.to(torch.float32).numpy())
                np.save(npy_out_dir / f"kurts_weight_{layer_sub}_{label}.npy",kurts.to(torch.float32).numpy())
                np.save(npy_out_dir / f"means_weight_{layer_sub}_{label}.npy",means.to(torch.float32).numpy())
                labels.append(label)
                


            if "spinquant" in compare_list:
                label = f"afterspinquant_groupsize{group_size}"
                in_act, out_act,weight = capture_nth_layer_act(model_rot, layer_sub, layer_idx, batch_inputs)
                np.save(npy_out_dir / f"in_act_{layer_sub}_{label}.npy",in_act.to(torch.float32).numpy())
                np.save(npy_out_dir / f"out_act_{layer_sub}_{label}.npy",out_act.to(torch.float32).numpy())
                np.save(npy_out_dir / f"weight_{layer_sub}_{label}.npy",out_act.to(torch.float32).numpy())
                
                
                if os.path.exists(npy_out_dir / f"mse_in_act_{layer_sub}_{label}.npy"):
                    mse_list_act_in.append(np.load(npy_out_dir / f"mse_in_act_{layer_sub}_{label}.npy"))
                    mse_list_act_out.append(np.load(npy_out_dir / f"mse_out_act_{layer_sub}_{label}.npy"))
                    mse_list_weight.append(np.load(npy_out_dir / f"mse_weight_{layer_sub}_{label}.npy"))
                else:
                    (q_in_act, metrics_in_act), (q_out_act, metrics_out_act), (q_weight, metrics_weight) = parallel_quantization(in_act, out_act, weight, group_size)
                    mse_list_act_in.append(np.array(metrics_in_act["mses"]))
                    mse_list_act_out.append(np.array(metrics_out_act["mses"]))
                    mse_list_weight.append(np.array(metrics_weight["mses"]))
                    np.save(npy_out_dir / f"mse_in_act_{layer_sub}_{label}.npy",metrics_in_act["mses"])
                    np.save(npy_out_dir / f"mse_out_act_{layer_sub}_{label}.npy",metrics_out_act["mses"])
                    np.save(npy_out_dir / f"mse_weight_{layer_sub}_{label}.npy",metrics_weight["mses"])

                means, kurts = analysis_within_group(in_act, group_size)
                means_all_act_in.append(means.to(torch.float32).numpy())
                kurts_all_act_in.append(kurts.to(torch.float32).numpy())
                np.save(npy_out_dir / f"kurts_in_act_{layer_sub}_{label}.npy",kurts.to(torch.float32).numpy())
                np.save(npy_out_dir / f"means_in_act_{layer_sub}_{label}.npy",means.to(torch.float32).numpy())

                means, kurts = analysis_within_group(out_act, group_size)
                means_all_act_out.append(means.to(torch.float32).numpy())
                kurts_all_act_out.append(kurts.to(torch.float32).numpy())
                np.save(npy_out_dir / f"kurts_out_act_{layer_sub}_{label}.npy",kurts.to(torch.float32).numpy())
                np.save(npy_out_dir / f"means_out_act_{layer_sub}_{label}.npy",means.to(torch.float32).numpy())
                means, kurts = analysis_within_group(weight, group_size)
                means_all_weight.append(means.to(torch.float32).numpy())
                kurts_all_weight.append(kurts.to(torch.float32).numpy())
                np.save(npy_out_dir / f"kurts_weight_{layer_sub}_{label}.npy",kurts.to(torch.float32).numpy())
                np.save(npy_out_dir / f"means_weight_{layer_sub}_{label}.npy",means.to(torch.float32).numpy())
                labels.append(label)

            if "grouprms" in compare_list:
                label = f"groupnorm_groupsize{group_size}"
                in_act, out_act,weight = capture_nth_layer_act(model_groupnrom, layer_sub, layer_idx, batch_inputs)
                np.save(npy_out_dir / f"in_act_{layer_sub}_{label}.npy",in_act.to(torch.float32).numpy())
                np.save(npy_out_dir / f"out_act_{layer_sub}_{label}.npy",out_act.to(torch.float32).numpy())
                np.save(npy_out_dir / f"weight_{layer_sub}_{label}.npy",out_act.to(torch.float32).numpy())
                
                if os.path.exists(npy_out_dir / f"mse_in_act_{layer_sub}_{label}.npy"):
                    mse_list_act_in.append(np.load(npy_out_dir / f"mse_in_act_{layer_sub}_{label}.npy"))
                    mse_list_act_out.append(np.load(npy_out_dir / f"mse_out_act_{layer_sub}_{label}.npy"))
                    mse_list_weight.append(np.load(npy_out_dir / f"mse_weight_{layer_sub}_{label}.npy"))
                else:
                    (q_in_act, metrics_in_act), (q_out_act, metrics_out_act), (q_weight, metrics_weight) = parallel_quantization(in_act, out_act, weight, group_size)
                    mse_list_act_in.append(np.array(metrics_in_act["mses"]))
                    mse_list_act_out.append(np.array(metrics_out_act["mses"]))
                    mse_list_weight.append(np.array(metrics_weight["mses"]))
                    np.save(npy_out_dir / f"mse_in_act_{layer_sub}_{label}.npy",metrics_in_act["mses"])
                    np.save(npy_out_dir / f"mse_out_act_{layer_sub}_{label}.npy",metrics_out_act["mses"])
                    np.save(npy_out_dir / f"mse_weight_{layer_sub}_{label}.npy",metrics_weight["mses"])
                
                means, kurts = analysis_within_group(in_act, group_size)
                means_all_act_in.append(means.to(torch.float32).numpy())
                kurts_all_act_in.append(kurts.to(torch.float32).numpy())
                np.save(npy_out_dir / f"kurts_in_act_{layer_sub}_{label}.npy",kurts.to(torch.float32).numpy())
                np.save(npy_out_dir / f"means_in_act_{layer_sub}_{label}.npy",means.to(torch.float32).numpy())

                means, kurts = analysis_within_group(out_act, group_size)
                means_all_act_out.append(means.to(torch.float32).numpy())
                kurts_all_act_out.append(kurts.to(torch.float32).numpy())
                np.save(npy_out_dir / f"kurts_out_act_{layer_sub}_{label}.npy",kurts.to(torch.float32).numpy())
                np.save(npy_out_dir / f"means_out_act_{layer_sub}_{label}.npy",means.to(torch.float32).numpy())
                means, kurts = analysis_within_group(weight, group_size)
                means_all_weight.append(means.to(torch.float32).numpy())
                kurts_all_weight.append(kurts.to(torch.float32).numpy())
                np.save(npy_out_dir / f"kurts_weight_{layer_sub}_{label}.npy",kurts.to(torch.float32).numpy())
                np.save(npy_out_dir / f"means_weight_{layer_sub}_{label}.npy",means.to(torch.float32).numpy())
                labels.append(label)

        # ---------- plot ----------
        plot_box_n(
            means_all_act_in,
            labels,
            out_dir / f"act_in_mean_{layer_sub}.png",
        )
        
        plot_box_n(
            kurts_all_act_in,
            labels,
            out_dir / f"act_in_kurt_{layer_sub}.png",
        )
        
        plot_box_n(
            mse_list_act_in,
            labels,
            out_dir / f"act_in_mse_{layer_sub}.png",
        )
        

        plot_box_n(
            means_all_act_out,
            labels,
            out_dir / f"act_out_mean_{layer_sub}.png",
        )
        
        plot_box_n(
            kurts_all_act_out,
            labels,
            out_dir / f"act_out_kurt_{layer_sub}.png",
        )
        print(len(labels))
        print(labels)
        print(len(mse_list_act_out))
        plot_box_n(
            mse_list_act_out,
            labels,
            out_dir / f"act_out_mse_{layer_sub}.png",
        )
        

        plot_box_n(
            means_all_weight,
            labels,
            out_dir / f"weight_mean_{layer_sub}.png",
        )
        
        plot_box_n(
            kurts_all_weight,
            labels,
            out_dir / f"weight_kurt_{layer_sub}.png",
        )
        
        plot_box_n(
            mse_list_weight,
            labels,
            out_dir / f"weight_mse_{layer_sub}.png",
        )
        
        print("Saved boxplots for", layer_sub)
        
    # dist.barrier()


def analysis_npy():
    import numpy as np

    input_dir = "/local/mnt/workspace/wanqi/tmp/activation_compare_llama3_spinquant/npy_data"
    output_dir = "/local/mnt/workspace/wanqi/tmp/activation_compare_llama3_spinquant/cross_layer_plot"
    group_size_list = [1024,16]
    target_layers = ["k_proj", "q_proj", "v_proj", "o_proj","gate_proj","up_proj","down_proj"]
    tag_list = ["in_act", "out_act", "weight"]
    label_list = ["baseline", "groupnorm"]
    metrics_list = ["kurts","means","mse"]
    for metric in metrics_list:
        for tag in tag_list:
            for group_size in group_size_list:
                plot_data = []
                plot_label = []
                for target_layer in target_layers:
                    for label in label_list:
                    
                        data = np.load(f"{input_dir}/{metric}_{tag}_{target_layer}_{label}_groupsize{group_size}.npy")
                        plot_data.append(data)
                        plot_label.append(f"{target_layer}_{label}")
                        
                plot_box_n(plot_data, plot_label, f"{output_dir}/{metric}_{tag}_groupsize{group_size}.png")
                print(f"save {metric}_{tag}_groupsize{group_size}.png")
if __name__ == "__main__":
    train()
