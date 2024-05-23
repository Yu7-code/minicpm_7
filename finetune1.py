import os
import glob
import json
import logging
from dataclasses import dataclass, field
from typing import Dict, Optional, List
import torch
from torch.utils.data import Dataset
import transformers
from trainer import CPMTrainer
import deepspeed
from deepspeed.runtime.zero.partition_parameters import ZeroParamStatus
from deepspeed import zero
from deepspeed import DeepSpeedConfig, DeepSpeedEngine

from dataset.dataset import SupervisedDataset
from lib.utils import load_config
from dataset.note_pair_dataset import NotePairDataset


from PIL import Image
from transformers import AutoModel, AutoTokenizer
from accelerate.utils import DistributedType

os.environ['PYTORCH_MAX_SPLIT_SIZE_MB'] = '100'

torch.set_printoptions(linewidth=200, threshold=3000, edgeitems=1000)
@dataclass
class ModelArguments:
    model_name_or_path: Optional[str] = field(default="openbmb/MiniCPM-V-2")


@dataclass
class DataArguments:
    data_path: str = field(
        default=None, metadata={"help": "Path to the training data."}
    )
    eval_data_path: str = field(
        default=None, metadata={"help": "Path to the evaluation data."}
    )
    lazy_preprocess: bool = False


@dataclass
class TrainingArguments(transformers.TrainingArguments):
    cache_dir: Optional[str] = field(default=None)
    optim: str = field(default="adamw_torch")
    gradient_checkpointing =  True
    model_max_length: int = field(
        default=2048,
        metadata={
            "help": "Maximum sequence length. Sequences will be right padded (and possibly truncated)."
        },
    )


def rank0_print(*args):
    if local_rank == 0:
        print(*args)

local_rank = 0

def data_collator(data):
    return data[0]

def train():
    global local_rank


    parser = transformers.HfArgumentParser(
        (ModelArguments, DataArguments, TrainingArguments)
    )

    (
        model_args,
        data_args,
        training_args,
    ) = parser.parse_args_into_dataclasses()
    print('model_args', model_args)
    
    if getattr(training_args, 'deepspeed', None):
        training_args.distributed_state.distributed_type = DistributedType.DEEPSPEED

    compute_dtype = (
        torch.float16
        if training_args.fp16
        else (torch.bfloat16 if training_args.bf16 else torch.float32)
    )

    local_rank = training_args.local_rank

    world_size = int(os.environ.get("WORLD_SIZE", 1))
    global_rank = int(os.environ.get("RANK", 0))
    ddp = world_size != 1
    device_map = {"": int(os.environ.get("LOCAL_RANK") or 0)} if ddp else None

    print('compute_dtype', compute_dtype)


    # print("Current memory allocated:", torch.cuda.memory_allocated())
    # print("Max memory allocated:", torch.cuda.max_memory_allocated())
    # 重置最大分配内存
    # torch.cuda.reset_max_memory_allocated()
    
    model = AutoModel.from_pretrained(model_args.model_name_or_path, trust_remote_code=True, torch_dtype=compute_dtype, device_map=device_map)
    
    # deepspeed
    ds_config_path = '/workspace/mnt/public/usr/yuzhiqi/minicpm-dev/ds_config_zero3.json'
    model, _, _, _ = deepspeed.initialize(model=model, config_params=ds_config_path)
    # print("Current memory allocated:", torch.cuda.memory_allocated())
    # print("Max memory allocated:", torch.cuda.max_memory_allocated())
    # 重置最大分配内存
    # torch.cuda.reset_max_memory_allocated()

    # 冻结llm.model.embed_tokens.weight
    model.llm.model.embed_tokens.weight.requires_grad = False
    # 冻结llm的block
    for layer_index in range(20):
        layer = model.llm.model.layers[layer_index]
        for param in layer.parameters():  
            param.requires_grad = False  
    # 冻结vpm.patch_embed.proj
    for name, param in model.vpm.patch_embed.named_parameters():
        if 'proj' in name:
            param.requires_grad = False
    # num_blocks_to_freeze = 24  # 冻结块数
    # for block_index in range(num_blocks_to_freeze):
    #     block = model.vpm.blocks[block_index]  
    #     for param in block.parameters():  
    #         param.requires_grad = False  
    # 确认参数是否成功冻结
    # print("embed_tokens.weight requires_grad:", model.llm.model.embed_tokens.weight.requires_grad)
    # for name, param in model.vpm.patch_embed.named_parameters():
    #     if 'proj' in name:
    #         print(f"{name} requires_grad:", param.requires_grad)
    tokenizer = AutoTokenizer.from_pretrained(model_args.model_name_or_path, trust_remote_code=True)

    args  = load_config('/workspace/mnt/public/usr/yuzhiqi/minicpm-dev/data_config.yaml')
    args.train_batch_size = args.batch_size
    train_dataset = NotePairDataset(args, global_rank, world_size, tokenizer, 2048, model.transform, model.config.__dict__, mode='train')
    eval_dataset = NotePairDataset(args, global_rank, world_size, tokenizer, 2048, model.transform, model.config.__dict__, mode='eval')
    #Load data
    #data_module = make_supervised_data_module(
    #    tokenizer=tokenizer, data_args=data_args, transform=model.transform,  data_collator=data_collator, slice_config=model.config.__dict__,
    #)
    # training_args.fp16=True
    training_args.gradient_checkpointing = True
    # model = model.to(torch.float16)
    model = model.to(torch.bfloat16)
    print('training_args.gradient_checkpointing',training_args.gradient_checkpointing)
    print('training_args.fp16', training_args.fp16)

    
    

    trainer = CPMTrainer(
        model=model,
        tokenizer=tokenizer,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=data_collator,
    )
    
    trainer.train()
    # trainer.evaluate() #此时显存占用在9G左右
    trainer.save_state()


if __name__ == "__main__":
    train()
