token reduction算法的paddle实现
目前适配llama，主要修改见paddlenlp/llama/modeling.py  paddlenlp/llama/utils.py



微调脚本: python -u -m paddle.distributed.launch --gpus "0,1,2,3" run_finetune.py ./config/llama/lora_argument.json