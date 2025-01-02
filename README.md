## Overview

- token reduction算法的paddle实现

- 目前适配llama，算法见paddlenlp/llama/modeling.py  paddlenlp/llama/utils.py



## Fine-tuning

`python -u -m paddle.distributed.launch --gpus "0,1,2,3" run_finetune.py ./config/llama/lora_argument.json`