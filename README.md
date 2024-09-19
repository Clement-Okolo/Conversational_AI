# A Pipeline Fine-Tuning Framework for Task Oriented Dialogue (TOD) Systems 

![PyPI - Python Version](https://img.shields.io/pypi/pyversions/convlab) 
- [Unified Dataset and Models](#unified-dataset-and-models)
- [AWS Sagemaker Studio Environment](#aws-sagemaker-studio-environment)
- [Installation](#installation)
- [T5 Model Fine-Tuning](#t5-model-fine-tuning)
- [DDPT Model Fine-Tuning](#ddpt-model-fine-tuning)
- [Build TOD System Demo](#build-tod-system-demo)

## Unified Dataset and Models
The dataset used for this project is the multiwoz2.1 dataset, which is an open-source conversational data transformed into the unified format and stored in the `../data/unified_datasets` directory.

Also, the Text-to-Text Transfer (T5) and Dynamic Dialog Policy Transformer (DDPT) models used in this project are intergrated in ConvLab-3 toolkit and supports the unified data format.

## AWS Sagemaker Studio Environment

You will need aws Sagemaker `d-xujukvaykmqg` prod enviroment with `PyTorch 1.13 Python 3.9` GPU Optimized image, and the `ml.p3dn.24xlarge` instance for accelerated computing.

## Installation

First, you will install ConvLab-3 in the development mode.

Clone the latest repository:

```bash
git clone --depth 1 https://github.com/ConvLab/ConvLab-3.git
```

Install ConvLab-3 package using pip:

```bash
cd ConvLab-3 && pip install -e .
```
Note that aws sagemaker studio has okta authentication with times out after 7 hours. When this happens, you might need to re-install the package.


Install transformers library via pip:

```bash
pip install transformers
```

## T5 Model Fine-tuning

The workflow for fine-tuning the T5 model for natural language understanding (NLU), dialog state tracking (DST), and natural language generation (NLG) tasks include writing a `shell` script that contains the following sequence of commands:

1. Specify training hyperparameters
2. create data specific for task
3. run model training
4. evaluate trained model

Note `nlu/run_nlu-Copy1.sh`, `dst/run_dst-Copy1.sh`, and `nlg/run_nlg-Copy1.sh` are the shell scripts for NLU, DST and NLG tasks.

#### Set-up training hyperparameters
For each task, we you will need to specify a data path, learning rate, epochs, batch size etc. For an example, see `$task/run_$task-Copy1.sh`.

#### Create data for specific task
Run `create_data.py` script with corresponding arguments:
```bash
python ../create_data.py -t ${task_name} -d ${dataset_name} -s ${speaker} -c ${context_window_size}
```
This will automatically create a folder where the specific data is saved under `data/${task_name}/${dataset_name}/${speaker}/context_${context_window_size`.

#### Run model training
To train the model for each task, run the `run_seq2seq.py` with the corresponding arguments:
```
python ../run_seq2seq.py \
    --task_name ${task_name} \
    --train_file ${train_file} \
    --validation_file ${validation_file} \
    --source_column ${source_column} \
    --target_column ${target_column} \
    --max_source_length ${max_source_length} \
    --max_target_length ${max_target_length} \
    --truncation_side ${truncation_side} \
    --model_name_or_path ${model_name_or_path} \
    --do_train \
    --do_eval \
    --save_strategy epoch \
    --evaluation_strategy epoch \
    --save_total_limit 1 \
    --prediction_loss_only \
    --cache_dir ${cache_dir} \
    --output_dir ${output_dir} \
    --logging_dir ${logging_dir} \
    --overwrite_output_dir \
    --preprocessing_num_workers 4 \
    --per_device_train_batch_size ${per_device_train_batch_size} \
    --per_device_eval_batch_size ${per_device_eval_batch_size} \
    --gradient_accumulation_steps ${gradient_accumulation_steps} \
    --learning_rate ${lr} \
    --num_train_epochs ${num_train_epochs} \
    --optim adafactor \
    --gradient_checkpointing
```
This will create a fine-tuned T5 model, named `pytorch_model.bin`, and saved in the `output/${task_name}/${dataset_name}/${speaker}/context_${context_window_size}` directory.


Test trained model by runing `run_seq2seq.py` with corresponding arguments:
```
python ../run_seq2seq.py \
    --task_name ${task_name} \
    --test_file ${test_file} \
    --source_column ${source_column} \
    --target_column ${target_column} \
    --max_source_length ${max_source_length} \
    --max_target_length ${max_target_length} \
    --truncation_side ${truncation_side} \
    --model_name_or_path ${output_dir} \
    --do_predict \
    --predict_with_generate \
    --metric_name_or_path ${metric_name_or_path} \
    --cache_dir ${cache_dir} \
    --output_dir ${output_dir} \
    --logging_dir ${logging_dir} \
    --overwrite_output_dir \
    --preprocessing_num_workers 4 \
    --per_device_train_batch_size ${per_device_train_batch_size} \
    --per_device_eval_batch_size ${per_device_eval_batch_size} \
    --gradient_accumulation_steps ${gradient_accumulation_steps} \
    --learning_rate ${lr} \
    --num_train_epochs ${num_train_epochs} \
    --optim adafactor \
    --gradient_checkpointing
```
This will create a `test_generated_predictions.json` file, located under `output/${task_name}/${dataset_name}/${speaker}/context_${context_window_size}`, that will be merged with original data for unified evaluation.
 
Run `merge_predict_res.py` to merge test generated prediction results with original data for unified evaluation:
```
python merge_predict_res.py -d ${dataset_name} -s ${speaker} -c ${context_window_size} -p ${output_dir}/test_generated_predictions.json
```
This will create a `predictions.json` file that will be used for unified evaluation. 

The merge prediction results script for each tasks are located under `../../$task/merge_predict_res.py` directories.


#### Evaluate trained model
For unified evaluation of trained model, locate and run evaluation script for each task under `../../$task/evaluate_unified_datasets.py` directories:
```
python ../../../nlu/evaluate_unified_datasets.py -p ${output_dir}/predictions.json
```

#### Run shell scripts
To add execution permission to the scripts, run `chmod +x ../../$task/run_$task-Copy1.sh`, and then execute:
```
bash ../../nlu/run_nlu-Copy1.sh
bash ../../dst/run_dst-Copy1.sh
bash ../../nlg/run_nlg-Copy1.sh
```

##  DDPT Model Fine-Tuning
The workflow for fine-tuning a dialog policy model include:
1. pre-train the DDPT model on a dataset 
3. run reinforcement training
4. evaluate trained model

#### Pre-train DDPT model on a dataset 
This is also known as supervised training. Run `train_supervised.py` which is located in `../../vtrace_DPT/supervised/train_supervised.py` folder, with corresponding arguments:
```
$ python supervised/train_supervised.py --dataset_name=DATASET_NAME --seed=SEED
```
This will automatically create a **data, experiments,** and **processed** folders under `../../vtrace_DPT/supervised/`. The pre-trained model named `supervised.pol.mdl` will be saved in `../../experiments/experiments-TIMESTAMP/save` folder.

You can fine-tune the pre-trained DDPT model on a different dataset by specifying the path to the pre-trained model as shown below:

```
$ python supervised/train_supervised.py --dataset_name=DATASET_NAME --seed=SEED --model_path=""
```

#### Reinforcement learning training
To start RL training, you need to first set-up the environment configuration and policy parameters. The config files are located in `../../vtrace_DPT/configs/` folder and  include:
1. `RuleUser-Semantic-RuleDST.json`
2. `Multiwoz21_dpt.json`

 `RuleUser-Semantic-RuleDST.json` defines an environment for the policy with the rule-based dialogue state tracker  and `Multiwoz21_dpt.json` is where the training hyperparameters are specified.

 You need to specify the path to the pre-trained DDPT model and evaluation parameters in the  `RuleUser-Semantic-RuleDST.json`file.

Run `train.py` to execute reinforcement training with the corresponding arguments:
```
$ python train.py --config_name=RuleUser-Semantic-RuleDST --seed=SEED
```
Once RL training is finished, a **finished_experiment** folder will be automatically created with a corresponding **experiment-TIMESTAMP** folder in it.

#### Evaluate trained model
Execute `evaluate.py`, located in `convlab/policy/` folder, with corresponding arguments to run evaluations dialogs:
```
$ python evaluate.py --model_name=DDPT --config_path=`../../vtrace_DPT/configs/RuleUser-Semantic-RuleDST.json` --verbose
```

## Build TOD System Demo
To use the fine-tuned NLU, DST, policy and NLG models to build an interactive TOD system, we will follow the workflow below:
1. create a `.py` file
2. import modules and classes from ConvLab-3 libraries, along with other python libraries
3. create agent modules and interaction loop
4. run `.py` file in the terminal

```
# set-up environment
from convlab.base_models.t5.nlu import T5NLU
from convlab.base_models.t5.dst import T5DST
from convlab.base_models.t5.nlg import T5NLG
from convlab.policy.vector.vector_nodes import VectorNodes
from convlab.policy.vtrace_DPT import VTRACE
from convlab.dialog_agent import PipelineAgent, BiSession
from convlab.evaluator.multiwoz_eval import MultiWozEvaluator
from pprint import pprint
import random
import numpy as np
import torch

# create agent modules
def run_conversation():
    sys_nlu = T5NLU(speaker='user', 
                    context_window_size=${context_window_size}, 
                    model_name_or_path='../../output/nlu/multiwoz21/user/context_${context_window_size}')
    sys_dst = T5DST(dataset_name='multiwoz21', 
                    speaker='user', 
                    context_window_size=${context_window_size}, 
                    model_name_or_path='../../output/dst/multiwoz21/user/context_${context_window_size}')
    vectorizer = VectorNodes(dataset_name='multiwoz21',
                         use_masking=True,
                         manually_add_entity_names=True,
                         seed=0,
                         filter_state=True)
    sys_policy = VTRACE(is_train=False,
              seed=0,
              vectorizer=vectorizer,
              load_path="convlab/policy/vtrace_DPT/supervised")
    sys_nlg = T5NLG(speaker='system', 
                    context_window_size=${context_window_size}, 
                    model_name_or_path='../../output/nlg/multiwoz21/system/context_${context_window_size}')
    
    # assemble
    sys_agent = PipelineAgent(sys_nlu, sys_dst, sys_policy, sys_nlg, name='sys')

    # initialize session
    sys_agent.init_session()

    # interaction loop
    print("IVR: Hello, how can I help you today?)

    while True:
        user = input("Client: ")
        if 'exit' in user.lower():
            break

        # get system response
        sys = sys.agent.response(user)
        print(f"IVR: {sys}")
    print("This conversation has ended!")

if __name__ == '__main__':
    run_conversation()
```

For example, run `demo-bot.py` in terminal:

```
python demo-bot.py
```


# Reference
@article{zhu2022convlab3,
    title={ConvLab-3: A Flexible Dialogue System Toolkit Based on a Unified Data Format},
    author={Qi Zhu and Christian Geishauser and Hsien-chin Lin and Carel van Niekerk and Baolin Peng and Zheng Zhang and Michael Heck and Nurul Lubis and Dazhen Wan and Xiaochen Zhu and Jianfeng Gao and Milica Gašić and Minlie Huang},
    journal={arXiv preprint arXiv:2211.17148},
    year={2022},
    url={http://arxiv.org/abs/2211.17148}
}
