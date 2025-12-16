from vllm import LLM
from vllm.sampling_params import SamplingParams
import pandas as pd
import argparse

from lmformatenforcer import CharacterLevelParser
from lmformatenforcer.integrations.vllm import build_vllm_logits_processor, build_vllm_token_enforcer_tokenizer_data
from lmformatenforcer import JsonSchemaParser

from typing import Union, List, Optional
from pydantic import BaseModel
class AnswerFormat(BaseModel):
     Extracted_Reflections: List[str]
     SUMMARY: str

sampling_params = SamplingParams(temperature=0.3, max_tokens=8000)

import os 
ListOrStrList = Union[str, List[str]]

os.environ['TRANSFORMERS_CACHE'] = "/ix1/dlitman/yaz118/.cache"
os.environ['HF_HOME'] = "/ix1/dlitman/yaz118/.cache"
os.environ['HF_DATASETS_CACHE'] = "/ix1/dlitman/yaz118/.cache"

def vllm_with_character_level_parser(llm, prompt: ListOrStrList, parser: Optional[CharacterLevelParser] = None, tokenizer_data=None) -> ListOrStrList:
    
    sampling_params = SamplingParams(temperature=0.3, max_tokens=8000)
    if parser:
        logits_processor = build_vllm_logits_processor(tokenizer_data, parser)
        sampling_params.logits_processors = [logits_processor]
    # Note on batched generation:
    # For some reason, I achieved better batch performance by manually adding a loop similar to this:
    # https://github.com/vllm-project/vllm/blob/main/examples/llm_engine_example.py,
    # I don't know why this is faster than simply calling llm.generate() with a list of prompts, but it is from my tests.
    # However, this demo focuses on simplicity, so I'm not including that here.
    #results = llm.generate(prompt, sampling_params=sampling_params)
    results = llm.chat(messages=prompt,
                   sampling_params=sampling_params,
                   use_tqdm=True)
    if isinstance(prompt, str):
        return results[0].outputs[0].text
    else:
        return [result.outputs[0].text for result in results]

def construct_prompt(max_len, aspect, reflections):
    converted = {i:j for i,j in enumerate(reflections.split("\n"))}
    conversation = [
    {
        "role": "system",
        "content": '''You are a TA for a undegraduate-level course, you are given a collection of student reflections after taking one lecture and tasked to write a summary to present to the instructor'''
    },
    {
        "role": "user",
        "content": f'''Given the students' reponses, and a focused topic {aspect}, create a short summary using your own words (no more than {max_len} words). The summary needs to be a coherent paragraph and should include the major points. The summary should focus on the provided topic only, contain only information about reflections, and avoid adding irrelevent sentences or suggestions such as 'make sure to bring this up in next class', or 'Consider this for future lectures', etc... 
 
        You are tasked to perform this task in two steps:
        (1) Extract the list of indexes and students' reflections in the given REFLECTIONs  that are relevant to the focused topic.
        (2) summarize them into a short summary using your own words (no more htan {max_len} words). 

        REFLECTIONS: {converted}

        FOCUSED TOPIC: {aspect} 

       your repsonse should be this json format: {{'Extracted_Reflections' : [your extracted data] (i.e. [STUDENT_REFLECTION_1_TEXT, STUDENT_REFLECTION_2_TEXT, ...]), 'SUMMARY': [your response]}} 
'''
    },
]
    return conversation

def main():
    parser = argparse.ArgumentParser(description="Script to configure model choice and max generated length.")
    
    # Adding arguments
    parser.add_argument(
        "--model", 
        type=str, 
        required=True, 
        help="Specify the model choice. Example: 'bert', 'gpt', 'transformer', etc."
    )
    parser.add_argument(
        "--max_length", 
        type=int, 
        default=100, 
        help="Specify the maximum generated length of the output. Default is 50."
    )
    
    parser.add_argument(
        "--run",
        type=int,
        default=0,
        help="Specify the maximum generated length of the output. Default is 50."
    )
    parser.add_argument(
        "--num_card",
        type=int,
        default=1,
        help="Specify the maximum generated length of the output. Default is 50."
    )

    args = parser.parse_args()
    model_choice = args.model
    max_len = args.max_length
    card_number = args.num_card 
    run_idx = args.run

    model_mapping = {'llama2': 'meta-llama/Llama-2-13b-chat-hf',
  		     'llama3': 'meta-llama/Meta-Llama-3-8B-Instruct',
 		     'llama3.1': 'meta-llama/Llama-3.1-8B-Instruct',
'llama3.1-70': 'neuralmagic/Meta-Llama-3.1-70B-Instruct-quantized.w8a8',
'nvidia': 'nvidia/Llama-3.1-Nemotron-70B-Instruct-HF',
    		     'mistral': 'mistralai/Mistral-NeMo-Instruct-2407'}
    model_name = model_mapping.get(model_choice, "NOT FOUND")
    if model_choice == 'mistral':
        llm = LLM(
    model=model_name,
    gpu_memory_utilization=0.90,
    max_model_len=10268,
tokenizer_mode="mistral",
    load_format="mistral",
    config_format="mistral", download_dir='/ix1/dlitman/yaz118/LLM_ckpts/',  tensor_parallel_size=card_number
)
    elif model_choice == 'nvidia':
        llm = LLM(model=model_name, tensor_parallel_size=card_number, download_dir='/ix1/dlitman/yaz118/LLM_ckpts',max_model_len=71056)
    else:
        llm = LLM(model=model_name, tensor_parallel_size=card_number, download_dir='/ix1/dlitman/yaz118/LLM_ckpts')
    tokenizer_data = build_vllm_token_enforcer_tokenizer_data(llm) 
    # load dataset.
    for run_idx in range(1,4):
        df = pd.read_csv("all_data.csv")
        import pickle 
        available_ids = pickle.load(open("gold_annotate_ids.p", "rb"))
    
        aspects = []
        inputs = []
        ids = []
        clusters = []
        for i, row in df.iterrows():
            #if i not in available_ids:
             #   continue 
            aspects.append(row['topic'])
            inputs.append(row['input'])
            ids.append(i)
            clusters.append(row['cluster'])
    
        conversations = [construct_prompt(max_len, x, y) for x,y in zip(aspects, inputs)]
        print(conversations[0])
#        all_texts = vllm_with_character_level_parser(llm, conversations, JsonSchemaParser(AnswerFormat.schema()), tokenizer_data) 
        results = llm.chat(messages=conversations,
                   sampling_params=sampling_params,
                   use_tqdm=True)
        fout = pd.DataFrame()
        all_texts = [result.outputs[0].text for result in results]
        fout['pred'] = all_texts
        fout['aspect'] = aspects
        fout['input'] = inputs
        fout['ids'] = ids
        fout['cluster'] = clusters 
        fout.to_csv(f"1000_E2A_outputs/{model_choice}_{max_len}_{run_idx}.csv")

main()
