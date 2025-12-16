from vllm import LLM
from vllm.sampling_params import SamplingParams
import sys
import pandas as pd
import argparse
from nltk.tokenize import sent_tokenize
from minicheck.minicheck import MiniCheck
import gc
from vllm.distributed.parallel_state import destroy_model_parallel
sampling_params = SamplingParams(temperature=0.3, max_tokens=1000)
import torch
import os 

os.environ['TRANSFORMERS_CACHE'] = "/ix1/dlitman/yaz118/.cache"
os.environ['HF_HOME'] = "/ix1/dlitman/yaz118/.cache"
os.environ['HF_DATASETS_CACHE'] = "/ix1/dlitman/yaz118/.cache"

def construct_eta_prompt(max_len, aspect, reflections):
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
        (1) Extract the list of students' reflections in the given REFLECTIONs  that are relevant to the focused topic.
        (2) summarize them into a short summary using your own words (no more than {max_len} words). 

        REFLECTIONS: {converted}

        FOCUSED TOPIC: {aspect} 

       your repsonse should be this json format: {{'Extracted_Reflections' : [your extracted data] (i.e. [STUDENT_REFLECTION_1_TEXT, STUDENT_REFLECTION_2_TEXT, ...]), 'SUMMARY': [your response]}} 
'''
    },
]
    return conversation


def construct_prompt(max_len, aspect, reflections):
    conversation = [
    {
        "role": "system",
        "content": '''You are a TA for a undegraduate-level course, you are given a collection of student reflections after taking one lecture and tasked to write a summary to present to the instructor'''
    },
    {
        "role": "user",
        "content": f'''Given the students' reponses, and a focused topic {aspect}, create a short summary using your own words (no more than {max_len} words). The summary needs to be a coherent paragraph and should include the major points. The summary should focus on the provided topic only, contain only information about reflections, and avoid adding irrelevent sentences or suggestions such as 'make sure to bring this up in next class', or 'Consider this for future lectures', etc... 

        REFLECTION: {reflections}

        FOCUSED TOPIC: {aspect} 

       your repsonse should be this format: SUMMARY: [your response] '''
    },
]
    return conversation

def get_refine_suggestion_prompt( reflections, ext_summary, summary, aspect, revised_sent, max_len):
    from ast import literal_eval
    revised_sents = literal_eval(revised_sent)
    conversation = [
    {
        "role": "system",
        "content": '''You are a TA for a undegraduate-level course, you are tasked to provide revision suggestions to a sumamry of student reflections.'''
    },
    {
        "role": "user",
        "content": f'''Given the students' reponses, and a focused topic {aspect}, you are given an extacted list of reflections that are related to the topic and a short summary. (no more than {max_len} words). \n\n             
        ALL REFLECTION: {reflections}
        FOCUSED TOPIC: {aspect} 
        Extracted List of origianl reflection and initial Summary: {ext_summary}
        Initial Summary: {summary}

Now, given the below GIVEN_SENTENCES in the summary that is identified as unfaithful, reason if there is any factually inconsistent span in the sentence and propose a way to improve the sentence, making it more concise and focused on the topic {aspect}, utilizing information from the extracted list of reflections. The suggestions should be based on the original reflections and the extracted list of reflections, don’t give generic suggestions.\n

        ***GIVEN_SENTENCES: {revised_sents}***
        
        ### Task:
        If GIVE_SENTENCES is []: your response should just return: Suggestions: <no revision needed>.
        Otherwise, for each sent in  GIVEN_SENTENCES, your repsonse should output: Suggestions: {{original_sent: <SENT1 from GIVEN_SENTENCES>, the error span: <span from SENT1>, the revision suggestion: <your suggested revision>(Delete if there is no need to keep or the post-edit version of SENT1)}}
     \n\n 
        ### EXAMPLE OUTPUTS: \n if GIVEN_SENTENCES: [], you should just return "<no suggestion needed>". \nif GIVEN_SENTENCES= [<SENT1>, <SENT2>], your output would be Suggestions: "{{original_sent: <SENT1>, the error span: <span in SENT1>, the revision suggestion: <Modified version of SENT1>}} {{oiriginal_sent: <SENT2> , the error span: <span in SENT2> ....}}".      
      Reminder: you should only provide Suggestions on sentences from the GIVEN_SENTENCES. If it is empty, just return Suggestions: <no revision needed>.
      '''
    },
]
    return conversation
def get_revision_prompt( reflections, summary, aspect, suggestions, max_len):
    conversation = [
    {
        "role": "system",
        "content": '''You are a TA for a undegraduate-level course, you are given a collection of student reflections after taking one lecture and tasked to write a summary to present to the instructor'''
    },
    {
        "role": "user",
        "content": f'''Given the students' reponses, and a focused topic {aspect}, a list of extracted responses that are relevant to the topic, and some suggestions on revisions, you are tasked to improve a short summary. (no more than {max_len} words). Please improve the short summary written below, encorporating the suggestions. The suggestions are on sentences of the summary, so please only modify those highlighted sentences and keep the remainder unchanged. The revised version should be no more than {max_len} words, focusing on the topic of {aspect} based on below reflections, The summary should be a coherent paragraph and should include the major points. If you think the initial summary is good enough, you can make minimal changes, You also need to pay attention to the extracted list of reflections that are related to the given topic {aspect}:        
      
        ALL REFLECTION: {reflections}
        FOCUSED TOPIC: {aspect} 
        Extracted List of origianl reflection and Initial Summary: {summary}
        Revision Suggestions: {suggestions}
       
       You should pay atention to the revision suggestions and decide whether you want to edit the sentences with suggested revisions. if the Revision Suggestions mentions "no suggestion needed", you should not revise the initial summary. Your summary should have minimal changes on the initial summary and be factual.
       your repsonse should just output the refined summary and should not include any extra explaination on changes. The format is --  REFINED SUMMARY: [your response]'''
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
    		     #'llama3.1-70': 'meta-llama/Meta-Llama-3.1-405B-Instruct-FP8',
			'mistral': 'mistralai/Mistral-NeMo-Instruct-2407'}
    model_name = model_mapping.get(model_choice, "NOT FOUND")
    if model_choice == 'mistral':
        llm = LLM(
    model=model_name,
    gpu_memory_utilization=0.95,
tokenizer_mode="mistral",
    load_format="mistral",
    config_format="mistral", download_dir='/ix1/dlitman/yaz118/LLM_ckpts/',  tensor_parallel_size=card_number, max_model_len=110000)
    elif model_choice == 'nvidia':
        llm = LLM(model=model_name, tensor_parallel_size=card_number, download_dir='/ix1/dlitman/yaz118/LLM_ckpts',max_model_len=71056)
    else:
        llm = LLM(model=model_name, tensor_parallel_size=card_number, download_dir='/ix1/dlitman/yaz118/LLM_ckpts')
    
    # load dataset.
    for run_idx in range(1,4):
        if not os.path.exists(f'MC_sent_detect/{model_choice}_{max_len}_{run_idx}.csv'):
         #   print("existed ", f'{model_choice}_{max_len}_{run_idx}.csv')
            continue
#        df = pd.read_csv("turn0.csv")
        df = pd.read_csv(f'MC_sent_detect/{model_choice}_{max_len}_{run_idx}.csv')
        #df = df.head(5)

        inputs = df['input']
        summary_texts = df['e2a_pred']
        ext_and_summaries = df['pred']
        aspects = df['aspect']
        all_input_sents = df['minicheck_all_input_detect']
        all_pred_ext_sents = df['minicheck_pred_ext_detect']
        all_ext_sents = df['minicheck_ext_detect']
        ids = df['ids'] 
        ## suggestion.
        for key in ['pred_ext', 'all_input', 'ext']:
            checker_row = f'minicheck_{key}_detect'
            check_rows = df[checker_row]
            suggestion_conversations = [get_refine_suggestion_prompt(inputs[i],ext_and_summaries[i], summary_texts[i], aspects[i], check_rows[i],  max_len) for i in range(len(summary_texts))]
            suggestion_outputs = llm.chat(messages=suggestion_conversations,
                   sampling_params=sampling_params,
                   use_tqdm=True)
            suggestions_texts = [out.outputs[0].text for out in suggestion_outputs]
            print(suggestions_texts[:3])
            refine_conv = [get_revision_prompt(inputs[i], summary_texts[i], aspects[i], suggestions_texts[i], max_len) for i in range(len(summary_texts))]  
            refine_outputs = llm.chat(messages=refine_conv,
                   sampling_params=sampling_params,
                   use_tqdm=True)
            refined_texts = [out.outputs[0].text for out in refine_outputs]
            print(refined_texts[:5])
            fout = pd.DataFrame()
        #fout['cluster'] = clusters 
            fout['pred'] = summary_texts
            fout['aspect'] = aspects
            fout['input'] = inputs
            fout['ids'] = ids
            fout[f'{key}_suggestion'] = suggestions_texts
            fout['refined_pred'] = refined_texts
            fout.to_csv(f"eta_mc_detect_refine/{model_choice}{key}_{max_len}_{run_idx}.csv")

main()
