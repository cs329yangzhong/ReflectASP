from vllm import LLM
from vllm.sampling_params import SamplingParams
import pandas as pd
import argparse

sampling_params = SamplingParams(temperature=0.3, max_tokens=1000)

import os 

os.environ['TRANSFORMERS_CACHE'] = "/ix1/dlitman/yaz118/.cache"
os.environ['HF_HOME'] = "/ix1/dlitman/yaz118/.cache"
os.environ['HF_DATASETS_CACHE'] = "/ix1/dlitman/yaz118/.cache"
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

def get_refine_suggestion_prompt( reflections, summary, aspect, max_len):
    conversation = [
    {
        "role": "system",
        "content": '''You are a TA for a undegraduate-level course, you are given a collection of student reflections after taking one lecture and tasked to write a summary to present to the instructor'''
    },
    {
        "role": "user",
        "content": f'''Given the students' reponses, and a focused topic {aspect}, you are given a short summary. (no more than {max_len} words). The summary is  a coherent paragraph and should include the major points. The summary should focus on the provided topic only, contain only information about reflections, and avoid adding irrelevent sentences or suggestions such as 'make sure to bring this up in next class', or 'Consider this for future lectures', etc... ; Now, Can you provide a short list of 2-3 suggestions to improve the generated summary, making it more concise and focused on the topic {aspect}? The suggestions should be based
on the original reflections and generated summaries, don’t give generic suggestions. 
        REFLECTION: {reflections}

        FOCUSED TOPIC: {aspect} 
        Original Summary: {summary}

       your repsonse should be this format: Suggestions: [your response] '''
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
        "content": f'''Given the students' reponses, and a focused topic {aspect}, you are tasked to improve a short summary. (no more than {max_len} words). Please improve the short summary written below, with the suggestions. The revised version should be no more than {max_len} words, focusing on the topic of {aspect} based on below
reflections, The summary is  a coherent paragraph and should include the major points.:        
      
        REFLECTION: {reflections}
        FOCUSED TOPIC: {aspect} 
        Original Summary: {summary}
        Suggestions: {suggestions}

       your repsonse should be this format: REFINED SUMMARY: [your response] '''
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
    config_format="mistral", download_dir='/ix1/dlitman/yaz118/LLM_ckpts/',  tensor_parallel_size=card_number
)
    elif model_choice == 'nvidia':
        llm = LLM(model=model_name, tensor_parallel_size=card_number, download_dir='/ix1/dlitman/yaz118/LLM_ckpts',max_model_len=71056)
    else:
        llm = LLM(model=model_name, tensor_parallel_size=card_number, download_dir='/ix1/dlitman/yaz118/LLM_ckpts')
    
    # load dataset.
    for run_idx in range(1,4):
        #if os.path.exists('outputs/{model_choice}_{max_len}_{run_idx}.csv'):
         #   print("existed ", f'{model_choice}_{max_len}_{run_idx}.csv')
          #  continue
        df = pd.read_csv("turn0.csv")
        import pickle 
        available_ids = pickle.load(open("treshold.p", "rb"))
    
        aspects = []
        inputs = []
        ids = []
        clusters = []
        for i, row in df.iterrows():
           # if i not in available_ids:
            #    continue 
            aspects.append(row['topic'])
            inputs.append(row['input'])
            ids.append(i)
            clusters.append(row['cluster'])
    
        conversations = [construct_prompt(max_len, x, y) for x,y in zip(aspects, inputs)]
        print(conversations[0])
        outputs = llm.chat(messages=conversations,
                   sampling_params=sampling_params,
                   use_tqdm=True)
        summary_texts = [out.outputs[0].text for out in outputs]
        
        # suggestions.
        suggestion_conversations = [get_refine_suggestion_prompt(inputs[i], summary_texts[i], aspects[i], max_len) for i in range(len(summary_texts))]
        suggestion_outputs = llm.chat(messages=suggestion_conversations,
                   sampling_params=sampling_params,
                   use_tqdm=True)
        suggestions_texts = [out.outputs[0].text for out in suggestion_outputs]
        #get_revision_prompt( reflections, summary, aspect, suggestions, max_len)
        refine_conv = [get_revision_prompt(inputs[i], summary_texts[i], aspects[i], suggestions_texts[i], max_len) for i in range(len(summary_texts))]  
        refine_outputs = llm.chat(messages=refine_conv,
                   sampling_params=sampling_params,
                   use_tqdm=True)
        refined_texts = [out.outputs[0].text for out in refine_outputs]
        fout = pd.DataFrame()
        fout['pred'] = summary_texts
        fout['aspect'] = aspects
        fout['input'] = inputs
        fout['ids'] = ids
        fout['cluster'] = clusters 
        fout['suggestion'] = suggestions_texts
        fout['refined_pred'] = refined_texts
        fout.to_csv(f"1000_refined_outputs/{model_choice}_{max_len}_{run_idx}.csv")

main()
