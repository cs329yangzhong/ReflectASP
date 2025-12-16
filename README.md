# ReflectASP
Dataset and Codebase for ACL 2025 accepted paper 

## Data
- human annotation dataset: **gold_annotate_data.p** 
	- format: {id: [LIST OF HUMAN-WRITTEN SUMMARIES]}

- **gold_annotate_ids.p**
	list of ids that have been annotated

- **all_data.csv** 
	1000+ data from ReflectSumm

## Code

The codebase utilizes vllm for fast inference.
We also release the generation output for these different approaches on our dataset.
- **vllm_e2a.py**
  
	Exract then abstract baseline
- **vllm_e2a_mc_refine.py**
  
	E2A w MC-Refine baseline. This requires a preprocessing of the data on the 
	E2A summaries to identify those sentences that are not faithful. You can follow the run_minicheck.py 
	script to perform the inference and modify the column of your own data.
- **vllm_self_refine.py**
  
	Self-refine baseline.

