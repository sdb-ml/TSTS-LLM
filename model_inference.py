from transformers import AutoModelForCausalLM, AutoTokenizer
from tqdm import tqdm
import pandas as pd
import gc

# model_name = "mistralai/Mistral-7B-Instruct-v0.3" OR "tiiuae/falcon-7b-instruct"
# model_checkpoint = "./results_packing/mistral_gpt_Checkpoint" # Change to your merged checkpoint

model_names = ["tiiuae/falcon-7b-instruct", "tiiuae/falcon-7b-instruct"]

checkpoints = ['./falcon_results_packing/falcon_llama_Checkpoint', './falcon_gpt_results_packing/falcon_gpt_Checkpoint']

file_paths = ['data/from_Llama3_traintest_final.csv', 'data/from_GPT_test_final.csv']

output_names = ['falcon_llama', 'falcon_gpt']

for model_name, model_checkpoint, file_path, output_name in zip(model_names, checkpoints, file_paths, output_names):
    
    model = AutoModelForCausalLM.from_pretrained(model_checkpoint,
                                                 device_map="auto", # automatically figures out how to best use CPU + GPU for loading model
                                                 trust_remote_code=False, # prevents running custom model files on your machine
                                                 revision="main") # which version of model to use in repo

    tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)

    model.eval() # model in evaluation mode (dropout modules are deactivated)
    print(model)

    if file_path == 'data/from_Llama3_traintest_final.csv':
        test_data = pd.read_csv(file_path, skiprows=range(1, 7393))
    elif file_path == 'data/from_GPT_test_final.csv':
        test_data = pd.read_csv(file_path)
    else:
        raise NotImplementedError(f"File path '{file_path}' not found.")

    instructions = ["Translate the following sentence into a form suitable for a child: ",
                    "Simplify the following so that a child will understand: ", 
                    "Simplify the following: "]

    for i, instruction in enumerate(instructions):

        if model_name == "tiiuae/falcon-7b-instruct":
            prompts = ['[INST] ' + instruction + ad.strip("'\"") + ' [/INST]'for ad in test_data['adult_definition']]
        else:
            raise NotImplementedError(f"Model '{model_name}' not implemented.")

        results = []
        for prompt in tqdm(prompts, desc='Generating', unit='Sentence'):
            # tokenize input
            inputs = tokenizer(prompt, return_tensors="pt") 

            # generate output
            outputs = model.generate(input_ids=inputs["input_ids"].to("cuda"), attention_mask=inputs['attention_mask'].to('cuda'), max_new_tokens=140)

            #save output
            results.append(tokenizer.batch_decode(outputs)[0])

        output_df = pd.DataFrame(results)

        output_df.to_csv(f"data/inference/{output_name}_inference_prompt_{i}.csv", header=False, index=False)
        
    del model
    torch.cuda.empty_cache()
    gc.collect()