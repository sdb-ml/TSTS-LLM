from transformers import AutoModelForCausalLM, AutoTokenizer
from tqdm import tqdm
import pandas as pd

model_name = "mistralai/Mistral-7B-Instruct-v0.3"
model_checkpoint = "./results_packing/Final_Model_Checkpoint" # Use this one for your finetuned model
model = AutoModelForCausalLM.from_pretrained(model_checkpoint,
                                             device_map="auto", # automatically figures out how to best use CPU + GPU for loading model
                                             trust_remote_code=False, # prevents running custom model files on your machine
                                             revision="main") # which version of model to use in repo

tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)

model.eval() # model in evaluation mode (dropout modules are deactivated)
print(model)

# craft prompt (single prompt)
articles = pd.read_csv('mistral/data/Books_Abstracts.csv', header=None, sep='^', engine='python', quoting=3)

instructions = ["Simplify the following so that a child will understand: "]

prompts = []
pd.set_option('display.max_colwidth', None)
for i, instruction in enumerate(instructions):
    for n, row in enumerate(articles.iterrows()):
        ls_row = []
        series = row[1]
        for entry in series:
            if pd.notna(entry) and str(entry).strip():
                entry_str = str(entry).strip()
                ls_row.append(entry_str)
        prompts = ['[INST] ' + instruction + str(sentence) + '[/INST]' for sentence in ls_row]

        results = []
        for prompt in tqdm(prompts, desc='Generating', unit='Sentence'):
            # tokenize input
            inputs = tokenizer(prompt, return_tensors="pt") 

            # generate output
            outputs = model.generate(input_ids=inputs["input_ids"].to("cuda"), attention_mask=inputs['attention_mask'].to('cuda'), max_new_tokens=140)

            #save output
            results.append(tokenizer.batch_decode(outputs)[0])

        output_df = pd.DataFrame(results)

        output_df.to_csv(f"mistral/data/Books_whole_row_{n}.csv", header=False, index=False)

#print(tokenizer.batch_decode(outputs)[0])
