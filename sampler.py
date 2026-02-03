import pandas as pd

# For qualitative eval of short text

filepaths = ["data/inference/falcon_gpt_preprocessed_inference_prompt0.csv",
             "data/inference/falcon_llama_preprocessed_inference_prompt0.csv",
             "data/inference/mistral_gpt_preprocessed_inference_prompt0.csv",
             "data/inference/mistral_llama_preprocessed_inference_prompt0.csv"
            ]

# Loop through each file
for filepath in filepaths:
    # Read the CSV file
    df = pd.read_csv(filepath)
    
    # Sample 10 random rows
    sample_df = df.sample(n=10)
    
    # Print the row number (index) and the row
    for index, row in sample_df.iterrows():
        print(f"Row number: {index}")
        print(row)
        print("\n")