import pandas as pd 
import torch 
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig, pipeline

train = pd.read_csv('./train.csv', encoding = 'utf-8-sig')
test = pd.read_csv('./test.csv', encoding = 'utf-8-sig')

samples = []

for i in range(10):
    sample = f"input : {train['input'][i]} \n output : {train['output'][i]}"
    samples.append(sample)
    
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type= 'nf4',
    bnb_4bit_use_double_quant = True,
    bnb_4bit_compute_dtype=torch.bfloat16
)
model_id = 'beomi/gemma-ko-7b'
model = AutoModelForCausalLM.from_pretrained(model_id, quantization_config = bnb_config, device_map={"":0})
tokenizer = AutoTokenizer.from_pretrained(model_id)
tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = 'right'

pipe = pipeline(
    task="text-generation",
    model=model, 
    tokenizer=tokenizer 
)

restored_reviews = []


for index, row in test.iterrows():
    query = row['input']  
    
    messages = [
        {
            "role": "system",
            "content": (
                "You are a helpful assistant specializing in restoring obfuscated Korean reviews. "
                "Your task is to transform the given obfuscated Korean review into a clear, correct, "
                "and natural-sounding Korean review that reflects its original meaning. "
                "Below are examples of obfuscated Korean reviews and their restored forms:\n\n"
                f"Example, {samples}"  
                "Spacing and word length in the output must be restored to the same as in the input. "
                "Do not provide any description. Print only in Korean."
            )
        },
        {
            "role": "user",
            "content": f"input : {query}, output : "
        },
    ]
    
    prompt = "\n".join([m["content"] for m in messages]).strip()
    

    outputs = pipe(
        prompt,
        do_sample=True,
        temperature=0.2,
        top_p=0.9,
        max_new_tokens=len(query),
        eos_token_id=pipe.tokenizer.eos_token_id
    )
    
    generated_text = outputs[0]['generated_text']
    result = generated_text[len(prompt):].strip()
        

    restored_reviews.append(result)
    
submission = pd.read_csv('./sample_submission.csv', encoding = 'utf-8-sig')
submission['output'] = restored_reviews
submission.to_csv('./baseline_submission.csv', index = False, encoding = 'utf-8-sig')
