import pandas as pd

train = pd.read_csv('./train.csv', encoding = 'utf-8-sig')
test = pd.read_csv('./test.csv', encoding = 'utf-8-sig')

match_dict = {}

for input_text, output_text in zip(train['input'], train['output']):
    input_words = input_text.split()
    output_words = output_text.split()
    for iw, ow in zip(input_words, output_words):
        match_dict[iw] = ow  
        
match_dict
