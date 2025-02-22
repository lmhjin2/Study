import pandas as pd

train = pd.read_csv('./train.csv', encoding = 'utf-8-sig')
test = pd.read_csv('./test.csv', encoding = 'utf-8-sig')

match_dict = {}

for input_text, output_text in zip(train['input'], train['output']):
    input_words = input_text.split()
    output_words = output_text.split()
    for iw, ow in zip(input_words, output_words):
        match_dict[iw] = ow  
        
def replace_words(input_text, match_dict):
    words = input_text.split() 
    replaced_words = [match_dict.get(word, word) for word in words] 
    return " ".join(replaced_words)

converted_reviews = test['input'].apply(lambda x: replace_words(x, match_dict)).tolist()

submission = pd.read_csv('./sample_submission.csv', encoding = 'utf-8-sig')
submission['output'] = converted_reviews
submission.to_csv('./baseline_submission.csv', index = False, encoding = 'utf-8-sig')
# https://dacon.io/competitions/official/236446/mysubmission