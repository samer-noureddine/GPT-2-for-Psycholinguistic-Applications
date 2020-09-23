import torch
from transformers import GPT2Tokenizer, GPT2LMHeadModel
import matplotlib.pyplot as plt
import pandas
import numpy as np

def softmax(x):
	exps = [np.exp(i) for i in x]
	tot= sum(exps)
	return [i/tot for i in exps]
    
def Sort_Tuple(tup):  
  
    # (Sorts in descending order)  
    # key is set to sort using second element of  
    # sublist lambda has been used  
    tup.sort(key = lambda x: x[1])  
    return tup[::-1]

# Load pre-trained model (weights) - this takes the most time
model = GPT2LMHeadModel.from_pretrained('gpt2-large', output_hidden_states = True, output_attentions = True)
model.eval()

tokenizer = GPT2Tokenizer.from_pretrained('gpt2-large')

def cloze_finalword(text):
    '''
    This is a version of cloze generator that can handle words that are not in the model's dictionary.
    '''
    whole_text_encoding = tokenizer.encode(text)
    # Parse out the stem of the whole sentence (i.e., the part leading up to but not including the critical word)
    text_list = text.split()
    stem = ' '.join(text_list[:-1])
    stem_encoding = tokenizer.encode(stem)
    # cw_encoding is just the difference between whole_text_encoding and stem_encoding
    # note: this might not correspond exactly to the word itself
    # e.g., in 'Joe flicked the grasshopper', the difference between stem and whole text (i.e., the cw) is not 'grasshopper', but
    # instead it is ' grass','ho', and 'pper'. This is important when calculating the probability of that sequence.
    cw_encoding = whole_text_encoding[len(stem_encoding):]

    # Run the entire sentence through the model. Then go back in time and look at what the model predicted for each token, starting at the stem.
    # e.g., for 'Joe flicked the grasshopper', go back to when the model had just received 'Joe flicked the' and
    # find the probability for the next token being 'grass'. Then for 'Joe flicked the grass' find the probability that
    # the next token will be 'ho'. Then for 'Joe flicked the grassho' find the probability that the next token will be 'pper'.

    # Put the whole text encoding into a tensor, and get the model's comprehensive output
    tokens_tensor = torch.tensor([whole_text_encoding])
    
    with torch.no_grad():
        outputs = model(tokens_tensor)
        predictions = outputs[0]   

    raw_probabilities = []
    # start at the stem and get downstream probabilities incrementally from the model(see above)
    # I should make the below code less awkward when I find the time
    start = -1-len(cw_encoding)
    for j in range(start,-1,1):
            raw_output = []
            for i in predictions[-1][j]:
                    raw_output.append(i.item())
    
            raw_probabilities.append(softmax(raw_output))
            
    # if the critical word is three tokens long, the raw_probabilities should look something like this:
    # [ [0.412, 0.001, ... ] ,[0.213, 0.004, ...], [0.002,0.001, 0.93 ...]]
    # Then for the i'th token we want to find its associated probability
    # this is just: raw_probabilities[i][token_index]
    conditional_probs = []
    for cw,prob in zip(cw_encoding,raw_probabilities):
            conditional_probs.append(prob[cw])
    # now that you have all the relevant probabilities, return their product.
    # This is the probability of the critical word given the context before it.
    return np.prod(conditional_probs)

def cloze_generator(text, critical_word, top_ten = False, constraint = False):
    '''
    run text through model, then output the probability of critical_word given the text.
    This is quite redundant with the cloze_nonword function. I should fix this later.
    '''



    # Encode a text inputs

    indexed_tokens = tokenizer.encode(text)

    tokens_tensor = torch.tensor([indexed_tokens])


    # Predict all tokens
    with torch.no_grad():
        outputs = model(tokens_tensor)
        predictions = outputs[0]

    # put the raw output into a vector, then softmax it
    raw_output = []
    # I use predictions[-1][-1] only because I'm interested in the probs of the word after the prompt.
    # However, cloze values for all the words are available after each word in the prompt.
    # So in a ten word sentence, There are 500,000 probability values (50k for each word)
    for i in predictions[-1][-1]:
            raw_output.append(i.item())

    raw_probabilities = softmax(raw_output)
    sorted_probabilities = Sort_Tuple([(i,j) for i,j in enumerate(raw_probabilities)])
    sorted_words = [(tokenizer.decode(i[0]).strip(), i[1]) for i in sorted_probabilities]
    if top_ten:
            h = []
            for i in zip(*sorted_words):
                h.append(i)
            if top_ten:
                print(sorted_words[:10])
                print("*****")
    if constraint:
            return sorted_probabilities[0][1]
    return cloze_finalword(' '.join([text, critical_word]))
	





