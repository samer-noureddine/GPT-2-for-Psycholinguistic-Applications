import torch
from transformers import *

from transformers import GPT2Tokenizer, GPT2LMHeadModel
from transformers import T5Tokenizer, AutoModelForCausalLM

from arabert import ArabertPreprocessor
from arabert.aragpt2.grover.modeling_gpt2 import GPT2LMHeadModel

from transformers import AutoTokenizer, AutoModel
from arabert.preprocess import ArabertPreprocessor

import numpy as np
import re


def softmax(x):
	exps = np.exp(x)
	return np.divide(exps, np.sum(exps))

# https://github.com/aub-mind/arabert
# pip install arabert

model_name = "aubmindlab/aragpt2-base"
arabert_prep = ArabertPreprocessor(model_name=model_name)
tokenizer = AutoTokenizer.from_pretrained(model_name)

model = GPT2LMHeadModel.from_pretrained(model_name, output_hidden_states=True, output_attentions=True) 
tokenizer = GPT2TokenizerFast.from_pretrained(model_name)
model.eval()



def cloze_prob(text):
	whole_text_encoding = tokenizer.encode(text)
	# Parse out the stem of the whole sentence (i.e., the part leading up to but not including the critical word)
	text_list = text.split()
	stem = ' '.join(text_list[:-1])
	stem_encoding = tokenizer.encode(stem)
	# Run the entire sentence through the model. Then go "back in time" to look at what the model predicted for each token, starting at the stem.
	cw_encoding = whole_text_encoding[len(stem_encoding):]
 
	# Put the whole text encoding into a tensor, and get the model's comprehensive output
	tokens_tensor = torch.tensor([whole_text_encoding])
	
	with torch.no_grad():
		outputs = model(tokens_tensor)
		predictions = outputs[0]   

	logprobs = []
	# start at the stem and get downstream probabilities incrementally from the model(see above)
	start = -1-len(cw_encoding)
	for j in range(start,-1,1):
			raw_output = []
			for i in predictions[-1][j]:
					raw_output.append(i.item())
	
			logprobs.append(np.log(softmax(raw_output)))
			
	# this is just: raw_probabilities[i][token_index]
	conditional_probs = []
	for cw,prob in zip(cw_encoding,logprobs):
			conditional_probs.append(prob[cw])
	# now that you have all the relevant probabilities, return their product.
	# This is the probability of the critical word given the context before it.

	return np.exp(np.sum(conditional_probs))

text= "شعرها جميل اليوم"
score = cloze_prob(text)
print(text, score)


