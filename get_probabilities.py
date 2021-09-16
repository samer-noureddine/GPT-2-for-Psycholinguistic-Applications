''' 
Authors: 
	Samer Nour Eddine (snoure01@tufts.edu)
	Feng Cheng (fcheng6@mgh.harvard.edu)
'''

import torch
from transformers import GPT2Tokenizer, GPT2LMHeadModel
import numpy as np
import re

def softmax(x):
	exps = np.exp(x)
	return np.divide(exps, np.sum(exps))
	
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

'''
This Function parses the text into word tokins and mark the composition of the 
text (i.e., the pos of a word in the sentence/text ). This function is
necessary because GPT2 tokenizer will likely break words into sub-word tokens.
By first dividing the text into word-level tokins (which also grants us the
liberty of defining "words": e.g., we want to count the period as part of the 
last word) and then passing each tokin to the GPT2 tokenizer, we know how each
word is converted into sub-word tokins and can therefore calculate the cloze of
each word by calculating the conditional probabilities of each word's sub-word
tokins. (Note that GPT2 Tokenizer is context independent, so it doesn't matter
whether we pass the words individually or we pass the entire text altogether)

text:		  text to parse
tkn_regex:	  token regular expression. Defines how the text is seperated into
              tokens. 
bl_regex:	  blacklist regular expression. Defines what characters will be 
			  deleted
eos_chars:	  characters that mark the end of sentence. If None is passed,
			  the function will no longer mark the composition of the text
eos_char_pos: the position where eos characters are expected. By default the
			  function will look for the last char in every string to identify
			  eos
'''
def parse_text(text, tkn_regex=r" *[\w'\"\.!?-]+", bl_regex=r"\.\.+|[,;]| -", eos_chars = [], eos_char_pos = -1):
	# Straighten quotation marks
	text = text.replace('“','"').replace('”','"')
	text = text.replace("’","'").replace("`","'")
	# Clean the punctuation marks
	if bl_regex is not None:
		text = re.sub(bl_regex, "", text)
	# Join the spaces
	text = re.sub(r"\s\s+", " ", text)
	# Divide the sentence
	text_tkn = re.findall(tkn_regex, text)
	# label text composition
	if len(eos_chars) > 0:
		eos_pos = []
		for i,tkn in enumerate(text_tkn):
			if tkn[eos_char_pos] in eos_chars: eos_pos.append(i)
		pos_labels = np.zeros((4, len(text_tkn)), dtype = int)
		pos_labels[2] = np.arange(len(text_tkn), dtype = int)
		prev_s_pos = 0
		for s_count, s_pos in enumerate(eos_pos):
			s_end = s_pos+1
			pos_labels[0][prev_s_pos:s_end] = s_count
			pos_labels[1][prev_s_pos:s_end] = np.arange(s_end - prev_s_pos)
			prev_s_pos = s_end
		pos_labels[:3] = np.add(pos_labels[:3], 1)
		pos_labels[3][eos_pos] = 1
		labels_dict = dict({"sentence_label":pos_labels[0], "sentence_pos":pos_labels[1], "utterance_pos":pos_labels[2], "eos":np.array(pos_labels[3], dtype = bool)})
		return text_tkn, labels_dict
	return text_tkn	


'''
This function uses GPT2 to generate the cloze probabilities of a given list of
word-level tokins. One way to obtain such tokins is to pass the text to the
function parse_text, and pass its output text_tkn as an input to this function. This function will calculate the conditional probability of each of the 
word-level tokin given the tokins preceeding it. Importantly, the first 
word-level tokin is assigned with the probability value of 0.
'''
def cloze_allword(text_tkn):
	pos_arr = []
	curr_pos = 0
	encoding = []
	for tkn in text_tkn:
		# pass each tokin into the GPT2 tokenizer
		curr_encoding = tokenizer.encode(tkn)
		# join the current encoding into the utterance encoding
		encoding.extend(curr_encoding)
		# marks the indices of the subword tokins that belone to the same word
		# tokin
		pos_labels = np.arange(curr_pos, curr_pos + len(curr_encoding))
		pos_arr.append(pos_labels)	
		curr_pos += len(curr_encoding)
	# Run the model
	tokens_tensor = torch.tensor([encoding])
	with torch.no_grad():
		outputs = model(tokens_tensor)
		 = outputs[0][0]
	results = predictions.detach().cpu().numpy()
	# Transform scores into log probability
	for r in range(results.shape[0] - 1):
		results[r] = np.log(softmax(results[r]))
	# Obtain conditional probability of each word token. Note that here we skip
	# the first word no matter how many sub-word tokens it has. We set the cloze
	# of the first sentence in the utterance/text to be 0.
	conditional_probs = []
	for tkn_pos in pos_arr[1:]:
		tkn_prob = []
		# each tkn_pos is an array of position indices
		for pos in tkn_pos:
			# we use pos-1 because the results at pos-1 is the prediction for 
			# the upcoming tokin at pos
			tkn_prob.append(results[pos-1][encoding[pos]])
		conditional_probs.append(np.sum(tkn_prob))
	conditional_probs = np.exp(np.array(conditional_probs))
	return np.insert(conditional_probs, 0, 0)

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
	print (cw_encoding)
	print (whole_text_encoding)

	# Run the entire sentence through the model. Then go back in time and look at what the model predicted for each token, starting at the stem.
	# e.g., for 'Joe flicked the grasshopper', go back to when the model had just received 'Joe flicked the' and
	# find the probability for the next token being 'grass'. Then for 'Joe flicked the grass' find the probability that
	# the next token will be 'ho'. Then for 'Joe flicked the grassho' find the probability that the next token will be 'pper'.

	# Put the whole text encoding into a tensor, and get the model's comprehensive output
	tokens_tensor = torch.tensor([whole_text_encoding])
	
	with torch.no_grad():
		outputs = model(tokens_tensor)
		predictions = outputs[0]   

	logprobs = []
	# start at the stem and get downstream probabilities incrementally from the model(see above)
	# I should make the below code less awkward when I find the time
	start = -1-len(cw_encoding)
	for j in range(start,-1,1):
			print (j)
			raw_output = []
			for i in predictions[-1][j]:
					raw_output.append(i.item())
	
			logprobs.append(np.log(softmax(raw_output)))
			
	# if the critical word is three tokens long, the raw_probabilities should look something like this:
	# [ [0.412, 0.001, ... ] ,[0.213, 0.004, ...], [0.002,0.001, 0.93 ...]]
	# Then for the i'th token we want to find its associated probability
	# this is just: raw_probabilities[i][token_index]
	conditional_probs = []
	for cw,prob in zip(cw_encoding,logprobs):
			print (prob[cw])
			conditional_probs.append(prob[cw])
	# now that you have all the relevant probabilities, return their product.
	# This is the probability of the critical word given the context before it.
	

	# logprobs = []
	# for j in range(len(whole_text_encoding)):
	# 		raw_output = []
	# 		for i in predictions[-1][j]:
	# 				raw_output.append(i.item())
	
	# 		logprobs.append(np.log(softmax(raw_output)))
	# print (np.array(logprobs).shape)
	# conditional_probs = []
	# for cw,prob in zip(whole_text_encoding,logprobs):
	# 		print (prob[cw])
	# 		conditional_probs.append(prob[cw])
	# print (conditional_probs)

	return np.exp(np.sum(conditional_probs))

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

	logprobs = np.log(softmax(raw_output))
	sorted_logprobs = Sort_Tuple([(i,j) for i,j in enumerate(logprobs)])
	sorted_words = [(tokenizer.decode(i[0]).strip(), np.exp(i[1])) for i in sorted_logprobs]
	if top_ten:
			h = []
			for i in zip(*sorted_words):
				h.append(i)
			if top_ten:
				print(sorted_words[:10])
				print("*****")
	if constraint:
			return np.exp(sorted_logprobs[0][1])
	return cloze_finalword(' '.join([text, critical_word]))
