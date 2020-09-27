''' figuring out what I can get from the model '''
import math
import matplotlib.pyplot as plt
import torch
import pandas as pd
from transformers import GPT2Tokenizer, GPT2LMHeadModel
from mpl_toolkits import mplot3d
import numpy as np
from scipy import stats

# changes tensor to list (a "vector")
tovec = lambda x: [x[i].item() for i in range(len(x))]

def remove(lst):
        if "__" in lst[-1]:
           return " ".join(lst[:-1])
        else:
            return " ".join(lst)


def smoothing(p):
    ''' takes a discrete distribution p and smoothes the zeros it contains into small values
    e.g.,
    >>> smoothing([0.5281777381896973, 0.10816965997219086, 0.3636525273323059, 0.0, 0.0])
    >>> [0.527844404856364, 0.10783632663885752, 0.36331919399897256, 0.0005, 0.0005]
    '''
    # if epsilon is too large, smoothing might make one of the nonzero probabilities negative
    epsilon = 1e-30
    first_zero = p.index(0)
    # count how many zeros to smooth
    num_zeros = len(p) - first_zero
    # subtract off fractions of epsilon that would sum to 1
    for i in range(0,first_zero):
        p[i] = p[i] - epsilon/first_zero
    # add fractions of epsilon that would sum to 1
    for i in range(first_zero, len(p)):
        p[i] = epsilon/num_zeros
    return p

def attn_weights(outputs,attn_head,layer):
    '''for a given set of outputs, attention head and layer,
    return the attentional weights of the last two words.
    ie, the weights that the values of each word will be given
    before they are summed up at that head.
    e.g., the below example is when you input a 5-item sentence like 'He ate the apple quickly', and you want to see the weight distribution in the 3rd attention head at the 11th block/layer
    >>> w = attn_weights(outputs, 3, 11)
    >>> print(w)
    >>> [[0.8785303831100464, 0.10506650805473328, 0.011130983941257, 0.005272094160318375, 0.0], [0.5191097855567932, 0.4490145742893219, 0.0169433131814003, 0.006480566691607237, 0.00845175702124834]]
    so w[0] would give you the weights at attn head 3, layer 11 before the last word was presented, and w[1] would give you the weights after the final word.
    '''
    attn_tensor = outputs[3][layer][attn_head][-2:]
    pre_post_attn = []
    before_cw, after_cw = tovec(attn_tensor[0]), tovec(attn_tensor[1])
    pre_post_attn.append(before_cw)
    pre_post_attn.append(after_cw)
    return pre_post_attn

def KL_div(after, before):
    '''takes two prob dists and computes how much
    information is lost when you attempt to approximate the after dist
    using the before dist. This is the same as KL divergence(after,before)
    '''
    divergence = []
    before = smoothing(before)
    count = 0
    for i, j in zip(after,before):
        divergence.append(i*math.log(i/j))
    return sum(divergence)

def finalword_avg_attn(output,layer):
    # get the attention value of the final word
    lastword_attn = []
    # loop through each attention head and find the weight placed on the last word
    for attn_head in range(20):
        w = attn_weights(output,attn_head, layer)
        lastword_attn.append(w[-1][-1])
    # for the layer in the argument, return the mean of the attentions assigned to the final word
    return np.mean(lastword_attn)

def finalword_attn_variance(output,layer):
    # get the attention value of the final word
    lastword_attn = []
    # loop through each attention head and find the weight placed on the last word
    for attn_head in range(20):
        w = attn_weights(output,attn_head, layer)
        lastword_attn.append(w[-1][-1])
    # for the layer in the argument, return the variance of the attentions assigned to the final word
    return np.var(lastword_attn)

def avg_KL_div(output,layer):
    '''
     after a cw is presented, the dist of attentional weights will change in every attention head in every layer.
     For any given layer, one can compute the KL divergence for each of the 20 attn heads.
     This function does that, and then returns the average of all 20 resultant KL divergences
    '''
    # create a vector to append the KL divergence of each attn head
    KLs = []
    # loop through each attention head and find KL divergence between last two distributions of attention weights
    for attn_head in range(20):
        w = attn_weights(output,attn_head, layer)
        before,after = w[0],w[1]
        # compute KL divergence and append it to the KLs vector
        KLs.append(KL_div(after,before))
    # return the mean of the KLs vector
    return np.mean(KLs)       

def compare_KL(sentences1,sentences2):
    # this is just for two sentences. I want to extend this function to sets of sentences
    fig = plt.figure(figsize=(10,5))
    ax = fig.add_subplot(111)
    ax.set_xlabel('Layer number')
    ax.set_ylabel('KL divergence after critical word')
    all_sent1 = []
    all_sent2 = []
    for sentence in sentences1:
        with torch.no_grad():
            current_sentence_output = model(torch.tensor(tokenizer.encode(sentence)))
        track_sent = [avg_KL_div(current_sentence_output,i) for i in range(36)]
        all_sent1.append(track_sent)
        plt.plot(track_sent, 'r',label =  "..." + sentence[-10:])
    for sentence in sentences2:
        with torch.no_grad():
            current_sentence_output = model(torch.tensor(tokenizer.encode(sentence)))
        track_sent = [avg_KL_div(current_sentence_output,i) for i in range(36)]
        all_sent2.append(track_sent)
        plt.plot(track_sent, 'b',label =  "..." + sentence[-10:])
    # hc and lc are outputs of the model
    # trackhc is the KL divergence avged across attention_heads; there is one value per layer
    # trackhc[0] is the avg KL divergence (before and after cw) in the first layer, avged across attention heads.
    # maybe try to check whether the attention heads agree with each other. measure their variance or something
    ax.legend(loc = 'best')
    plt.show()
    return (all_sent1,all_sent2)

def each_attn_head(s1,s2,layer):
    '''
    >>> each_attn_head('Joe mowed the lawn', 'Joe mowed the yard', 11)
    This function grabs the attentional weights placed on the final word in s1 (from all 20 heads) and those placed on the final word in s2, and plots them against each other
    It gives you a sense of how much each head cares about the final word after it is presented.
    '''
    with torch.no_grad():
        out1 = model(torch.tensor(tokenizer.encode(s1)))
        out2 = model(torch.tensor(tokenizer.encode(s2)))
    w1 = [attn_weights(out1,head,layer)[-1][-1] for head in range(20)]
    w2 = [attn_weights(out2,head,layer)[-1][-1] for head in range(20)]
    plt.plot(w1)
    plt.plot(w2)
    plt.show()

def all_attn_heads(sentence):
        '''
        this function gets the attentional weights placed on the last word in the sentence, in each attention head and in each layer.
        I did this for plotting in 3D, hoping to see some obvious pattern emerge.
        In every 2D slice, you see 'weight' plotted vs 'attention head'. I later realized this was meaningless.
        '''
        a = []
        with torch.no_grad():
            out1 = model(torch.tensor(tokenizer.encode(sentence)))
        for layer in range(36):
            a.append([attn_weights(out1,i,layer)[-1][-1] for i in range(20)])
        for i in range(len(a)):
            for j in range(len(a[i])):
                a[i][j] = [j,a[i][j]]
        all_attn = [[i,j] for i,j in enumerate(a)]
        return all_attn

def attn_head_vs_time(sentence):
        '''
        this function gets the attentional weights placed on the last word in the sentence, in each attention head and in each layer.
        I did this for later plotting in 3D, hoping to see some obvious pattern emerge.
        In every 2D slice, you see 'weight' plotted vs 'layer'. Every 2D slice comes from a different attention head.
        '''
        a = []
        with torch.no_grad():
                out1 = model(torch.tensor(tokenizer.encode(sentence)))
        for attn in range(20):
                a.append([attn_weights(out1,attn,i)[-1][-1] for i in range(36)])
        for i in range(len(a)):
                for j in range(len(a[i])):
                        a[i][j] = [j,a[i][j]]
        all_attn = [[i,j] for i,j in enumerate(a)]
        return all_attn

def plot_all(data):
    # plot 3D data
    xyz = [(x,y,z) for (x, yy) in data for (y,z) in yy]
    f, ax = plt.subplots(subplot_kw={'projection':'3d'})
    for x, yz in data:
        ax.plot([x]*len(yz), *zip(*yz),'b')

def compare_attn(data1,data2, mi, ma):
    # plot 3D data
    xyz = [(x,y,z) for (x, yy) in data for (y,z) in yy]
    f, ax = plt.subplots(subplot_kw={'projection':'3d'})
    ax.set_xlim(mi,ma)
##    ax.set_ylim(mi,ma)
##    ax.set_zlim(mi,ma)
    for x, yz in data1:
        ax.plot([x]*len(yz), *zip(*yz),'b')
    for x, yz in data2:
        ax.plot([x]*len(yz), *zip(*yz),'r')
       

def attn_entropy(p):
    p = np.array(p)
    logp = np.log(p)/np.log(2)
    plogp = p*logp
    entropy = -np.sum(plogp)
    return entropy

def sentence_entropy(s1):
    '''given sentences s1 and s2, compute the mean entropy
       in each layer (each head has one entropy value)
    '''
    with torch.no_grad():
        out1 = model(torch.tensor(tokenizer.encode(s1)))
    allweights = [[attn_weights(out1,h,l) for h in range(20)] for l in range(36)]
    all_entropy = [[attn_entropy(attn_weights(out1,h,l)) for h in range(20)] for l in range(36)]
    mean_ent = [np.mean(np.array(all_entropy[i])) for i in range(len(all_entropy))]
    return mean_ent
# set up the model and tokenizer
model = GPT2LMHeadModel.from_pretrained('gpt2-large', output_hidden_states = True, output_attentions = True)
model.eval()
tokenizer = GPT2Tokenizer.from_pretrained('gpt2-large')
