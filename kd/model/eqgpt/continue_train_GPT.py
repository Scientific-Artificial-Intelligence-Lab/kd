import json
import pickle

import matplotlib.pyplot as plt
import pandas as pd
import torch
import torch.utils.data as Data
from torch import nn, optim
import numpy as np
import time
from tqdm import tqdm
from .gpt_model import *
import torch.nn.functional as F
from sympy import Matrix
from .calculate_terms import calculate_terms
from torch.autograd import Variable
from .surrogate_model import *
import numpy
import warnings
from ._device import device, empty_cache, load_checkpoint
from pathlib import Path as _Path
_DIR = _Path(__file__).parent
empty_cache()
dict_datas = json.load(open(_DIR / 'dict_datas_0725.json', 'r'))
word2id = dict_datas["word2id"]
id2word = dict_datas["id2word"]

def set_random_seeds(rand_seed=348, np_rand_seed=2345):
    random.seed(rand_seed)
    np.random.seed(np_rand_seed)

set_random_seeds()
# 忽略UserWarning
warnings.filterwarnings("ignore", category=FutureWarning)

def get_sub_set(nums):
    sub_sets = [[]]
    for i in range(len(nums)):
        x=nums[i]
        sub_sets.extend([item + [x] for item in sub_sets])
    sub_sets.pop(0)
    sub_sets.pop(-1)
    return sub_sets

def make_data(datas):
    train_datas = []
    for data in datas:
        data = data.strip()
        train_data = ['S']+[i for i in data] + ['E']
        train_datas.append(train_data)

    return train_datas


class MyDataSet(Data.Dataset):
    def __init__(self, datas):
        self.datas = datas

    def __getitem__(self, item):
        data = self.datas[item]
        decoder_input = data[:-1]
        decoder_output = data[1:]

        decoder_input_len = len(decoder_input)
        decoder_output_len = len(decoder_output)

        return {"decoder_input": decoder_input, "decoder_input_len": decoder_input_len,
                "decoder_output": decoder_output, "decoder_output_len": decoder_output_len}

    def __len__(self):
        return len(self.datas)

    def padding_batch(self, batch):
        decoder_input_lens = [d["decoder_input_len"] for d in batch]
        decoder_output_lens = [d["decoder_output_len"] for d in batch]

        decoder_input_maxlen = max(decoder_input_lens)
        decoder_output_maxlen = max(decoder_output_lens)

        for d in batch:
            d["decoder_input"].extend([word2id["<pad>"]] * (decoder_input_maxlen - d["decoder_input_len"]))
            d["decoder_output"].extend([word2id["<pad>"]] * (decoder_output_maxlen - d["decoder_output_len"]))
        decoder_inputs = torch.tensor([d["decoder_input"] for d in batch], dtype=torch.long)
        decoder_outputs = torch.tensor([d["decoder_output"] for d in batch], dtype=torch.long)
        return decoder_inputs, decoder_outputs


def epoch_time(start_time, end_time):
    elapsed_time = end_time - start_time
    elapsed_mins = int(elapsed_time / 60)
    elapsed_secs = int(elapsed_time - (elapsed_mins * 60))
    return elapsed_mins, elapsed_secs


def train_step(model, data_loader, optimizer, criterion, clip=1, print_every=None):
    model.train()

    if print_every == 0:
        print_every = 1

    print_loss_total = 0  # 每次打印都重置

    epoch_loss = 0

    for i, (dec_inputs, dec_outputs) in enumerate(tqdm(data_loader)):
        '''
        dec_inputs: [batch_size, tgt_len]
        dec_outputs: [batch_size, tgt_len]
        '''
        optimizer.zero_grad()
        dec_inputs, dec_outputs = dec_inputs.to(device), dec_outputs.to(device)
        # outputs: [batch_size * tgt_len, tgt_vocab_size]
        outputs, dec_self_attns = model(dec_inputs)

        loss = criterion(outputs, dec_outputs.view(-1))
        print_loss_total += loss.item()
        epoch_loss += loss.item()
        loss.backward()

        # 梯度裁剪
        torch.nn.utils.clip_grad_norm_(model.parameters(), clip)

        optimizer.step()

        if print_every and (i + 1) % print_every == 0:
            print_loss_avg = print_loss_total / print_every
            print_loss_total = 0
            print('\tCurrent Loss: %.4f' % print_loss_avg)

    return epoch_loss / len(data_loader)


def train(model, data_loader):
    criterion = nn.CrossEntropyLoss(ignore_index=0).to(device)
    optimizer = optim.Adam(model.parameters(), lr=1e-4)

    for epoch in range(epochs):
        start_time = time.time()
        train_loss = train_step(model, data_loader, optimizer, criterion, CLIP, print_every=10)
        end_time = time.time()

        torch.save(model.state_dict(), str(_DIR / 'GPT2.pt'))

        epoch_mins, epoch_secs = epoch_time(start_time, end_time)
        print(f'Epoch: {epoch + 1:02} | Time: {epoch_mins}m {epoch_secs}s')
        print(f'\tTrain Loss: {train_loss:.3f}')


def print_num_parameters(model):
    # Find total parameters and trainable parameters
    total_params = sum(p.numel() for p in model.parameters())
    print(f'{total_params:,} total parameters.')
    total_trainable_params = sum(

        p.numel() for p in model.parameters() if p.requires_grad)
    print(f'{total_trainable_params:,} training parameters.')

def delete_duplicate(sentence):
    all_sentence=[]
    temp_sentence=[]
    if sentence[-1]==1:
        sentence.pop(-1)
    for word in sentence:
        if word!=2:
            temp_sentence.append(word)
        else:
            all_sentence.append(temp_sentence)
            temp_sentence=[]
    all_sentence.append(temp_sentence)
    no_duplicate=[]
    true_duplicate=[]
    for sentence in all_sentence:
        if sorted(sentence[::2]) not in no_duplicate:
            no_duplicate.append(sorted(sentence[::2]))
            true_duplicate.append(sentence)
    concise_sentence=[]
    for sentence in true_duplicate:
        concise_sentence.extend(sentence)
        concise_sentence.append(2)
    concise_sentence.pop(-1)
    concise_sentence.append(1)
    return concise_sentence

def delete_dulplicate_A_column(A):
    A_delete=[]
    for i in range(A.shape[1]):
        if A[:,i].tolist() not in A_delete:
            A_delete.append(A[:,i].tolist())
        else:
            continue
    A_delete=np.array(A_delete).T
    return A_delete

def calculate_reward(ulti_sentence,Net,database,words2value,mask_invalid,variables,nx,nt):

    #ulti_sentence=[4, 2, 5, 3, 6, 2, 30, 1]

    ulti_sentence=delete_duplicate(ulti_sentence)
    vis_sentence = [id2word[int(id)] for id in ulti_sentence]
    # print("".join(vis_sentence))
    # print(ulti_sentence)


    terms=ulti_sentence[::2]
    operators=ulti_sentence[1::2]
    if max(operators)>4:
        reward=0
    elif len(operators)!=len(terms):
        reward=0
    else:
        A=[]
        A_column=1
        divide_flag=0
        for i in range(len(terms)):
            term=terms[i]
            operator=operators[i]
            word = id2word[term]
            if word not in words2value.keys():
                value = calculate_terms(word, Net, database,variables,nx,nt).reshape(-1, )
                contain_nan = (True in np.isnan(value))
                if contain_nan==True:
                    mask_invalid[term]=0
                words2value[word] = value
                if divide_flag==0:
                    A_column *= value
                else:
                    A_column /= value
                    divide_flag=0
            else:
                if divide_flag == 0:
                    A_column *= words2value[word]
                else:
                    A_column /= words2value[word]
                    divide_flag=0
            if operator==2:
                A.append(A_column)
                A_column=1
            elif operator==3:
                continue
            elif operator==4:
                divide_flag=1
            elif operator==1:
                A.append(A_column)


        #========delete inf===============
        A = np.vstack(A).T
        A=delete_dulplicate_A_column(A)
        df=pd.DataFrame(A)
        inf_index=df[df.isin([np.inf]).any(axis=1)].index.tolist()
        A=np.delete(A,inf_index,axis=0)
        df = pd.DataFrame(A)
        inf_index = df[df.isin([-np.inf]).any(axis=1)].index.tolist()
        A = np.delete(A, inf_index, axis=0)

        #A=np.hstack((A,np.ones_like(A[:,0].reshape([-1,1]))))

        #print(np.corrcoef(A.T))
        A_contain_nan = (True in np.isnan(A))
        if A_contain_nan==True:
            reward=0
        else:
            b = A[:, 0].copy()
            # we impose a condition that first term be 1,
            try:
                x = np.linalg.lstsq(A[:, 1:], -b)[0]
                #print(x)

                # u, d, v = np.linalg.svd(A, full_matrices=False)
                # coef_NN = v.T[:, -1] / (v.T[:, -1][0] + 1e-8)
                # coef_tt = -coef_NN[1:].reshape(coef_NN.shape[0] - 1, )
                # print(coef_tt)

                #x = np.r_[1, x]
                #print(x)
                # #x /= np.linalg.norm(x)
                RHS=A[:,1:].dot(x)
                LHS=-A[:,0]
                MSE=np.mean((RHS-LHS)**2)

                R2=1 - (((LHS -RHS) ** 2).sum() / ((LHS - LHS.mean()) ** 2).sum())
                #print(R2)

                MSE_true=MSE/np.linalg.norm(LHS)**2
                #reward = (1 - 0.3 * np.log10(A.shape[1])) * R2  #wave equation, PDE_compound
                reward=(1-0.02*np.log10(A.shape[1])) *R2
                #reward = (1 - 0.1 * np.log10(A.shape[1])) * R2
                #reward = (1-np.log10(A.shape[1])) / np.sqrt(MSE_true)
                #print(reward)
                if reward>1e4:
                    reward=0
            except numpy.linalg.LinAlgError:
                reward=0

    if ({word2id['u'], word2id['sin(u)']}.issubset(terms)
            or {word2id['u'], word2id['sinh(u)']}.issubset(terms)
            or {word2id['sin(u)'], word2id['sinh(u)']}.issubset(terms)
            or {word2id['u'],word2id['u^2'],word2id['u^3']}.issubset(terms)
            or {word2id['x'],word2id['sinx']}.issubset(terms)):
        reward=0

    vis_sentence = [id2word[int(id)] for id in ulti_sentence]
    for i in range(len(vis_sentence)):
        if vis_sentence[i]=='sqrt(x)':
            vis_sentence[i] = 'sqr(x)'
        if vis_sentence[i]=='sqrt(u)':
            vis_sentence[i] = 'sqr(u)'
    equation="".join(vis_sentence)
    #print(equation)
    for variable in variables:
        if variable in equation or 'Div' in equation or 'Laplace' in equation:
            continue
        else:
            reward=0






    return reward,words2value,mask_invalid

def get_mask_invalid(variables):
    mask_invalid = torch.ones(len(id2word)).to(device)
    for i in range(len(id2word)):
        if 't' in id2word[i]:
            mask_invalid[i] = 0
    if 't' not in variables:
        for i in range(len(id2word)):
            if 't' in id2word[i]:
                mask_invalid[i] = 0
    if 'x' not in variables:
        for i in range(len(id2word)):
            if 'x' in id2word[i]:
                mask_invalid[i] = 0

    if 'y' not in variables:
        for i in range(len(id2word)):
            if 'y' in id2word[i]:
                mask_invalid[i]=0
            if 'Laplace' in id2word[i]:
                mask_invalid[i]=0
            if 'BiLaplace' in id2word[i]:
                mask_invalid[i]=0
            if 'Div' in id2word[i]:
                mask_invalid[i]=0
    if 'z' not in variables:
        for i in range(len(id2word)):
            if 'z' in id2word[i]:
                mask_invalid[i] = 0
            if 'Div' in id2word[i]:
                mask_invalid[i]=0
    return mask_invalid

def find_min_no_repeat(all_reward,samples=400):
    best_index = torch.topk(all_reward, samples).indices.data.numpy().tolist()
    best_award = torch.topk(all_reward, samples).values.data.numpy().tolist()
    min_index=[]
    min_award=[]
    for i in range(len(best_award)):
        index=best_index[i]
        award=best_award[i]
        if award not in min_award:
            min_award.append(award)
            min_index.append(index)
        if len(min_award)==10:
            break

    return min_index,min_award

def get_concise_form(best_sentence,Net, database, words2value, mask_invalid, variables):
        sentence=best_sentence[0]
        reward,words2value,mask_invalid,A=calculate_reward(sentence, Net, database, words2value, mask_invalid, variables)
        print(reward)
        index_list=np.arange(1,A.shape[1]).tolist()
        print(index_list)
        sub_list=get_sub_set(index_list)
        for sub in sub_list:
            sub=[1,2]
            b = A[:, 0].copy()
            x = np.linalg.lstsq(A[:, sub].reshape([A.shape[0],len(sub)]), -b)[0]
            print(x)
            RHS = A[:, sub].reshape([A.shape[0],len(sub)]).dot(x)
            LHS = -A[:, 0]
            R2 = 1 - (((LHS - RHS) ** 2).sum() / ((LHS - LHS.mean()) ** 2).sum())
            reward = (1 - 0.1 * np.log10(len(sub)+1)) * R2
            print(reward, A[:, sub].reshape([A.shape[0],len(sub)]))
        print(index_list)
        print(sub_list)

        print(sentence)

if __name__ == '__main__':
    # device set by _device.py
    model_Q= GPT().to(device)
    model_Q.load_state_dict(load_checkpoint(str(_DIR / f'gpt_model/PDEGPT_{Equation_name}.pt')))

    optimize_epoch=5
    words2value={}
    variables=['t','x']
    Net, database,nx,nt=get_meta()
    mask_invalid=get_mask_invalid(variables)
    best_award_save=np.zeros([optimize_epoch,10])
    best_equation_save=[]
    best_sentence_save=[]

    try:
        os.makedirs(str(_DIR / f'result_save/{Equation_name}/{choose}_{noise_level}_{trail_num}/'))
    except OSError:
        pass

    # file = open(f"result_save/{Equation_name}/{choose}_{noise_level}_{noise_type}/equations.txt", 'w').close()
    # file=open(f"result_save/{Equation_name}/{choose}_{noise_level}_{noise_type}/equations.txt","a+")

    formula = 'ut,+,ux,*,u^2,+,ux,*,uxx,+,ux,+,uxxx,/,x'.split(',')
    sentence = [word2id[word] for word in formula]
    sentence.append(1)
    reward, words2value, mask_invalid = calculate_reward(sentence, Net, database, words2value, mask_invalid,
                                                         variables,nx,nt)
    print(sentence)
    print(reward)
    print('================================')

    formula = 'ut,+,ux'.split(',')
    sentence = [word2id[word] for word in formula]
    sentence.append(1)
    print(sentence)
    reward, words2value, mask_invalid = calculate_reward(sentence, Net, database, words2value, mask_invalid,
                                                         variables,nx,nt)
    print(reward)
    print('================================')

    # formula = 'ut,+,ux,+,env(u),*,ux'.split(',')  # env(u) not in vocab, skipped
    # formula = 'ut,+,uxxx,+,ux,+,u^2,*,uxxx'.split(',')  # skipped

    #get_concise_form([sentence], Net, database, words2value, mask_invalid, variables)

    #raise OSError

    # formula='(1/x^2),*,uyy,+,(1/x),*,ux,+,uxx'.split(',')
    # sentence=[word2id[word] for word in formula]
    # sentence.append(1)
    # print(sentence)
    # reward, words2value,mask_invalid = calculate_reward(sentence, Net, database, words2value,mask_invalid,variables)
    # print(reward)
    # raise OSError

    optimizer = optim.Adam(model_Q.parameters(), lr=1e-5)
    samples=400
    plt.figure(11,figsize=(4,2),dpi=300)
    for epoch in range(optimize_epoch):
        start_time=time.time()
        optimizer.zero_grad()
        #model_Q.eval()
        # 初始输入是空，每次加上后面的对话信息
        all_reward=torch.zeros([samples])
        all_sentence=[]
        for i in range(samples):
            #sentence = [5]
            sentence = [5,6,2]
            #model_Q.answer(sentence)
            while len(sentence)<max_pos-1:
                next_step,prob=model_Q.step(sentence,mask_invalid)
                sentence.append(next_step)
                if next_step==1:
                    break



            '''
            when calculate reward, delete the start symbol
            '''
            sentence.pop(0)
            reward,words2value,mask_invalid=calculate_reward(sentence,Net,database,words2value,mask_invalid,variables,nx,nt)
            all_reward[i]=reward
            all_sentence.append(sentence)

        #print(all_reward)
        if epoch==0:
            best_index,best_award=find_min_no_repeat(all_reward)
            best_sentence=[]
            for index in best_index:
                best_sentence.append(all_sentence[index])
            # print(best_sentence)
            # print(best_award)
        else:
            best_second_index,best_second_award=find_min_no_repeat(all_reward)
            for p_index in range(len(best_second_award)):
                potential_award=best_second_award[p_index]
                potential_text=all_sentence[best_second_index[p_index]]
                array = np.asarray(best_award)
                idx = np.where((array -potential_award) < 0)[0]
                if len(idx)==0:
                    continue
                elif np.isin(potential_award, array)==True:
                    continue
                else:
                    idx=idx[0]
                    best_sentence.insert(idx,potential_text)
                    best_award.insert(idx, potential_award)
                    best_sentence.pop(-1)
                    best_award.pop(-1)
            #print(best_award)

        #plt.plot(np.arange(0,len(best_award),1),best_award,marker='x',label=f'epoch={epoch}')


        continue_train_data=[]

        best_equation=[]
        for i in range(len(best_award)):
            sentence_data=best_sentence[i]
            sentence_data=delete_duplicate(sentence_data)
            sentence_data.pop(-1)
            if 5 not in sentence_data:
                sentence_data.insert(0,5)
                sentence_data.append(1)
            continue_train_data.append(sentence_data)
            #print(best_sentence[i], best_award[i])
            vis_sentence = [id2word[int(id)] for id in sentence_data]
            print("".join(vis_sentence[1:-1]), best_award[i])
            #file.write("".join(vis_sentence[1:-1])+ f"\t  {best_award[i]}")
            best_equation.append("".join(vis_sentence[1:-1]))
            best_award_save[epoch,i]=best_award[i]
        best_equation_save.append(best_equation)
        best_sentence_save.append(continue_train_data)
        #print(continue_train_data)

        #==============train model====================
        batch_size = len(continue_train_data)
        dataset = MyDataSet(continue_train_data)

        data_loader = Data.DataLoader(dataset, batch_size=batch_size, collate_fn=dataset.padding_batch)

        model = GPT().to(device)

        # model.load_state_dict(torch.load('GPT2.pt'))

        criterion = nn.CrossEntropyLoss(ignore_index=0).to(device)
        for i in range(5):
            train_loss = train_step(model, data_loader, optimizer, criterion, CLIP, print_every=10)
        end_time = time.time()
        print(end_time-start_time)
    best_award_save=pd.DataFrame(best_award_save)
    best_equation_save=pd.DataFrame(best_equation_save)

    best_award_save.to_csv(str(_DIR / f'result_save/{Equation_name}/{choose}_{noise_level}_{trail_num}/awards.csv'))
    best_equation_save.to_csv(str(_DIR / f'result_save/{Equation_name}/{choose}_{noise_level}_{trail_num}/equations.csv'))
    pickle.dump(best_sentence_save,open(str(_DIR / f'result_save/{Equation_name}/{choose}_{noise_level}_{trail_num}/sentences.pkl'), 'wb'))

    # font1egend = {'family': 'Arial',
    #               'weight': 'normal',
    #               # "style": 'italic',
    #               'size': 7.5,
    #               }
    # plt.legend(prop=font1egend)
    # plt.xticks(fontproperties='Arial', size=8)
    # plt.yticks(fontproperties='Arial', size=8)
    # plt.tight_layout()
    # plt.show()







