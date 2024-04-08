import jittor as jt

PackedSequence = None

def pad_sequence(sequences,batch_first=False,padding_value=0.0):
    max_f = max([len(s) for s in sequences])
    # max_f = 512
    b = len(sequences)
    if batch_first:
        ret = sequences[0].new_full([b,max_f,]+list(sequences[0].shape[1:]),padding_value)
        for i,s in enumerate(sequences):
            ret[i,:len(s)] = s 
    else:
        ret = sequences[0].new_full([max_f,b,]+list(sequences[0].shape[1:]),padding_value)
        for i,s in enumerate(sequences):
            ret[:len(s),i] = s
    # print(ret.shape)
    # ret = ret[:,:406]
    return ret 
    