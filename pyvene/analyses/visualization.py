import seaborn
import torch

def rotation_token_heatmap(rotate_layer, 
                           tokens, 
                           token_size, 
                           variables, 
                           intervention_size):

    W = rotate_layer.weight.data
    in_dim, out_dim = W.shape

    assert in_dim % token_size == 0
    assert in_dim / token_size >= len(tokens) 

    assert out_dim % intervention_size == 0
    assert out_dim / intervention_size >= len(variables) 
    
    heatmap = []
    for j in range(len(variables)):
        row = []
        for i in range(len(tokens)):
            row.append(torch.norm(W[i*token_size:(i+1)*token_size, j*intervention_size:(j+1)*intervention_size]))
        mean = sum(row)
        heatmap.append([x/mean for x in row])
    return seaborn.heatmap(heatmap, 
                    xticklabels=tokens, 
                    yticklabels=variables)