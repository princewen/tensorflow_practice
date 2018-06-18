import numpy as np

def ndcg(pred_rel):
    dcg = 0
    for (index,rel) in enumerate(pred_rel):
        dcg += (rel * np.reciprocal(np.log2(index+2)))
    print("dcg " + str(dcg))
    idcg = 0
    for(index,rel) in enumerate(sorted(pred_rel,reverse=True)):
        idcg += (rel * np.reciprocal(np.log2(index+2)))
    print("idcg " + str(idcg))


    return dcg/idcg


pred_rel = [3,1,2,3,2]
print("ndcg " + str(ndcg(pred_rel)))

