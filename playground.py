import numpy as np
from hmmlearn import hmm
"""
arr = []
# this is the case for multi-feature. In this case 4 features. where the data is list of (n_words X n_features)
arr.append(np.array([[1,2,3,4],[2,4,6,8]]))
print(arr[0].shape)
arr.append(np.array([5,6,7,8]))
arr.append(np.array([9,10,11,12]))
arr.append(np.array([20,21,22,23]))
print(arr[0])
sentence1 = []
for word in arr[0]:
    print(word)
    sentence1.append(word)
print(np.vstack(sentence1))



data_hmmlearn_formatted = np.transpose(np.vstack(arr))
print(data_hmmlearn_formatted)
print(data_hmmlearn_formatted.shape)

print("\nnew test")

arr2 = []
arr2.append(np.atleast_2d(np.array([1,2,3,4,5])))
arr2.append(np.atleast_2d(np.array([6,7,8])))
print(arr2[0].shape)
data_hmmlearn_formatted = np.transpose(np.vstack(arr2))
data_hmmlearn_formatted = np.squeeze(data_hmmlearn_formatted)
print(data_hmmlearn_formatted)

"""
"""
arr2 = []
arr2.append(np.array([1,2,3,4,5]))
arr2.append(np.array([6,7,8]))
sentences_length = [int(sentence.shape[0]) for sentence in arr2]
print(sentences_length)
"""
model = hmm.GaussianHMM()
model.fit(np.array([1,2,3,4,5,6,7,8,9,10]).reshape(-1,1),[3,4,3])
print(model.startprob_)

