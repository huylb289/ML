import numpy as np
import matplotlib.pyplot as plt
from matplotlib import style
import warnings
from math import sqrt
from collections import Counter

style.use('fivethirtyeight')


def k_nearest_neighbors(data, predict, k=3):
    '''
    data: dataset = {'k': [[1,2],[2,3],[3,1]], 'r': [[6,5],[7,7],[8,6]]}
    predict: predict center of cluster
    k: the top k nearest elements
    '''
    if len(data) >=k:
        warnings.warn('K is set to a value less than total voting groups!')

    distances = []
    for group in data:
        for features in data[group]:
            # using Numpy version
##            euclidean_distance = sqrt((features[0]-predict[0])**2 + (features[1]-predict[1])**2)
            # using efficient library
            euclidean_distance = np.linalg.norm(np.array(features) - np.array(predict))
            print('euclidean_distance: {}'.format(euclidean_distance))
            distances.append([euclidean_distance, group])

    print ("distances={}".format(distances))    
    # get the first k nearest elements
    votes = [i[1] for i in sorted(distances)[:k]]
    print ("votes={}".format(votes))
    # Collections finds the most common elements. In our case, we just want the single most common, but you can find the top 3 or top 'x.' Without doing the [0][0] part, you get [('r', 3)]. Thus, [0][0] gives us the first element in the tuple.
    vote_result = Counter(votes).most_common(1)[0][0]
    print ("vote_result={}".format(vote_result))
    return vote_result

dataset = {'k': [[1,2],[2,3],[3,1]], 'r': [[6,5],[7,7],[8,6]]}
new_feature = [5,7]


# Simple form
[[plt.scatter(ii[0], ii[1], s=100, color=i) for ii in dataset[i]] for i in dataset]

# for loop form
for i in dataset:
    for ii in dataset[i]:
        plt.scatter(ii[0], ii[1], s=100, color=i)

plt.scatter(new_feature[0], new_feature[1], s=200)
result = k_nearest_neighbors(dataset, new_feature)
plt.scatter(new_feature[0], new_feature[1], s=100, color=result)

plt.show()
