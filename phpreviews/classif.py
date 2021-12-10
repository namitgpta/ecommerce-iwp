import os
import numpy as np
from collections import Counter
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import confusion_matrix


def make_Dictionary(root_dir):
    data_dirs = [os.path.join(root_dir,f) for f in os.listdir(root_dir)]    
    all_words = []       
    for data_dir in data_dirs:
        dirs = [os.path.join(data_dir,f) for f in os.listdir(data_dir)]
        for d in dirs:
            messages = [os.path.join(d,f) for f in os.listdir(d)]
            for message in messages:
                with open(message, encoding='utf-8') as m:
                    for line in m:
                        words = line.split()
                        all_words += words
    dictionary = Counter(all_words)
    list_to_remove = list(dictionary.keys())
    
    for item in list_to_remove:
        if item.isalpha() == False: 
            del dictionary[item]
        elif len(item) == 1:
            del dictionary[item]
    dictionary = dictionary.most_common(3000)
    
    np.save('dictionary.npy',dictionary) 
    
    return dictionary
    
def extract_features(root_dir): 
    data_dirs = [os.path.join(root_dir,f) for f in os.listdir(root_dir)]  
    docID = 0
    features_matrix = np.zeros((16541,3000))
    train_labels = np.zeros(16541)
    for data_dir in data_dirs:
        dirs = [os.path.join(data_dir,f) for f in os.listdir(data_dir)]
        for d in dirs:
            messages = [os.path.join(d,f) for f in os.listdir(d)]
            for messageFile in messages:
                with open(messageFile, encoding='utf-8') as m:
                    all_words = []
                    for line in m:
                        words = line.split()
                        all_words += words
                    for word in all_words:
                      wordID = 0
                      for i,d in enumerate(dictionary):
                        if d[0] == word:
                          wordID = i

                          # the number of occurrences of a substring in the given string
                          features_matrix[docID,wordID] = all_words.count(word)

                # labels: True -> 1, False -> 0
                train_labels[docID] = int(messageFile.split(".")[-2] == 'spam')  
                docID = docID + 1                
    return features_matrix, train_labels
    
#Create a dictionary of words with its frequency

root_dir = 'data'
dictionary = make_Dictionary(root_dir)


# Prepare feature vectors per training mail and its labels
# features_matrix, train_labels
features_matrix,labels = extract_features(root_dir) 
np.save('features_matrix.npy', features_matrix)
np.save('labels.npy',labels)

# 5172 rows, 3000 cols (5172, 3000)
print (features_matrix.shape)

print (labels.shape)

# e.g 3672 1500 (ham, spam) sum in dictionary
print (sum(labels==0),sum(labels==1))
X_train, X_test, y_train, y_test = train_test_split(features_matrix, labels, test_size=0.40)

## Training models and its variants
model = MultinomialNB()

# train
model.fit(X_train,y_train)

result = model.predict(X_test)

# Confustion Matrix: The number of correct and incorrect predictions are summarized with count
# values and broken down by each class.
"""
example confusion matrix output with 5172 messages

- MultinomialNB
    [[1380   77]
    [  23  589]]

+----------------+-----+------+
| Multinomial NB | Ham | Spam |
+----------------+-----+------+
| Ham            |1380 |  77  |
+----------------+-----+------+
| Spam           | 23  |  589 |
+----------------+-----+------+


"""
print (confusion_matrix(y_test, result))

