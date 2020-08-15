

import numpy as np

# loading model for testing

tag_dictionary = {}
word_dictionary = {}
tag_set = set()
word_set = set()
file_read = open("./model/unique_tags_words.txt", 'r')
number_of_unique_tag = int(file_read.readline())


for i in range(number_of_unique_tag):
    line = file_read.readline()
    key = line.replace("\n", "")   
    line = file_read.readline()
    val = line.replace("\n", "")
    tag_dictionary[key] = int(val)
    tag_set.add(key)


number_of_unique_word = int(file_read.readline())

for i in range(number_of_unique_word):
    line = file_read.readline()
    key = line.replace("\n", "")   
    line = file_read.readline()
    val = line.replace("\n", "")
    word_dictionary[key] = int(val)
    word_set.add(key)

file_read.close()

# print(number_of_unique_tag)
# print(tag_dictionary)
# print(number_of_unique_word)
# print(word_dictionary)

# matrix for bigram probabilities
bigram_matrix = np.zeros((number_of_unique_tag, number_of_unique_tag))
# matrix for lexical probabilities
lexical_matrix = np.zeros((number_of_unique_tag, number_of_unique_word))
# array to store count of particular tag in our training set
tag_count = np.zeros(number_of_unique_tag)

i = 0
with open("./model/bigram.txt", 'r') as file1_read:
    lines = file1_read.readlines()
    for line in lines:
        line = line.replace("\n", "")   
        list = line.split(' ')
        j = 0
        for prob in list:
            bigram_matrix[i][j] = float(prob)
            j += 1
        i += 1
        
file1_read.close()

i = 0
with open("./model/lexical.txt", 'r') as file1_read:
    lines = file1_read.readlines()
    for line in lines:
        line = line.replace("\n", "")   
        list = line.split(' ')
        j = 0
        for prob in list:
            lexical_matrix[i][j] = float(prob)
            j += 1
        i += 1
        
file1_read.close()

# print(len(lexical_matrix[0]))

file_read = open("./model/tag_count.txt", 'r')
line = file_read.readline()
line = line.replace("\n", "")   
list = line.split(' ')

for i in range(number_of_unique_tag):
    tag_count[i] = float(list[i])




# testing

# return key corresponding to a value in dictionary
def get_tag(val):
    for key, value in tag_dictionary.items():
        if (val == value):
            return key
        
char_list1 = ['0','1','2','3','4','5','6','7','8','9']
char_list2 = ['-','$','\'','%']
list_count = ["zero", "one", "two", "three", "four","five", "six", "seven", "eight", "nine",
              "ten", "eleven", "twelve", "thirteen", "fourteen", "fifteen", "sixteen", "seventeen", 
              "eighteen", "nineteen", "twenty", "thirty", "forty", "fifty", "sixty", "seventy", 
              "eighty", "ninety", "hundred", "thousand"]


# giving tag to unknown word(not in traing set)
# first, we check a word for 'num' tag otherwise we marked it as a 'noun'

tag_count_correct = np.zeros(number_of_unique_tag)    # to count correct prediction
tag_count_total = np.zeros(number_of_unique_tag)      # to count total actual tags 
tag_count_predict_total = np.zeros(number_of_unique_tag)  # to count total predicted tags

def check_unknow_tag(word):
    list_w = word.split(' ')
    
    if( any(c in word for c in char_list1) and (any(c in word for c in char_list2) != 1) 
        or any(w in list_w for w in list_count)):
        index = word_dictionary['1to9']
        return index
    else:
        max_prob = lexical_matrix[tag_dictionary['noun']][0]
        max_index = 0
        for j in range(1, number_of_unique_word):
            if lexical_matrix[tag_dictionary['noun']][j] > max_prob:
                max_prob = lexical_matrix[tag_dictionary['noun']][j]
                max_index = j
        return max_index
        
# viterbi algorithm
def test(test_sentence):
    test_sentence = test_sentence.replace("\n", "")
    list = test_sentence.split('@')
    words = []
    tags_actual = []
    for unit in list:
        tempStr = unit.split('#')
        tempStr[0] = tempStr[0].lower()
        tempStr[1] = tempStr[1].lower()
        if (tempStr[1] != "."):
            words.append(tempStr[0])
            tags_actual.append(tempStr[1])

    total_tagged = 0
    
    if len(words)>1:
        SEQSCORE = np.zeros((number_of_unique_tag, len(words)))
        BACKPTR = np.zeros((number_of_unique_tag, len(words)))

        for tag in tag_set:
            if words[0] in word_dictionary.keys():
                index1 = tag_dictionary[tag]
                index2 = word_dictionary[words[0]]
                SEQSCORE[index1][0] = bigram_matrix[tag_dictionary['^']][index1] * lexical_matrix[index1][index2]
                BACKPTR[index1][0] = -1
            else:
                index2 = check_unknow_tag(words[0])
                index1 = tag_dictionary[tag]
                SEQSCORE[index1][0] = bigram_matrix[tag_dictionary['^']][index1] * lexical_matrix[index1][index2]
                BACKPTR[index1][0] = -1

        for i in range(1, len(words)):
            if words[i] in word_dictionary.keys():
                index3 = word_dictionary[words[i]]
            else:
                index3 = check_unknow_tag(words[i])

            for tag in tag_set:
                index2 = tag_dictionary[tag]
                max_prob = 0.0
                for temp_tag in tag_set: 
                    index1 = tag_dictionary[temp_tag]            
                    temp = SEQSCORE[index1][i-1] * bigram_matrix[index1][index2] * lexical_matrix[index2][index3]
                    if temp > max_prob:
                        max_prob = temp
                        SEQSCORE[index2][i] = max_prob
                        BACKPTR[index2][i] = index1

        # sequence identification step
        max_prob = 0.0
        T = len(words)
        C = np.zeros(T)
        T = T-1

        for i in range(number_of_unique_tag):
            if SEQSCORE[i][T] > max_prob:
                max_prob = SEQSCORE[i][T]
                C[T] = i

        for i in range(len(words) - 1):
            T = T-1
            C[T] = BACKPTR[int(C[T+1])][T+1]

        tags_predict = []
        for i in range(len(words)):
            tags_predict.append(get_tag(C[i]))

#         print()
#         print(words)
#         print(tags_actual)
#         print(tags_predict)

        total_tagged = len(tags_actual)
        for i in range(total_tagged):
            tag_count_total[tag_dictionary[tags_actual[i]]] += 1
            tag_count_predict_total[tag_dictionary[tags_predict[i]]] += 1
            
            if(tags_actual[i] == tags_predict[i]):
                tag_count_correct[tag_dictionary[tags_actual[i]]] += 1
        

def predict():
    with open("test1.txt", 'r') as file1_read:    # change filename for testing 
        lines = file1_read.readlines()
        
        for line in lines:
            test(line)
    
    file_write = open("output.txt", 'w')
    
    total_predict = 0
    total_actual = 0
    total_correct = 0
    
    for i in range(1, number_of_unique_tag):
        
        total_predict += tag_count_predict_total[i]
        total_actual += tag_count_total[i]
        total_correct += tag_count_correct[i]
        
        if tag_count_total[i] > 0:
            recall = tag_count_correct[i]/tag_count_total[i]
        else:
            recall = 0
        if tag_count_predict_total[i] > 0:
            precision = tag_count_correct[i]/tag_count_predict_total[i]
        else:
            precision = 0
        
        if (recall + precision) > 0:
            fscore = 2*(recall*precision)/(recall + precision)
        else:
            fscore = 0
        
        file_write.write("Tag: " + get_tag(i) + "\n")
        file_write.write("Precision: " + str(precision) + "\n")
        file_write.write("Recall: " + str(recall) + "\n")
        file_write.write("F-score: " + str(fscore) + "\n")
    
    
    recall = total_correct/total_actual
    precision = total_correct/total_predict
    fscore = 2*(recall*precision)/(recall + precision)
    
    file_write.write("Overall:\n")
    file_write.write("Precision: " + str(precision) + "\n")
    file_write.write("Recall: " + str(recall) + "\n")
    file_write.write("F-score: " + str(fscore) + "\n")

    file_write.close()
    file1_read.close()


predict()





