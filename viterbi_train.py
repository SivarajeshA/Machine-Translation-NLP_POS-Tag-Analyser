
import numpy as np
import os

# file for separating words and tags
file2_write = open("preprocess.txt", 'w')

# sets contain unique tags and words
tag_set = set()
word_set = set()

with open("train.txt", 'r') as file1_read:
    lines = file1_read.readlines()
    for line in lines:
        line = line.replace("\n", "")   
        list = line.split('@')
        words = "^ "       # ^ for denoting start of sentence
        tags = "^ "
        for unit in list:
            tempStr = unit.split('#')  
            tempStr[0] = tempStr[0].lower()   # converting tags and words into lower case
            tempStr[1] = tempStr[1].lower()

            if (tempStr[1] != "."):      # ignoring the words which haves '.' tag
                word_set.add(tempStr[0])     #inserting into sets
                tag_set.add(tempStr[1])
                words = words + tempStr[0] + " "   # making sentence from words
                tags = tags + tempStr[1] + " "     # tags associated with sentence

        if len(words)>1:
            words = words.strip()    #remove extra space from end
            tags = tags.strip()
            file2_write.write(words + "\n")    # storing in file
            file2_write.write(tags + "\n")
        
file1_read.close()
file2_write.close()

# print(tag_set)
# print(word_set)



# making dictionary of unique tags and words so that we can map tags or words by their index to access matrices

tag_dictionary = {'^':0}
word_dictionary = {'^':0}

count = 1
for item in tag_set:
    tag_dictionary[item] = count
    count += 1

number_of_unique_tag = len(tag_dictionary)

# print(tag_dictionary)
# print(number_of_unique_tag)

count = 1
for item in word_set:
    word_dictionary[item.lower()] = count
    count += 1
word_dictionary['1to9'] = count   # 1to9 is a word for representing all words having 'num' tag
count += 1
number_of_unique_word = len(word_dictionary)

tag_set.add('^')
word_set.add('^')

# print(word_dictionary)
# print(number_of_unique_word)



# matrix for bigram probabilities
bigram_matrix = np.zeros((number_of_unique_tag, number_of_unique_tag))
# matrix for lexical probabilities
lexical_matrix = np.zeros((number_of_unique_tag, number_of_unique_word))
# array to store count of particular tag in our training set
tag_count = np.zeros(number_of_unique_tag)

with open("preprocess.txt", 'r') as file1_read:
    lines = file1_read.readlines()
    i = 0
    n = len(lines)
    while i < n:
        line1 = lines[i]
        line2 = lines[i+1]
        
        line1 = line1.replace("\n", "")
        line2 = line2.replace("\n", "")
        
        list1 = line1.split(' ')
        list2 = line2.split(' ')
        
        i = i + 2
        
        for j in range(1, int(len(list2))):
            row_i = tag_dictionary[list2[j-1]]
            col_i = tag_dictionary[list2[j]]
            bigram_matrix[row_i][col_i] = bigram_matrix[row_i][col_i] + 1   #bigram count
            if(j == 1):
                tag_count[tag_dictionary['^']] = tag_count[tag_dictionary['^']] + 1
                
            tag_count[row_i] = tag_count[row_i] + 1
            
            row_j = tag_dictionary[list2[j]]
            col_j = word_dictionary[list1[j]]
            lexical_matrix[row_j][col_j] = lexical_matrix[row_j][col_j] + 1   #lexical count
            if list2[j]=='num':
                ind1 = tag_dictionary['num']
                ind2 = word_dictionary['1to9']
                lexical_matrix[ind1][ind2] = lexical_matrix[ind1][ind2] + 1
            
            
for item in tag_set:
    index = tag_dictionary[item]
    bigram_matrix[index][:] = bigram_matrix[index][:] / tag_count[index]     #bigram probability
    lexical_matrix[index][:] = lexical_matrix[index][:] / tag_count[index]   #lexical probability

    
# converting all zero probability into a fix number so that probaility will not vanish
for i in range(number_of_unique_tag):
    for j in range(number_of_unique_tag):
        if bigram_matrix[i][j] < 0.0000001:
            bigram_matrix[i][j] = 0.0000001
            
for i in range(number_of_unique_tag):
    for j in range(number_of_unique_word):
        if lexical_matrix[i][j] < 0.0000001:
            lexical_matrix[i][j] = 0.0000001
            

# print(bigram_matrix)
# print(lexical_matrix)
# print(tag_count)


# creating model

os.system('rm -rf model')
os.mkdir("model")     # create folder named model 

# storing unique tags and words

file_write = open("./model/unique_tags_words.txt", 'w')
file_write.write(str(number_of_unique_tag) + '\n')
for key, val in tag_dictionary.items():
    file_write.write(str(key) + '\n')
    file_write.write(str(val) + '\n')

file_write.write(str(number_of_unique_word) + '\n')
for key, val in word_dictionary.items():
    file_write.write(str(key) + '\n')
    file_write.write(str(val) + '\n')

file_write.close()

# storing bigram probabilities

file_write = open("./model/bigram.txt", 'w')
for i in range(number_of_unique_tag):
    for j in range(number_of_unique_tag):
        if j != number_of_unique_tag - 1:
            file_write.write(str(bigram_matrix[i][j]) + " ")
        else:
            file_write.write(str(bigram_matrix[i][j]) + "\n")

file_write.close()

# storing lexical probabilities

file_write = open("./model/lexical.txt", 'w')
for i in range(number_of_unique_tag):
    for j in range(number_of_unique_word):
        if j != number_of_unique_word - 1:
            file_write.write(str(lexical_matrix[i][j]) + " ")
        else:
            file_write.write(str(lexical_matrix[i][j]) + "\n")
            
file_write.close()

# storing count of tags

file_write = open("./model/tag_count.txt", 'w')
for i in range(number_of_unique_tag):
    if i != number_of_unique_tag - 1:
        file_write.write(str(tag_count[i]) + " ")
    else:
        file_write.write(str(tag_count[i]))

file_write.close()


