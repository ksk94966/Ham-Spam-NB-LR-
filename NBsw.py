# -*- coding: utf-8 -*-
"""
Created on Sun Oct 20 21:41:49 2019

@author: SAI KRISHNA
"""

# -*- coding: utf-8 -*-
"""
Created on Wed Oct 16 11:58:37 2019

@author: SAI KRISHNA
"""

#import sys
import os
import string
import pandas as pd
import math
import sys

stopwords = [] 

def loaddatafromfiles(path):
    files = []
    for r, d, f in os.walk(path):
        for file in f:
            if '.txt' in file:
                files.append(os.path.join(r, file))
    return files

def getWords(Files):
    Wordslist = []
    for file in Files:
        f = open(file,'r',errors ='ignore')     #Here getting encoding error for charpmap so wrote error ignore
        words = f.read().replace('\n',' ')
        Wordslist.extend(words.split())
    Wordslist = [''.join(c for c in s if c not in string.punctuation) for s in Wordslist]   #Ref Stackoverflow
    Wordslist = [x for x in Wordslist if x]
    Wordslist = [x for x in Wordslist if x not in stopwords]  #Removing Stop words
    return Wordslist
    

def Calunique(Words):              #if we use list comprehension over here we get O(n square) Complexity
    stats = {}
    for i in Words:
        if i in stats:
            stats[i] += 1
        else:
            stats[i] = 1
    #print(stats)
    return(stats)
        

def CalCondtionProb(Words,label,CombinedWords):     #From the given formula for CP in Naive Bayes(NB)
    FinalCondProb = {} 
    b = len(CombinedWords)                      
    DictWords = Calunique(Words)
    for word in CombinedWords:
        if word in DictWords:
            FinalCondProb[word+':'+label] =  float((DictWords[word] + 1)/(len(Words) + b)) #if word found in class
        else:
            FinalCondProb[word+':'+label] =  float(1/(len(Words) + b))   #if word not found in that class
    return FinalCondProb

def NaiveBayes(Prior,TestPathFiles,Condprob,truelabel):
    CorrectPredict = 0;
    Incorrectpredict = 0;
    for file in TestPathFiles:
        WordsInFile= []
        f = open(file,'r',errors ='ignore')     #Here getting encoding error for charpmap so wrote error ignore
        words = f.read().replace('\n',' ')
        WordsInFile.extend(words.split())
        WordsInFile = [''.join(c for c in s if c not in string.punctuation) for s in WordsInFile]   #Ref Stackoverflow
        WordsInFile = [x for x in WordsInFile if x]
        WordsInFile = [x for x in WordsInFile if x not in stopwords]  #Removing Stop words
        WordsInFileUnique = Calunique(WordsInFile)
        #print(WordsInFileUnique)
        #print("\n")
        #print(WordsInFileUnique)
        label = ['ham','spam']
        predval = {}
        for lb in label:
            if (lb == 'ham'):
                pr = Prior[0]
            else:
                pr = Prior[1]
            sumprob = 0
            
            for w,c in WordsInFileUnique.items():
                if (w+':'+lb) in Condprob:
                    sumprob += c*(math.log(Condprob[w+':'+lb]))
            
            predval[lb] = sumprob + math.log(pr) 
            #print(predval)
        max_value = max(predval.values())  # maximum value
        #print(max_value)
        predlabel = (([k for k, v in predval.items() if v == max_value]))
        if (predlabel[0]==truelabel):
            CorrectPredict = CorrectPredict + 1
        else:
            Incorrectpredict = Incorrectpredict + 1
    return CorrectPredict, Incorrectpredict
            

if __name__=='__main__':

    #Getting Stop words
    
    pathstop = '.\\stopwords.txt'
    f = open(pathstop, 'r')  #read stop words from text file
    for line in f.readlines():
        stopwords.append(line.strip())
    #print(len(stopwords))
    print("Using Stop Words")
    #Taking the input argument(argv[1],argv[2] files when compiling the pythongfile
    
    HamTrainFiles = loaddatafromfiles(sys.argv[1]+'/ham')
    SpamTrainFiles = loaddatafromfiles(sys.argv[1]+'/spam')
    
    #print(len(HamTrainFiles + SpamTrainFiles))
    
    HamTrainWords  =  getWords(HamTrainFiles)
    SpamTrainWords = getWords(SpamTrainFiles)
    
    #print(HamTrainWords)
    CombinedWords = HamTrainWords + SpamTrainWords
    
    #print(CombinedWords)
    UniqueCombinedWords = list(dict.fromkeys(CombinedWords))  #getting unique words by eliminating duplicates
    
    #print(UniqueCombinedWords)
    #Lets calculate the Condition probabilties for each term
    CondProbHam = dict()
    CondProbSpam = dict()
    
    CondProbHam = CalCondtionProb(HamTrainWords,'ham',UniqueCombinedWords)  #Passing all HamWords for Calc Probabilties
    CondProbSpam = CalCondtionProb(SpamTrainWords,'spam',UniqueCombinedWords)  #Passing all SpamWords for Calc Probabilties
    #print(CondProbSpam)
    
    
    CondProbAll = {**CondProbHam,**CondProbSpam}    #Merging two Dictionaries
    
    #print(CondProbAll)
    
    #print(pd.DataFrame(list(CondProbAll.items())))    #To Display in  a DataFrame
    
    #Calculating Priors
    
    HamPrior = len(HamTrainFiles)/(len(HamTrainFiles)+len(SpamTrainFiles))
    
    SpamPrior = len(SpamTrainFiles)/(len(HamTrainFiles)+len(SpamTrainFiles))
    
    #print(HamPrior) print(SpamPrior)
    
    #Now Lets calculate the Correct and Inocrrect predictions using Naive Bayes
    
    #Lets get the test path
    
    HamTestFiles = loaddatafromfiles(sys.argv[2]+'/ham')
    SpamTestFiles = loaddatafromfiles(sys.argv[2]+'/spam')
    
    Prior = [HamPrior , SpamPrior]
    
    #print(Prior)
    
    CorPredHam, InCorPredHam = NaiveBayes(Prior,HamTestFiles,CondProbAll,'ham') 
    
    AccuracyHam = CorPredHam/(CorPredHam + InCorPredHam)
    
    CorPredSpam, InCorPredSpam = NaiveBayes(Prior,SpamTestFiles,CondProbAll,'spam') 
    
    AccuracySpam = CorPredSpam/(CorPredSpam+InCorPredSpam)
    
    total_accuracy = (CorPredHam + CorPredSpam)/(CorPredHam + InCorPredHam + CorPredSpam+InCorPredSpam)
    
    print("TotalAccuracy:",total_accuracy*100)  #In terms of percentages
    
    print("HamAccuracy:",AccuracyHam*100)
    
    print("SpamAccuracy:",AccuracySpam*100)
      
     