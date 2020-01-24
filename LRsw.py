# -*- coding: utf-8 -*-
"""
Created on Fri Oct 18 12:17:35 2019

@author: SAI KRISHNA
"""
import os
import string
import pandas as pd
import math
import numpy as np
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

def Sigmoid(x):
    s = 1.0/(1 + np.exp(-x))
    return s

def ChangeWeights(lam,iter,weights,FileWords,TrueLabels):
                     #given Lambda values 0.001 and 0.1
        for j in range(iter):               #given 50 and 100 iterations
            a=np.array(np.dot(FileWords,weights),dtype=np.float32)
            S=Sigmoid(a)
            Labels=TrueLabels
            g=np.dot(FileWords.transpose(),Labels-S)   #(L-S) is the error over here
            weights=weights+(0.001*(g))-(0.001*lam*weights)   #Here 0.01 is the learning rate
        return weights

def LogRegClassification(weights,Testarray,label):
    
    #print(Testarray.shape)
    y=np.array(np.dot(Testarray,weights),dtype=np.float32)
    correctham =0
    correctspam = 0
    Incorrectham = 0
    Incorrectspam = 0
    predicted_label=[]
    #print(y)
    #print(y.shape)
    #print(y.size)
    for i in y:
        if i>0:
            predicted_label.append(1)
        else:
            predicted_label.append(0)
    #print(predicted_label)
    #print(label)
    #print(len(label))
    for i in range(len(label)):
        if label[i]==predicted_label[i]:
            if label[i]==1:
                correctham += 1
            else:
                correctspam += 1
        else:        
            if predicted_label[i]==1:
                Incorrectham += 1
            else:
                Incorrectspam += 1
    #print(correctham,correctspam,Incorrectham,Incorrectspam)
    return correctham,correctspam,Incorrectham,Incorrectspam
        
    
    

if __name__=='__main__':
    
    
    pathstop = '.\\stopwords.txt'
    f = open(pathstop, 'r')  #read stop words from text file
    for line in f.readlines():
        stopwords.append(line.strip())
    #print(len(stopwords))
    print("Using Stop Words")
    
    #Getting Files Path
    
    HamTrainFiles = loaddatafromfiles(sys.argv[1]+'/ham')
    SpamTrainFiles = loaddatafromfiles(sys.argv[1]+'/spam')
    
    TrainFiles = HamTrainFiles + SpamTrainFiles
    
    #print(len(TrainFiles))
    
    HamTrainWords  =  getWords(HamTrainFiles)
    SpamTrainWords = getWords(SpamTrainFiles)
    
    #print(HamTrainWords)
    CombinedWords = HamTrainWords + SpamTrainWords
    
    #print(CombinedWords)
    UniqueCombinedWords = list(dict.fromkeys(CombinedWords))
    #eliminating the duplicate words
    #print(len(UniqueCombinedWords))
    
    count =0;
    EachFileWordsDict = dict()  #for holding each File words (all train files namely Ham and Spam)
    for file in TrainFiles:
        Wordslist=[]
        f = open(file,'r',errors ='ignore')     #Here getting encoding error for charpmap so wrote error ignore
        words = f.read().replace('\n',' ')
        Wordslist.extend(words.split())
        Wordslist = [''.join(c for c in s if c not in string.punctuation) for s in Wordslist]   #Ref Stackoverflow
        Wordslist = [x for x in Wordslist if x]
        Wordslist = [x for x in Wordslist if x not in stopwords]  #Removing Stop words
        row=[0]*len(UniqueCombinedWords)
        for i in range(len(UniqueCombinedWords)):        
            if UniqueCombinedWords[i] in Wordslist:
                feature_count=Wordslist.count(UniqueCombinedWords[i])
                row[i]=feature_count               #Here we are labling as 1 if word is present or else as zero
            else:
                row[i] = 0
        row.insert(0,1)                #add one more column infront to indicate the feature 
        EachFileWordsDict[count]=row
        count+=1
        
    #print(EachFileWordsDict)
    
    labels = []             #We have to set original label for the given files say Ham as '1' and Spam as '0'
    for i in range(len(HamTrainFiles)):
        labels.append(1)
    for i in range(len(SpamTrainFiles)):
        labels.append(0)
    
    labelsarray = np.asarray(labels)
    #print(labelsarray)
    #print(labels)
    
    EachFileWords = list(EachFileWordsDict.values())
    
    EachFileWordsArray = np.asarray(EachFileWords)
    


    
    #print(changedWeights)
    
    HamTestFiles = loaddatafromfiles(sys.argv[2]+'/ham')
    SpamTestFiles = loaddatafromfiles(sys.argv[2]+'/spam')
    
    TestFiles = HamTestFiles + SpamTestFiles
    
    EachFileWordsDictTest = dict()
    count1 = 0
    for file in TestFiles:
        WordslistTest=[]
        f = open(file,'r',errors ='ignore')     #Here getting encoding error for charpmap so wrote error ignore
        wordstest = f.read().replace('\n',' ')
        WordslistTest.extend(wordstest.split())
        WordslistTest = [''.join(c for c in s if c not in string.punctuation) for s in WordslistTest]   #Ref Stackoverflow
        WordslistTest = [x for x in WordslistTest if x]
        WordslistTest = [x for x in WordslistTest if x not in stopwords]  #Removing Stop words
        row1=[0]*len(UniqueCombinedWords)
        for i in range(len(UniqueCombinedWords)):        
            if UniqueCombinedWords[i] in WordslistTest:
                feature_count=WordslistTest.count(UniqueCombinedWords[i])
                row1[i]=feature_count                #Here we are labling as 1 if word is present or else as zero
            else:
                row1[i]=0
                
        row1.insert(0,1)                #add one more column infront to indicate the feature 
        EachFileWordsDictTest[count1]=row1
        count1+=1
    
    labels1 = []             #We have to set original label for the given files say Ham as '1' and Spam as '0'
    for i in range(len(HamTestFiles)):
        labels1.append(1)
    for i in range(len(SpamTestFiles)):
        labels1.append(0)
    
    EachFileWordsTest = list(EachFileWordsDictTest.values())
    
    EachFileWordsArrayTest = np.asarray(EachFileWordsTest)
    
    lamda = [0.001,0.01,2]
    
    iterations = 100
    
    for L in lamda:
        weights = np.zeros(EachFileWordsArray.shape[1])

        changedWeights = ChangeWeights(L,iterations,weights,EachFileWordsArray,labelsarray)      
   
        CorPredHam, CorPredSpam, InCorPredHam, InCorPredSpam = LogRegClassification(changedWeights,EachFileWordsArrayTest,labels1)
    
        AccuracyHam = CorPredHam/(CorPredHam + InCorPredHam)
    
        AccuracySpam = CorPredSpam/(CorPredSpam+InCorPredSpam)
    
        total_accuracy = (CorPredHam + CorPredSpam)/(CorPredHam + InCorPredHam + CorPredSpam+InCorPredSpam)
    
        print("TotalAccuracy :",total_accuracy*100)  #In terms of percentages
    
        print("HamAccuracy :",AccuracyHam*100)
    
        print("SpamAccuracy :",AccuracySpam*100)
        
        print("\n")
    
    
    