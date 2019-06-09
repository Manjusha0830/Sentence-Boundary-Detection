from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
import pandas as p
from collections import defaultdict
import sys


def namely(trainRead,testRead):

    #Reading data from SBD.train file
    
    wordsList1 = []
    wordsList1 = trainRead.readlines() 
    trainRead.close()

    #Reading data from SBD.test file
    
    wordsList2 = []
    wordsList2 = testRead.readlines() 
    testRead.close()
    
    #columns = (['Left','Right','LeftBelowThree','LeftCap','RightCap','Period_in_L','Period_in_R','Numeric_L','Boundary'])
    trainDataList = defaultdict(list)
    
    for i in range(0, len(wordsList1)):
        
        if '. '  in wordsList1[i]:
            Lwords =[] 
            L =''
            R = 'EOP'  
            #Feature 1: L contains the left word  to the "."        
            Lwords = wordsList1[i].split('. ')    
            L = Lwords[0].split(' ')[1]
            trainDataList['Left'].append(L)

            #Feature 1: L contains the right word  to the "." 
            if(i<len(wordsList1)-1):
                R = wordsList1[i+1].split(' ')[1]
            trainDataList['Right'].append(R)

            #class cloumn result if 'NEOS' or 'EOS'
            if 'NEOS'  in Lwords[1]:               
                trainDataList['Boundary'].append('YES')               
            elif 'EOS' in Lwords[1]:
                trainDataList['Boundary'].append('NO')
            else:
                trainDataList['Boundary'].append('NA') #this is invalid path

            #feature 3: if length of left word > 3    
            if(len(L)<3):
                trainDataList['LeftBelowThree'].append('YES')
            else:
                trainDataList['LeftBelowThree'].append('NO')

            #feature 4: check if 1st character of left word is capital    
            if(L.isnumeric()):
                trainDataList['LeftCap'].append('NA')
            elif(L.istitle()):
                trainDataList['LeftCap'].append('YES')
            else:
                trainDataList['LeftCap'].append('NO')

            #feature 5: check if 1st character of right word is capital
            if(R.isnumeric()):
                trainDataList['RightCap'].append('NA')
            elif(R.istitle()):
                trainDataList['RightCap'].append('YES')
            else:
                trainDataList['RightCap'].append('NO')
                
            
            # feature 6: checks if Left word contains any "."
            if('.' in L):
                trainDataList['Period_in_L'].append('YES') 
            else:
                trainDataList['Period_in_L'].append('NO')
            
            # feature 6: checks if Right word contains any "."
            if('.' in R):
                trainDataList['Period_in_R'].append('YES') 
            else:
                trainDataList['Period_in_R'].append('NO')
            
            # feature 6: checks if Left word is numeric
            if(L.isnumeric()):
                trainDataList['Numeric_L'].append('YES')
            else:
                trainDataList['Numeric_L'].append('NO')

    testDataList = defaultdict(list)
    
    for i in range(0, len(wordsList2)):
        
        if '. '  in wordsList2[i]:
            Lwords =[] 
            L =''
            R = 'EOP'  
            #Feature 1: L contains the left word  to the "."        
            Lwords = wordsList2[i].split('. ')    
            L = Lwords[0].split(' ')[1]
            testDataList['Left'].append(L)

            #Feature 1: L contains the right word  to the "." 
            if(i<len(wordsList2)-1):
                R = wordsList1[i+1].split(' ')[1]
            testDataList['Right'].append(R)

            #class cloumn result if 'NEOS' or 'EOS'
            if 'NEOS'  in Lwords[1]:               
                testDataList['Boundary'].append('YES')               
            elif 'EOS' in Lwords[1]:
                testDataList['Boundary'].append('NO')
            else:
                testDataList['Boundary'].append('NA') #this is invalid path

            #feature 3: if length of left word > 3    
            if(len(L)<3):
                testDataList['LeftBelowThree'].append('YES')
            else:
                testDataList['LeftBelowThree'].append('NO')

            #feature 4: check if 1st character of left word is capital    
            if(L.isnumeric()):
                testDataList['LeftCap'].append('NA')
            elif(L.istitle()):
                testDataList['LeftCap'].append('YES')
            else:
                testDataList['LeftCap'].append('NO')

            #feature 5: check if 1st character of right word is capital
            if(R.isnumeric()):
                testDataList['RightCap'].append('NA')
            elif(R.istitle()):
                testDataList['RightCap'].append('YES')
            else:
                testDataList['RightCap'].append('NO')
                
            
            # feature 6: checks if Left word contains any "."
            if('.' in L):
                testDataList['Period_in_L'].append('YES') 
            else:
                testDataList['Period_in_L'].append('NO')
            
            # feature 7: checks if Right word contains any "."
            if('.' in R):
                testDataList['Period_in_R'].append('YES') 
            else:
                testDataList['Period_in_R'].append('NO')
            
            # feature 8: checks if Left word is numeric
            if(L.isnumeric()):
                testDataList['Numeric_L'].append('YES')
            else:
                testDataList['Numeric_L'].append('NO')   

    
    trainData =p.DataFrame.from_dict(trainDataList)
    testData =p.DataFrame.from_dict(testDataList)
   
    
    #Accuracy calculation for core features
    trainData['LeftBelowThree'] = trainData.index
    testData['LeftBelowThree'] = testData.index
    trainData['Left'] = trainData.index
    testData['Left'] = testData.index
    trainData['Right'] = trainData.index
    testData['Right'] = testData.index
    trainData['Boundary'] = trainData['Boundary'].map({'YES': 1, 'NO': 0, 'NA':-1})
    testData['Boundary'] = testData['Boundary'].map({'YES': 1, 'NO': 0, 'NA':-1})
    
    trainData['LeftCap'] = trainData['LeftCap'].map({'YES': 1, 'NO': 0, 'NA':-1})
    testData['LeftCap'] = testData['LeftCap'].map({'YES': 1, 'NO': 0, 'NA':-1})
    trainData['RightCap'] = trainData['RightCap'].map({'YES': 1, 'NO': 0, 'NA':-1})
    testData['RightCap'] = testData['RightCap'].map({'YES': 1, 'NO': 0, 'NA':-1})

    trainData['Period_in_L'] = trainData['Period_in_L'].map({'YES': 1, 'NO': 0})
    testData['Period_in_L'] = testData['Period_in_L'].map({'YES': 1, 'NO': 0})
    trainData['Period_in_R'] = trainData['Period_in_R'].map({'YES': 1, 'NO': 0})
    testData['Period_in_R'] = testData['Period_in_R'].map({'YES': 1, 'NO': 0})
    trainData['Numeric_L'] = trainData['Numeric_L'].map({'YES': 1, 'NO': 0})
    testData['Numeric_L'] = testData['Numeric_L'].map({'YES': 1, 'NO': 0})

    core_features = ['Boundary','Left','Right','LeftBelowThree','LeftCap','RightCap']
    X = trainData[core_features] 
    y = trainData['Boundary']
    test_data_X = testData[core_features]
    test_data_y = testData['Boundary']
    X_train,  X_test, y_train, y_test = train_test_split(X, y,test_size=0.2 )

    # Create Decision Tree classifer object
    clf = DecisionTreeClassifier()

# Build Decision Tree Classifer from train set X and Y
    
    clf = clf.fit(X_train,y_train)
#predict class for X 
   
  
    test_pred=clf.predict(test_data_X)
    testaccuracy=accuracy_score(test_data_y,test_pred)
    print("Accuracy for core features:",testaccuracy*100)

    # All 8 features
    all_features = ['Boundary','Left','Right','LeftBelowThree','LeftCap','RightCap','Period_in_L','Period_in_R','Numeric_L']
    X_all = trainData[all_features] 
    
    y_all = trainData['Boundary']
    test_data_X_all = testData[all_features]

    test_data_y_all = testData['Boundary']
    X_train_all,  X_test_all, y_train_all, y_test_all = train_test_split(X_all, y_all,test_size=0.2 )

     # Create Decision Tree classifer object
    clf_all = DecisionTreeClassifier()

# Build Decision Tree Classifer from train set X and Y
    
    clf_all = clf_all.fit(X_train_all,y_train_all)
#predict class for X 
   
    
    test_pred_all=clf_all.predict(test_data_X_all)
    testaccuracy_all=accuracy_score(test_data_y_all,test_pred_all)
    print("Accuracy for ALL features:",testaccuracy_all*100)

    # Added 3 features
    added_features = ['Period_in_L','Period_in_R','Numeric_L']
    X_added = trainData[added_features] 
    
    y_added = trainData['Boundary']
    test_data_X_added = testData[added_features]

    test_data_y_added = testData['Boundary']
    X_train_added,  X_test_added, y_train_added, y_test_added = train_test_split(X_added, y_added,test_size=0.2 )

     # Create Decision Tree classifer object
    clf_added = DecisionTreeClassifier()

# Build Decision Tree Classifer from train set X and Y
    
    clf_added = clf_added.fit(X_train_added,y_train_added)
#predict class for X 
   
    
    test_pred_added=clf_added.predict(test_data_X_added)
    testaccuracy_added=accuracy_score(test_data_y_added,test_pred_added)
    print("Accuracy for Added features:",testaccuracy_added*100)

    
def main():
    
    input1 = sys.argv[1]
    input2 = sys.argv[2]
    trainfile = open(input1)
    testfile = open(input2)
    namely(trainfile,testfile)
    
    
        
if __name__ == "__main__":
    main()      
