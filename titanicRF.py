# This script analyzes the titanic data set from kaggle.
# It uses a random forest to make a prediction based on the training set.


import numpy as np
import matplotlib.pyplot as plt
import matplotlib.font_manager
import matplotlib.cm as cm
from sklearn import cross_validation
from sklearn.ensemble import RandomForestClassifier
import utils

def TestRandForest(dat, lab):

    '''                                        
    This function finds the optimal parameters for the classifier                                        
    Parameters:                                                                                       
    ----------
    dat: numpy array with all records
    lab: numpy array with class labels of all records                                                        
    Returns:
    --------
    par: optimal parameters for the classifier 
    '''

    # RF parameters. Will choose one based on which does best on the validation set
    # n_estimators, max_features
    est = range(5, 26, 5)
    feat = range(2, 8, 1)
    par = [(e,f) for e in est for f in feat]

    # want to try different ensembles to get error bar on score
    num = 10
    seed = np.random.randint(1000000, size = num)
    valScore = np.zeros((num, len(par)))
    testScore = np.zeros((num, len(par)))

    for nv in xrange(0,num):

        print 'Ensemble:', nv + 1

        # split training data into train, validation, test (60, 20, 20)
        xTrain, xTmp, yTrain, yTmp = cross_validation.train_test_split(dat, lab, 
                                                                       test_size = 0.4, 
                                                                       random_state = seed[nv])
        xVal, xTest, yVal, yTest = cross_validation.train_test_split(xTmp, yTmp, 
                                                                     test_size = 0.5, 
                                                                     random_state = seed[nv])

        # now train RF for each parameter combination
        for i in xrange(0,len(par)):
        
            clf = RandomForestClassifier(n_estimators=par[i][0], 
                                         max_features = par[i][1], 
                                         min_samples_split = 1)
            clf = clf.fit(xTrain, yTrain)
            valScore[nv,i] = clf.score(xVal, yVal)
            testScore[nv,i] = clf.score(xTest, yTest)

    # Find optimal parameters
    tmp = np.argmax(np.mean(valScore, axis = 0))
    print
    print 'Optimal parameters (num_estimators, max_features):', par[tmp]
    print ('Mean | Std Score (Validation set):', np.mean(valScore, axis = 0)[tmp],
           '|', np.std(valScore, axis = 0)[tmp])
    print ('Mean | Std Score (Test set):', np.mean(testScore, axis = 0)[tmp],
           '|', np.std(testScore, axis = 0)[tmp])

    # Return optimal parameters
    return par[tmp]

def plotLearningCurve(dat, lab, optim):

     ''' 
    This function plots the learning curve for the classifier                                                                                                         
    Parameters:
    ----------- 
    dat: numpy array with all records
    lab: numpay array with class labels of all records
    optim: optimal parameters for classifier                                                           
 
    '''
     
     clf = RandomForestClassifier(n_estimators=optim[0], 
                                  max_features = optim[1], 
                                  min_samples_split = 1, 
                                  compute_importances = True)

     # split training data into train and test (already chose optimal parameters)
     xTrain, xTest, yTrain, yTest = cross_validation.train_test_split(dat, lab, 
                                                                      test_size = 0.3)

     # choose various sizes of training set to model on to generate learning curve
     szV = range(10,np.shape(xTrain)[0], int(np.shape(xTrain)[0]) / 10)
     szV.append(np.shape(xTrain)[0])

     LCvals=np.zeros((len(szV),3), dtype = np.float64) # store data points of learning curve
     for i in xrange(0, len(szV)):
         clf = clf.fit(xTrain[:szV[i],:], yTrain[:szV[i]])
         LCvals[i,0] = szV[i]
         LCvals[i,1] = clf.score(xTest, yTest)
         LCvals[i,2] = clf.score(xTrain[:szV[i],:], yTrain[:szV[i]])

    #print LCvals
         
    # generate figure
     fig = plt.figure(1, figsize = (10,10))
     prop = matplotlib.font_manager.FontProperties(size = 15.5)
     ax = fig.add_subplot(1, 1, 1)
     ax.plot(LCvals[:,0] / np.float64(np.shape(xTrain)[0]), 1.0 - LCvals[:,1], 
             label = 'Test Set')
     ax.plot(LCvals[:,0] / np.float64(np.shape(xTrain)[0]), 1.0 - LCvals[:,2], 
             label = 'Training Set')
     ax.set_ylabel(r"Error", fontsize = 20)
     ax.set_xlabel(r"% of Training Set Used", fontsize = 20)
     ax.axis([0.0, 1.0, -0.1, 0.5])
     plt.legend(loc='upper right', prop = prop)
     plt.savefig('LC_RF.pdf', bbox_inches = 'tight')
     fig.clear()

    # where is model failing?
     clf = clf.fit(xTrain, yTrain)
     mask = clf.predict(xTest) != yTest
    #print 'Age'
    #print xTest[mask,2]
    #print 'Gender'
    #print xTest[mask,0]
    #print 'Class'
    #print xTest[mask,1]
    #print '3'
    #print xTest[mask,3]
    #print '4'
    #print xTest[mask,4]
    #print '5'
    #print xTest[mask,5]
    #print '6'
    #print xTest[mask,6]
    #print '7'
    #print xTest[mask,7]
    #print '8'
    #print xTest[mask,8]
    #print mask.sum(), np.shape(xTest)

     print clf.feature_importances_
    
     predProb = clf.predict_proba(xTest)
     tmp = np.zeros((np.shape(predProb)[0], np.shape(predProb)[1] + 2))
     tmp[:,:-2] = predProb
     tmp[:,-2] = clf.predict(xTest)
     tmp[:,-1] = yTest
     mask = tmp[:,-2] != tmp[:,-1]
     print tmp[mask]
     print mask.sum(), len(xTest)
    
     print tmp[:50,:]


def main():

    # Read training set
    print 
    print 'Reading training set...'
    train = utils.ReadFile('train.csv', 1)
    print 'Finished reading...\n'

    # Preliminary Statistics
    print 'Preliminary Statistics:'
    print np.shape(train)[0] - 1, 'people.', np.shape(train)[1] - 2, 'features.'
    print (train[1:,1] == '1').sum(), 'survivors.', (train[1:,1] == '0').sum(), 'deceased.\n'
    
    #Testing
    id = 10
    mask = train[1:,id] == ''
    #print list(set(tmp))
    #print train[1:,id]
    #print mask.sum()
        
    # Map string features to floats
    print 'Mapping Features to Floats.\n'
    dictN = {} # modified in call (useful for name feature) 
    dictC = {} # modified in call (useful for cabin feature)
    dat, dictN, dictC = utils.mapToF(train[1:,:], 0, dictN, dictC)

    # Class labels
    lab = np.array([int(h) for h in train[1:,1]])

    # Generate better model for missing Age feature
    means = np.zeros(len(dictN), dtype = np.float64)
    dat, means = utils.AgeModel(dat, dictN, means, 1)

    # Preliminary Plots
    print 'Generating preliminary scatter plots of data.\n'
    utils.PrelimPlots(dat, lab)

    # ML algorithms
    print "Choosing best parameters for Random Forest algorithm:"
    optim = TestRandForest(dat, lab)

    # Plotting Learning Curve
    print "Plotting the learning curve\n"
    plotLearningCurve(dat, lab, optim)

    # Read in test set
    print "Reading in Test Set\n"
    test = utils.ReadFile('test.csv', 0)

    # Map to floats
    testF, dictN, dictC = utils.mapToF(test[1:,:], 1, dictN, dictC)

    # Make better prediction for missing Age Features
    testF, means = utils.AgeModel(testF, dictN, means, 0)

    # Make prediction
    print "Making Prediction\n"
    clf = RandomForestClassifier(n_estimators = optim[0], 
                                 max_features = optim[1], 
                                 min_samples_split = 1)
    clf = clf.fit(dat, lab)
    pred = clf.predict(testF)

    # Now output prediction
    print "Outputting Prediction\n"
    utils.OutputFile(pred, train[0,:2], test[1,0], 0)

    print "Done"

if __name__ == '__main__':
    main()
