# This script analyzes the titanic data set from kaggle.

# questions to address
# how much do relationships matter? (does whole family survive?)
# does ticket number play a role
# embarkation?
# can model age potentially

# interesting to ask what features have most distinguishing power
# retrain on everything or use trained model with part of the data?
# python passing by reference?
# better model for age
# use name information
# use more cabin information
# why does learning curve look like that?
# include data sets on github?

# investigate where algorithm is going wrong
# more plots
# different forests for different genders?
# confused by parameters of model

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.font_manager
import matplotlib.cm as cm
import csv
from sklearn import cross_validation
from sklearn.ensemble import RandomForestClassifier

def ReadFile(path,out):
# out specifies whether or not to output column labels
    
    data=[]
    with open(path, 'rb') as f:
        reader = csv.reader(f)
        for row in reader:
            data.append(row)

    if (out==1):
        print 'Features in data set: ' + ', '.join(data[0])
    return np.array(data[1:]) # neglecting first labeled row
    
def PrelimPlots(dat, lab):

    fig = plt.figure(1, figsize=(10,10))
    prop = matplotlib.font_manager.FontProperties(size=10.5)
    msV=3.0
    fs=10

    # mask for labels
    mask0 = lab==0 # all deceased data points
    
    # shift data so that all points aren't on top of each other
    shift = np.random.normal(scale=0.09, size=len(lab))
    shift2 = np.random.normal(scale=0.09, size=len(lab))
    
    # plotting gender vs pclass
    ax = fig.add_subplot(3, 3, 1)
    ax.plot(dat[mask0,0]+shift2[mask0],dat[mask0,1]+shift[mask0],'o',label='deceased',markersize=msV,c='b')
    ax.plot(dat[~mask0,0]+shift2[~mask0],dat[~mask0,1]+shift[~mask0],'o',label='survived',markersize=msV,c='g')
    
    # formatting
    ax.get_xaxis().set_ticks([1,2])
    ax.get_xaxis().set_ticklabels(['Male', 'Female'],rotation=30,fontsize=fs)
    ax.get_yaxis().set_ticks([1,2,3])
    ax.get_yaxis().set_ticklabels(['1st class', '2nd class', '3rd class'],rotation=30,fontsize=fs)
    ax.axis([0.5, 2.5, 0.5, 3.5])
    #plt.legend(loc='upper right',prop=prop)

    # plotting gender vs age
    ax = fig.add_subplot(3, 3, 2)
    ax.plot(dat[mask0,0]+shift2[mask0],dat[mask0,2],'o',label='deceased',markersize=msV,c='b')
    ax.plot(dat[~mask0,0]+shift2[~mask0],dat[~mask0,2],'o',label='survived',markersize=msV,c='g')
    
    # formatting 
    ax.get_xaxis().set_ticks([1,2])
    ax.get_xaxis().set_ticklabels(['Male', 'Female'],rotation=30,fontsize=fs)
    ax.set_ylabel(r"Age",fontsize=fs)
    ax.axis([0.5, 2.5, -5, max(dat[:,2])+4])
    plt.legend(loc='upper right',prop=prop)

    # plotting gender vs siblings present 
    ax = fig.add_subplot(3, 3, 3)
    ax.plot(dat[mask0,0]+shift2[mask0],dat[mask0,3]+shift[mask0],'o',label='deceased',markersize=msV,c='b')
    ax.plot(dat[~mask0,0]+shift2[~mask0],dat[~mask0,3]+shift[~mask0],'o',label='survived',markersize=msV,c='g')

    # formatting                        
    ax.get_xaxis().set_ticks([1,2])
    ax.get_xaxis().set_ticklabels(['Male', 'Female'],rotation=30,fontsize=fs)
    ax.set_ylabel(r"Siblings / Spouses Present",fontsize=fs)
    ax.axis([0.5, 2.5, -2, max(dat[:,3])+2])

    # plotting gender vs parents present
    ax = fig.add_subplot(3, 3, 4)
    ax.plot(dat[mask0,0]+shift2[mask0],dat[mask0,4]+shift[mask0],'o',label='deceased',markersize=msV,c='b')
    ax.plot(dat[~mask0,0]+shift2[~mask0],dat[~mask0,4]+shift[~mask0],'o',label='survived',markersize=msV,c='g')

    # formatting                                                               
    ax.get_xaxis().set_ticks([1,2])
    ax.get_xaxis().set_ticklabels(['Male', 'Female'],rotation=30,fontsize=fs)
    ax.set_ylabel(r"Parents / Children Present",fontsize=fs)
    ax.axis([0.5, 2.5, -2, max(dat[:,4])+2])

    # plotting gender vs fare present                           
    ax = fig.add_subplot(3, 3, 5)
    ax.plot(dat[mask0,0]+shift2[mask0],dat[mask0,5],'o',label='deceased',markersize=msV,c='b')
    ax.plot(dat[~mask0,0]+shift2[~mask0],dat[~mask0,5],'o',label='survived',markersize=msV,c='g')

    # formatting                                                               
    ax.get_xaxis().set_ticks([1,2])
    ax.get_xaxis().set_ticklabels(['Male', 'Female'],rotation=30,fontsize=fs)
    ax.set_ylabel(r"Fare",fontsize=fs)
    ax.axis([0.5, 2.5, -2, max(dat[:,5])+2])

    # plotting gender vs embarked
    ax = fig.add_subplot(3, 3, 6)
    ax.plot(dat[mask0,0]+shift2[mask0],dat[mask0,6]+shift[mask0],'o',label='deceased',markersize=msV,c='b')
    ax.plot(dat[~mask0,0]+shift2[~mask0],dat[~mask0,6]+shift[~mask0],'o',label='survived',markersize=msV,c='g')

    # formatting                                                               
    ax.get_xaxis().set_ticks([1,2])
    ax.get_xaxis().set_ticklabels(['Male', 'Female'],rotation=30,fontsize=fs)
    ax.get_yaxis().set_ticks([0,1,2,3])
    ax.get_yaxis().set_ticklabels(['NA', 'C','Q','S'],rotation=0,fontsize=fs)
    ax.set_ylabel(r"Embarkation",fontsize=fs)
    ax.axis([0.5, 2.5, -1, 4])

    # plotting gender vs ticket number
    ax = fig.add_subplot(3, 3, 7)
    ax.plot(dat[mask0,0]+shift2[mask0],dat[mask0,7]+shift[mask0],'o',label='deceased',markersize=msV,c='b')
    ax.plot(dat[~mask0,0]+shift2[~mask0],dat[~mask0,7]+shift[~mask0],'o',label='survived',markersize=msV,c='g')

    # formatting                                  
    ax.get_xaxis().set_ticks([1,2])
    ax.get_xaxis().set_ticklabels(['Male', 'Female'],rotation=30,fontsize=fs)
    ax.get_yaxis().set_ticks([0,1])
    ax.get_yaxis().set_ticklabels(['No Char','Char'],rotation=0,fontsize=fs)
    ax.set_ylabel(r"Ticket",fontsize=fs)
    ax.axis([0.5, 2.5, -1, 2])

    # plotting gender vs whether Cabin is Listed
    ax = fig.add_subplot(3, 3, 8)
    ax.plot(dat[mask0,0]+shift2[mask0],dat[mask0,8]+shift[mask0],'o',label='deceased',markersize=msV,c='b')
    ax.plot(dat[~mask0,0]+shift2[~mask0],dat[~mask0,8]+shift[~mask0],'o',label='survived',markersize=msV,c='g')

    # formatting              
    ax.get_xaxis().set_ticks([1,2])
    ax.get_xaxis().set_ticklabels(['Male', 'Female'],rotation=30,fontsize=fs)
    ax.get_yaxis().set_ticks([0,1])
    ax.get_yaxis().set_ticklabels(['No Cabin','Cabin'],rotation=0,fontsize=fs)
    ax.axis([0.5, 2.5, -1, 2])


    # save figure
    plt.savefig('prelim.pdf', bbox_inches='tight')
    fig.clear()

def mapToF(train,ch):
# ch specifies the shift necessary since the training set has the class labels in column 0 
# while the test set has the gender in column 0

    # Using 9 features (not name)
    dat = np.zeros((np.shape(train)[0],9),dtype=np.float64)
    
    # 1st feature is gender 
    mask = train[:,3-ch]=='male'
    dat[mask,0]=1.0 # male = 1
    dat[~mask,0]=2.0 # female = 2

    # 2nd feature is class
    dat[:,1] = np.float64(np.array([int(h) for h in train[:,1-ch]]))

    # 3rd is age. 
    mask = train[:,4-ch]==''
    train[mask,4-ch]='-1' # assigning -1 to all people without listed ages
    dat[:,2] = np.float64(train[:,4-ch])

    # 4th is siblings / spouses
    dat[:,3] = np.float64(train[:,5-ch])

    # 5th is parents / children
    dat[:,4] = np.float64(train[:,6-ch])

    # 6th is fare
    mask = train[:,8-ch]==''
    train[mask,8-ch]='-1'
    dat[:,5] = np.float64(train[:,8-ch])

    # 7th is embarkation
    dict={}
    dict['C']=1.0
    dict['Q']=2.0
    dict['S']=3.0
    dict['']=0.0
    dat[:,6] = np.float64([dict[h] for h in train[:,10-ch]])

    # 8th is ticket number (whether or not ticket feature has letters or not)
    mask = np.array([len(h.split())>1 for h in train[:,7]])
    dat[mask,7]=1.0

    # 9th is cabin information. whether or not there is a cabin listed
    mask = np.array([len(h)>1 for h in train[:,9]])
    dat[mask,8]=1.0
   
    return dat

def TestRandForest(dat,lab):

    # RF parameters. Will choose one based on which does best on the validation set
    # n_estimators, max_features
    est=range(5,26,5)
    feat=range(2,8,1)
    par = [(e,f) for e in est for f in feat]

    # want to try different ensembles to get error bar on score
    num=10
    seed = np.random.randint(1000000, size=num)
    val_score = np.zeros((num,len(par)))
    test_score = np.zeros((num,len(par)))

    for nv in xrange(0,num):

        print 'Ensemble:', nv+1

        # split training data into train, validation, test (60, 20, 20)
        X_train, X_tmp, y_train, y_tmp = cross_validation.train_test_split(dat, lab, test_size=0.4, random_state=seed[nv])
        X_val, X_test, y_val, y_test = cross_validation.train_test_split(X_tmp, y_tmp, test_size=0.5, random_state=seed[nv])

        # now train RF for each parameter combination
        for i in xrange(0,len(par)):
        
            clf = RandomForestClassifier(n_estimators=par[i][0],max_features=par[i][1],min_samples_split=1)
            clf = clf.fit(X_train, y_train)
            val_score[nv,i] = clf.score(X_val, y_val)
            test_score[nv,i] = clf.score(X_test,y_test)

    # Find optimal parameters
    tmp = np.argmax(np.mean(val_score,axis=0))
    print
    print 'Optimal parameters (num_estimators, max_features):', par[tmp]
    print 'Mean | Std Score (Validation set):', np.mean(val_score,axis=0)[tmp], '|', np.std(val_score,axis=0)[tmp]
    print 'Mean | Std Score (Test set):', np.mean(test_score,axis=0)[tmp], '|', np.std(test_score,axis=0)[tmp]

    # Return optimal parameters
    return par[tmp]

def plotLearningCurve(dat,lab,optim):

    clf = RandomForestClassifier(n_estimators=optim[0],max_features=optim[1],min_samples_split=1,compute_importances=True)
    #clf = RandomForestClassifier()

    # split training data into train and test (already chose optimal parameters)
    X_train, X_test, y_train, y_test = cross_validation.train_test_split(dat, lab, test_size=0.3)

    # choose various sizes of training set to model on to generate learning curve
    szV = range(10,np.shape(X_train)[0],int(np.shape(X_train)[0])/10)
    szV.append(np.shape(X_train)[0])

    LCvals=np.zeros((len(szV),3),dtype=np.float64) # store data points of learning curve
    for i in xrange(0,len(szV)):
        clf = clf.fit(X_train[:szV[i],:], y_train[:szV[i]])
        LCvals[i,0]=szV[i]
        LCvals[i,1]=clf.score(X_test,y_test)
        LCvals[i,2]=clf.score(X_train[:szV[i],:], y_train[:szV[i]])

    #print LCvals

    # generate figure
    fig = plt.figure(1, figsize=(10,10))
    prop = matplotlib.font_manager.FontProperties(size=15.5)
    ax = fig.add_subplot(1, 1, 1)
    ax.plot(LCvals[:,0]/np.float64(np.shape(X_train)[0]),1.0-LCvals[:,1],label='Test Set')
    ax.plot(LCvals[:,0]/np.float64(np.shape(X_train)[0]),1.0-LCvals[:,2],label='Training Set')
    ax.set_ylabel(r"Error",fontsize=20)
    ax.set_xlabel(r"% of Training Set Used",fontsize=20)
    ax.axis([0.0, 1.0, -0.1, 0.5])
    plt.legend(loc='upper right',prop=prop)
    plt.savefig('LC.pdf', bbox_inches='tight')
    fig.clear()

    # where is model failing?
    clf = clf.fit(X_train, y_train)
    mask = clf.predict(X_test)!=y_test
    #print 'Age'
    #print X_test[mask,2]
    #print 'Gender'
    #print X_test[mask,0]
    #print 'Class'
    #print X_test[mask,1]
    #print '3'
    #print X_test[mask,3]
    #print '4'
    #print X_test[mask,4]
    #print '5'
    #print X_test[mask,5]
    #print '6'
    #print X_test[mask,6]
    #print '7'
    #print X_test[mask,7]
    #print '8'
    #print X_test[mask,8]
    #print mask.sum(), np.shape(X_test)

    #print clf.feature_importances_

def AgeModel(dat,lab):

    mask = dat[:,2]!=-1.0
    hist, bin_edges = np.histogram(dat[mask,2])
    print hist, bin_edges
    print hist.sum()

    return dat

def main():

    # Read training set
    print 
    print 'Reading training set...'
    train=ReadFile('train.csv',1)
    print 'Finished reading...\n'

    # Preliminary Statistics
    print 'Preliminary Statistics:'
    print np.shape(train)[0], 'people.', np.shape(train)[1]-1, 'features.'
    print (train[:,0] == '1').sum(), 'survivors.', (train[:,0] =='0').sum(), 'deceased.\n'
    
    #Testing
    #id=4
    #mask = train[:,id]==''
    #print train[:,id]
    #print mask.sum()

    # Map string features to floats
    print 'Mapping Features to Floats.\n'
    dat=mapToF(train,0)

    # Class labels
    lab = np.array([int(h) for h in train[:,0]])

    # Generate better model for missing Age feature (To do)
    #dat = AgeModel(dat,lab)

    # Preliminary Plots
    print 'Generating preliminary scatter plots of data.\n'
    PrelimPlots(dat,lab)

    # ML algorithms
    print "Choosing best parameters for Random Forest algorithm:"
    optim = TestRandForest(dat,lab)

    # Plotting Learning Curve
    print "Plotting the learning curve\n"
    plotLearningCurve(dat,lab,optim)

    # Read in test set
    print "Reading in Test Set\n"
    test=ReadFile('test.csv',0)

    # Map to floats
    testF = mapToF(test,1)

    # Make prediction
    print "Making Prediction\n"
    clf = RandomForestClassifier(n_estimators=optim[0],max_features=optim[1],min_samples_split=1)
    clf = clf.fit(dat, lab)
    pred = clf.predict(testF)

    # Now output prediction
    print "Outputting Prediction\n"
    open_file_object = csv.writer(open("predRF.csv", "wb"))
    for p in pred:
        open_file_object.writerow(str(p))

    #print pred
    print "Done"

if __name__ == '__main__':
    main()
