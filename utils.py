# these functions are useful utilities that are used for different training algorithms.

import numpy as np
import csv
import matplotlib.pyplot as plt
import matplotlib.font_manager
import matplotlib.cm as cm

def ReadFile(path, out):

    '''    

    Reads in data 

    Parameters:
    --------
    path: specifies which file to read in
    out: specifies whether or not to print each columns label
    
    Returns:
    --------
    data: A numpy matrix with string arguments. Each row is a different record. 
    Each column is a different feature. 

    '''

    data=[]
    with open(path, 'rb') as f:
        reader = csv.reader(f)
        for row in reader:
            data.append(row)

    if (out == 1):
        print 'Features in data set: ' + ', '.join(data[0])

    return np.array(data) 


def OutputFile(pred, header, idB, pathV):

    '''
    Outputs prediction to a file
    
    Parameters:
    ----------
    pathV: specifies the output file
    pred: an array of class predictions
    header: column labels
    idB: first passenger id

    Returns:
    ----------
    none

    '''

    paths = ['predRF.csv', 'predSVM.csv', 'predGB.csv']

    open_file_object = csv.writer(open(paths[pathV], "wb"))
    open_file_object.writerow(header)

    for i in xrange(0, len(pred)):
        open_file_object.writerow([int(idB) + i, int(pred[i])])


def mapToF(strSET, ch, dictN, dictC):

    '''
    Maps strings in data set to floats.

    Parameters:
    -----------
    strSET: numpy string array (either training set or test set)
    ch: offset that allows us to use the same code to map the test set as well (column numbers shifted)
    dictN: a dictionary of name titles that maps to ints. we define this for the training set and then use it on the training and test set
    dictC: a dictionary of cabins that map to ints. we define this for the training set and then use it on the training and test set
    
    Returns:
    --------
    dat: numpy float array
    dictN: a dictionary of name titles that maps to ints
    dictC: a dictionary of cabin letters that maps to ints

    '''
    
    # Using all features
    dat = np.zeros((np.shape(strSET)[0], 10), dtype = np.float64)
    
    # 1st feature is gender 
    mask = strSET[:,4 - ch] == 'male'
    dat[mask,0] = 1.0 # male = 1
    dat[~mask,0] = 2.0 # female = 2

    # 2nd feature is class
    dat[:,1] = np.float64(np.array([int(h) for h in strSET[:,2-ch]]))

    # 3rd is age. 
    mask = strSET[:,5 - ch] == ''
    strSET[mask,5 - ch] = '-1' # assigning -1 to all people without listed ages
    dat[:,2] = np.float64(strSET[:,5 - ch])

    # 4th is siblings / spouses
    dat[:,3] = np.float64(strSET[:,6 - ch])

    # 5th is parents / children
    dat[:,4] = np.float64(strSET[:,7 - ch])

    # 6th is fare
    mask = strSET[:,9 - ch] == ''
    strSET[mask,9 - ch] = '-1'
    dat[:,5] = np.float64(strSET[:,9 - ch])

    # 7th is embarkation
    dict2={}
    dict2['C'] = 1.0
    dict2['Q'] = 2.0
    dict2['S'] = 3.0
    dict2[''] = 0.0
    dat[:,6] = np.float64([dict2[h] for h in strSET[:,11 - ch]])

    # 8th is ticket number (whether or not ticket feature has letters or not)
    mask = np.array([len(h.split()) > 1 for h in strSET[:,8 - ch]])
    dat[mask,7] = 1.0

    # 9th is cabin information. first letter of cabin or zero
    if (ch == 0):
        # find unique cabins
        cabins = list(set([h[0] for h in strSET[:,10 - ch] if len(h) > 0]))
        # now make dict
        dictC = {}
        for i in xrange(0,len(cabins)):
            dictC[cabins[i]] = i + 1
        
    # now use dict to map cabins to float    
    for i in xrange(0, len(strSET)):
        if (len(strSET[i,10 - ch]) > 0):
            if (strSET[i,10 - ch][0] in dictC):
                dat[i,8] = np.float64(dictC[strSET[i,10 - ch][0]])
        # otherwise it is automatically zero

    # 10th is title of name
    names = [((h.split(',')[1]).split('.')[0]).strip() for h in strSET[:,3 - ch]]
    if (ch == 0):
        # find unique names
        namesU = list(set(names))
        # make dict out of unique titles
        dictN={}
        for i in xrange(0, len(namesU)):
            dictN[namesU[i]] = i + 1

    # now use dict to map names to float
    for i in xrange(0, len(strSET)):
        if (names[i] in dictN):
            dat[i,9] = np.float64(dictN[names[i]])
        # otherwise it is automatically zero
   
    return dat, dictN, dictC


def PrelimPlots(dat, lab):

    '''
    Generates 2d scatter plots of various features. Colors points based on class.

    Parameters:
    ----------
    dat: numpy matrix containing training set
    lab: numpy array containing class labels for each record in dat

    '''

    fig = plt.figure(1, figsize = (10,10))
    prop = matplotlib.font_manager.FontProperties(size = 10.5)
    msV = 3.0
    fs = 10

    # mask for labels
    mask0 = lab == 0 # all deceased data points                                                        

    # shift data so that all points aren't on top of each other
    shift = np.random.normal(scale = 0.09, size = len(lab))
    shift2 = np.random.normal(scale = 0.09, size = len(lab))

    # plotting gender vs pclass
    ax = fig.add_subplot(3, 3, 1)
    ax.plot(dat[mask0,0] + shift2[mask0], dat[mask0,1] + shift[mask0],
            'o', label = 'deceased', markersize = msV, c = 'b')
    ax.plot(dat[~mask0,0] + shift2[~mask0], dat[~mask0,1] + shift[~mask0],
            'o', label = 'survived', markersize = msV, c = 'g')

    # formatting
    ax.get_xaxis().set_ticks([1,2])
    ax.get_xaxis().set_ticklabels(['Male', 'Female'], rotation = 30, fontsize = fs)
    ax.get_yaxis().set_ticks([1,2,3])
    ax.get_yaxis().set_ticklabels(['1st class', '2nd class', '3rd class'], 
                                  rotation = 30, fontsize = fs)
    ax.axis([0.5, 2.5, 0.5, 3.5])
  
    # plotting gender vs age                                                       
    ax = fig.add_subplot(3, 3, 2)
    ax.plot(dat[mask0,0] + shift2[mask0], dat[mask0,2], 
            'o', label = 'deceased', markersize = msV, c = 'b')
    ax.plot(dat[~mask0,0] + shift2[~mask0], dat[~mask0,2], 
            'o', label = 'survived', markersize = msV, c = 'g')

    # formatting
    ax.get_xaxis().set_ticks([1,2])
    ax.get_xaxis().set_ticklabels(['Male', 'Female'], rotation = 30, fontsize = fs)
    ax.set_ylabel(r"Age", fontsize = fs)
    ax.axis([0.5, 2.5, -5, max(dat[:,2]) + 4])
    plt.legend(loc='upper right', prop = prop)

    # plotting gender vs siblings present
    ax = fig.add_subplot(3, 3, 3)
    ax.plot(dat[mask0,0] + shift2[mask0], dat[mask0,3] + shift[mask0], 
            'o', label = 'deceased', markersize = msV, c = 'b')
    ax.plot(dat[~mask0,0] + shift2[~mask0], dat[~mask0,3] + shift[~mask0], 
            'o', label = 'survived', markersize = msV, c = 'g')

    # formatting
    ax.get_xaxis().set_ticks([1,2])
    ax.get_xaxis().set_ticklabels(['Male', 'Female'], rotation = 30, fontsize = fs)
    ax.set_ylabel(r"Siblings / Spouses Present", fontsize = fs)
    ax.axis([0.5, 2.5, -2, max(dat[:,3]) + 2])

    # plotting gender vs parents present                                             
    ax = fig.add_subplot(3, 3, 4)
    ax.plot(dat[mask0,0] + shift2[mask0], dat[mask0,4] + shift[mask0],
            'o',label = 'deceased', markersize = msV, c = 'b')
    ax.plot(dat[~mask0,0] + shift2[~mask0], dat[~mask0,4] + shift[~mask0], 
            'o',label = 'survived', markersize = msV, c = 'g')

    # formatting                                                                                       
    ax.get_xaxis().set_ticks([1,2])
    ax.get_xaxis().set_ticklabels(['Male', 'Female'], rotation = 30, fontsize = fs)
    ax.set_ylabel(r"Parents / Children Present", fontsize = fs)
    ax.axis([0.5, 2.5, -2, max(dat[:,4]) + 2])

    # plotting gender vs fare present   
    ax = fig.add_subplot(3, 3, 5)
    ax.plot(dat[mask0,0] + shift2[mask0], dat[mask0,5], 
            'o', label = 'deceased', markersize = msV, c = 'b')
    ax.plot(dat[~mask0,0] + shift2[~mask0], dat[~mask0,5], 
            'o', label = 'survived', markersize = msV, c = 'g')

    # formatting
    ax.get_xaxis().set_ticks([1,2])
    ax.get_xaxis().set_ticklabels(['Male', 'Female'], rotation = 30, fontsize = fs)
    ax.set_ylabel(r"Fare",fontsize = fs)
    ax.axis([0.5, 2.5, -2, max(dat[:,5]) + 2])

    # plotting gender vs embarked                                                             
    ax = fig.add_subplot(3, 3, 6)
    ax.plot(dat[mask0,0] + shift2[mask0], dat[mask0,6] + shift[mask0], 
            'o',label = 'deceased', markersize = msV, c = 'b')
    ax.plot(dat[~mask0,0] + shift2[~mask0], dat[~mask0,6] + shift[~mask0], 
            'o',label = 'survived', markersize = msV, c = 'g')

    # formatting  
    ax.get_xaxis().set_ticks([1,2])
    ax.get_xaxis().set_ticklabels(['Male', 'Female'], rotation = 30, fontsize = fs)
    ax.get_yaxis().set_ticks([0,1,2,3])
    ax.get_yaxis().set_ticklabels(['NA', 'C','Q','S'], rotation = 0, fontsize = fs)
    ax.set_ylabel(r"Embarkation", fontsize = fs)
    ax.axis([0.5, 2.5, -1, 4])

    # plotting gender vs ticket number
    ax = fig.add_subplot(3, 3, 7)
    ax.plot(dat[mask0,0] + shift2[mask0], dat[mask0,7] + shift[mask0], 
            'o', label = 'deceased', markersize = msV, c = 'b')
    ax.plot(dat[~mask0,0] + shift2[~mask0], dat[~mask0,7] + shift[~mask0], 
            'o', label = 'survived', markersize = msV, c = 'g')

    # formatting
    ax.get_xaxis().set_ticks([1,2])
    ax.get_xaxis().set_ticklabels(['Male', 'Female'], rotation = 30, fontsize = fs)
    ax.get_yaxis().set_ticks([0,1])
    ax.get_yaxis().set_ticklabels(['No Char','Char'], rotation = 0, fontsize = fs)
    ax.set_ylabel(r"Ticket", fontsize = fs)
    ax.axis([0.5, 2.5, -1, 2])

    # plotting gender vs whether Cabin is Listed                                             
    ax = fig.add_subplot(3, 3, 8)
    ax.plot(dat[mask0,0] + shift2[mask0], dat[mask0,8] + shift[mask0], 
            'o', label = 'deceased', markersize = msV, c = 'b')
    ax.plot(dat[~mask0,0] + shift2[~mask0], dat[~mask0,8] + shift[~mask0], 
            'o', label = 'survived', markersize = msV, c = 'g')

    # formatting
    ax.get_xaxis().set_ticks([1,2])
    ax.get_xaxis().set_ticklabels(['Male', 'Female'], rotation = 30, fontsize = fs)
    #ax.get_yaxis().set_ticks([0,1])
    #ax.get_yaxis().set_ticklabels(['No Cabin','Cabin'], rotation = 0, fontsize = fs)
    ax.set_ylabel(r"Cabin Information", fontsize = fs)
    #ax.axis([0.5, 2.5, -1, 2])

    # plotting title vs age
    ax = fig.add_subplot(3, 3, 9)
    ax.plot(dat[mask0,9] + 0.5 * shift2[mask0], dat[mask0,2], 
            'o', label = 'deceased', markersize = msV, c = 'b')
    ax.plot(dat[~mask0,9] + 0.5 * shift2[~mask0], dat[~mask0,2], 
            'o', label = 'survived', markersize = msV, c = 'g')
    ax.set_ylabel(r"Age", fontsize = fs)
    ax.set_xlabel(r"Different Titles (Mr, Miss, ...)", fontsize = fs)

    # save figure
    plt.savefig('prelim.pdf', bbox_inches = 'tight')
    fig.clear()

def AgeModel(dat, dict, means, ch):

    '''
    This function replaces  missing ages with more accurate guesses based on other features of record 
    
    Parameters:
    ----------
    dat: numpy array storing all records of features
    dict: dictionary storing all titles of people
    means: means of age within a given group (initially all zero)
    ch: specifies whether we want to calculate means (ch==1)

    Returns:
    -------
    dat: numpy array with new ages in positions that used to be empty
    means: mean of age within a given group defined by title

    '''

    # first find all records without missing age
    mask = dat[:,2] != -1.0

    if (ch == 1):
        
        for i in xrange(0, len(dict)):
            mask2 = dat[mask][:,9] == np.float64(i + 1)
            means[i] = np.mean(dat[mask][mask2,2])

    # now replace missing age value (-1) with appropriate mean depending on title
    # not most efficient way

    for i in xrange(0, len(dat)):
        if (dat[i,2] == -1.0):
            dat[i,2] = means[ np.int32(dat[i,9]) ]

    return dat, means

def MeanNorm(dat):

    '''

    This function subtracts the mean and divides by the standard deviation of the feature
    Parameters:
    -----------
    dat: numpy array of records
    
    Returns:
    -------
    dat: numpy array of records

    '''
    
    mn = np.mean(dat, axis = 0)
    std = np.std(dat, axis = 0)
    dat = (dat - mn) / std

    return dat
