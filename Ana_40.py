import numpy as np
import os
from conf import *
from matplotlib.pylab import *
from operator import itemgetter

if __name__ == '__main__':
    
    filenames = [x for x in os.listdir(result_address) if '.csv' in x]
    '''
    articlesSingle = {}
    articlesMultiple = {}
    articlesHours = {}
    '''
    CoLinCTR = {}
    GOBLinCTR = {}
    RandomCTR = {}
    LinUCBCTR = {}
    Temp1 = {}
    Temp2 = {}

    CoLinCTRRatio = {}
    LinUCBCTRRatio = {}
    GOBLinCTRRatio = {}
    '''
    ucba = {}
    ucbc = {}
    randa = {}
    randc = {}
    greedya = {}
    greedyc = {}
    ucbCTR = {}
    randCTR = {}
    greedyCTR = {}
    ucbCTRRatio = {}
    greedyCTRRatio = {}
    exp3CTRRatio={}
    '''
    tim = {}
    GOBtim = {}
    #i = -1 
    for x in filenames:    
        filename = os.path.join(result_address, x)
        if 'CoLin_40_01' in x:
            i = -1 
            with open(filename, 'r') as f:
                print str(filename)      
                for line in f:
                   
                    i = i + 1
                    words = line.split(',')
                    if words[0].strip() != 'data':
                        continue
                    RandomCTR[i], CoLinCTR[i], LinUCBCTR[i]= [float(x) for x in words[2].split(';')]
                    CoLinCTRRatio[i] = CoLinCTR[i]/RandomCTR[i]
                    LinUCBCTRRatio[i] = LinUCBCTR[i]/RandomCTR[i]
                    
                    #tim[i] = int(words[1])
                    tim[i] = i
        if 'CoLin_40_02' in x:
            with open(filename, 'r') as f:
                print str(filename)      
                for line in f:
                   
                    i = i + 1
                    words = line.split(',')
                    if words[0].strip() != 'data':
                        continue
                    RandomCTR[i], CoLinCTR[i], LinUCBCTR[i]= [float(x) for x in words[2].split(';')]
                    CoLinCTRRatio[i] = CoLinCTR[i]/RandomCTR[i]
                    LinUCBCTRRatio[i] = LinUCBCTR[i]/RandomCTR[i]


                    
                    #tim[i] = int(words[1])
                    tim[i] = i
                    
        if 'GOB_40_01' in x:
            i = -1
            with open(filename, 'r') as f:
                print str(filename)
            
                for line in f:
                   
                    i = i + 1
                    words = line.split(',')
                    if words[0].strip() != 'data':
                        continue
                    Temp1[i], GOBLinCTR[i]= [float(x) for x in words[2].split(';')]
                    GOBLinCTRRatio[i] = GOBLinCTR[i]/Temp1[i]
                    
                    #tim[i] = int(words[1])
                    GOBtim[i] = i
        if 'GOB_40_02' in x:
            with open(filename, 'r') as f:
                print str(filename)
            
                for line in f:
                   
                    i = i + 1
                    words = line.split(',')
                    if words[0].strip() != 'data':
                        continue
                    Temp1[i], GOBLinCTR[i]= [float(x) for x in words[2].split(';')]
                    GOBLinCTRRatio[i] = GOBLinCTR[i]/Temp1[i]
                    
                    #tim[i] = int(words[1])
                    GOBtim[i] = i
                    
    #print 'len(tim)', len(tim), 'len', len(ucbCTRRatio)
    #print ucbCTRRatio 
    print len(tim), len(GOBLinCTR)
    
    plt.plot(tim.values(), CoLinCTRRatio.values(), label = 'CoLin')
    plt.plot(GOBtim.values(), GOBLinCTRRatio.values(),  label = 'GOB.Lin')
    plt.plot(tim.values(), LinUCBCTRRatio.values(), label = 'LinUCB')
    plt.xlabel('time')
    plt.ylabel('CTR-Ratio')
    plt.legend(loc = 'lower right')
    plt.axis([0, 4500, 0, 2])
    plt.title('UserNum = 40')
    plt.show()
    
    '''      
    plt.plot(tim.values(), ucbCTRRatio.values(), label = 'Restart_ucbCTR Ratio')
    plt.plot(tim.values(), greedyCTRRatio.values(),  label = 'greedyCTR Ratio')
    plt.legend()
    plt.show()
    '''

   