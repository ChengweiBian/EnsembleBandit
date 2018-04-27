from conf import * 	# it saves the address of data stored and where to save the data produced by algorithms
import time
import re 			# regular expression library
from random import random, choice 	# for random strategy
from operator import itemgetter
import datetime
import numpy as np 	
from scipy.sparse import csgraph
from scipy.spatial import distance


# time conventions in file name
# dataDay + Month + Day + Hour + Minute

def vectorize(M):
	temp = []
	for i in range(M.shape[0]*M.shape[1]):
		temp.append(M.T.item(i))
	V = np.asarray(temp)
	return V

def matrixize(V, C_dimension):
	temp = np.zeros(shape = (C_dimension, len(V)/C_dimension))
	for i in range(len(V)/C_dimension):
		temp.T[i] = V[i*C_dimension : (i+1)*C_dimension]
	W = temp
	return W

#read centroids from file
def getClusters(fileNameWriteCluster):
    with open(fileNameWriteCluster, 'r') as f:
        clusters = []
        for line in f:
            vec = []
            line = line.split(' ')
            for i in range(len(line)-1):
                print line
            	vec.append(float(line[i]))
            clusters.append(np.asarray(vec))
    	return np.asarray(clusters)

# get cluster assignment of V, M is cluster centroids
def getIDAssignment(V, M):
	MinDis = float('+inf')
	assignment = None
	for i in range(M.shape[0]):
		dis = distance.euclidean(V, M[i])
		if dis < MinDis:
			assignment = i
			MinDis = dis
	return assignment


# generate graph W according to similarity
def initializeW(userFeatureVectors):
	n = len(userFeatureVectors)
	
	W = np.zeros(shape = (n, n))
		
	for i in range(n):
		sSim = 0
		for j in range(n):
			sim = np.dot(userFeatureVectors[i], userFeatureVectors[j])			
			W[i][j] = sim
			sSim += sim
			
		W[i] /= sSim
		for a in range(n):
			print '%.3f' % W[i][a],
		print ''
	
	#W = np.identity(n)
	#print W
	return W.T

# data structure to store ctr	
class articleAccess():
	def __init__(self):
		self.accesses = 0.0 # times the article was chosen to be presented as the best articles
		self.clicks = 0.0 	# of times the article was actually clicked by the user
		self.CTR = 0.0 		# ctr as calculated by the updateCTR function

	def updateCTR(self):
		try:
			self.CTR = self.clicks / self.accesses
		except ZeroDivisionError: # if it has not been accessed
			self.CTR = -1
		return self.CTR

	def addrecord(self, click):
		self.clicks += click
		self.accesses += 1

# structure to save data from random strategy as mentioned in LiHongs paper
class randomStruct:
	def __init__(self):
		self.learn_stats = articleAccess()

# structure to save data from CoLinUCB strategy
class CoLinUCBStruct:
	def __init__(self, lambda_, d, userNum, userFeatureVectors):
		self.learn_stats = articleAccess()	

		self.d = d
		self.userNum = userNum

		self.reward = np.zeros(userNum)
		self.W = initializeW(userFeatureVectors)

		self.A = lambda_* np.identity(n = d*userNum)
		self.b = np.zeros(d*userNum)
		self.AInv = np.linalg.inv(self.A)

		self.theta = matrixize(np.dot(self.AInv , self.b), self.d)
		self.CoTheta = np.dot(self.theta, self.W)

		self.CCA = np.identity(n = d*userNum)
		self.BigW = np.kron(np.transpose(self.W), np.identity(n=d))
			
	def updateParameters(self, PickedfeatureVector, reward, userID):
		X = vectorize(np.outer(PickedfeatureVector, self.W.T[userID])) 
		self.A += np.outer(X, X)	
		self.b += reward*X
		self.AInv = np.linalg.inv(self.A)

		self.theta = matrixize(np.dot(self.AInv , self.b), self.d)
		self.CoTheta = np.dot(self.theta, self.W)
		self.CCA = np.dot(np.dot(self.BigW , self.AInv), np.transpose(self.BigW) )

	def getCoLinUCBPta(self, alpha, featureVector, userID):
		featureVectorV = np.zeros(self.d*self.userNum)
		featureVectorV[int(userID*self.d):(int(userID)+1)*self.d] = np.asarray(featureVector)
                '''
		featureVectorM = np.zeros(shape = (self.d, self.userNum))
		featureVectorM.T[userID] = featureVector
		featureVectorV = vectorize(featureVectorM)
		#print len(self.theta), len(featureVector)
		'''     
		mean = np.dot(self.CoTheta.T[userID], featureVector)
		var = np.sqrt(np.dot(np.dot(featureVectorV, self.CCA) , featureVectorV))
		#Dur_time = time.time() - start_time
		#print 'time', 10000* Dur_time
		pta = mean + alpha*var
                
		return pta
		

# This code simply reads one line from the source files of Yahoo!
def parseLine(line):
	line = line.split("|")

	tim, articleID, click = line[0].strip().split(" ")
	tim, articleID, click = int(tim), int(articleID), int(click)
	user_features = np.array([float(x.strip().split(':')[1]) for x in line[1].strip().split(' ')[1:]])

	pool_articles = [l.strip().split(" ") for l in line[2:]]
	pool_articles = np.array([[int(l[0])] + [float(x.split(':')[1]) for x in l[1:]] for l in pool_articles])
	return tim, articleID, click, user_features, pool_articles

def save_to_file(fileNameWrite, recordedStats, tim):
	with open(fileNameWrite, 'a+') as f:
		f.write('data') # the observation line starts with data;
		f.write(',' + str(tim))
		f.write(',' + ';'.join([str(x) for x in recordedStats]))
		f.write('\n')
    

if __name__ == '__main__':
	# regularly print stuff to see if everything is going alright.
	# this function is inside main so that it shares variables with main and I dont wanna have large number of function arguments
	def printWrite():
		randomLearnCTR = articles_random.learn_stats.updateCTR()	
		CoLinUCBCTR = CoLinUCB_USERS.learn_stats.updateCTR()

		print totalObservations
		print 'random', randomLearnCTR,'  CoLin', CoLinUCBCTR
		
		recordedStats = [randomLearnCTR, CoLinUCBCTR]
		# write to file
		save_to_file(fileNameWrite, recordedStats, tim) 


	timeRun = datetime.datetime.now().strftime('_%m_%d_%H_%M') 	# the current data time
	# dataDays = ['01', '02', '03', '04', '05', '06', '07', '08', '09', '10']
	dataDays = ['10']
	fileSig = 'CoLin_20_'
	batchSize = 2000							# size of one batch
	
	d = 5 	        # feature dimension
	alpha = 0.3     # control how much to explore
	lambda_ = 0.2   # regularization used in matrix A
    
	totalObservations = 0

	fileNameWriteCluster = os.path.join(data_address, 'kmeans_model_20.dat')
	userFeatureVectors = getClusters(fileNameWriteCluster)	
	userNum = len(userFeatureVectors)
	
	articles_random = randomStruct()
	CoLinUCB_USERS = CoLinUCBStruct(lambda_ , d, userNum, userFeatureVectors)
	

	for dataDay in dataDays:
		fileName = "./data" + "/ydata-fp-td-clicks-v1_0.200905" + dataDay	
		fileNameWrite = os.path.join(save_address, fileSig + dataDay + timeRun + '.csv')

		# put some new data in file for readability
		with open(fileNameWrite, 'a+') as f:
			f.write('\nNew Run at  ' + datetime.datetime.now().strftime('%m/%d/%Y %H:%M:%S'))
			f.write('\n, Time,RandomCTR;CoLinUCBCTR\n')

		print fileName, fileNameWrite


		with open(fileName, 'r') as f:
			# reading file line ie observations running one at a time
			for line in f:
				totalObservations +=1

				tim, article_chosen, click, user_features, pool_articles = parseLine(line)
				currentUser_featureVector = user_features[:-1]

				currentUserID = getIDAssignment(np.asarray(currentUser_featureVector), userFeatureVectors)                
                
                                #-----------------------------Pick an article (CoLinUCB, LinUCB, Random)-------------------------
                                currentArticles = []
                                CoLinUCB_maxPTA = float('-inf')
                                CoLinUCBPicked = None      
                                #CoLinUCB_PickedfeatureVector = np.array([0,0,0,0,0])
                               
                                for article in pool_articles:
                                        article_id = article[0]
                                        article_featureVector =np.asarray(article[1:6])
                                        currentArticles.append(article_id)
                                        # CoLinUCB pick article
                                        if len(article_featureVector)==5:
                                                CoLinUCB_pta = CoLinUCB_USERS.getCoLinUCBPta(alpha, article_featureVector, currentUserID)
                                                if CoLinUCB_maxPTA < CoLinUCB_pta:
                                                        CoLinUCBPicked = article_id    # article picked by CoLinUCB
                                                        CoLinUCB_PickedfeatureVector = article_featureVector
                                                        CoLinUCB_maxPTA = CoLinUCB_pta
                                #print CoLinUCBPicked
                                               

                                # article picked by random strategy
                             
                                articles_random.learn_stats.addrecord(click)
                                if CoLinUCBPicked == article_chosen:
                                        CoLinUCB_USERS.learn_stats.addrecord(click)
                                        CoLinUCB_USERS.updateParameters(CoLinUCB_PickedfeatureVector, click, currentUserID)

                                # if the batch has ended
                                if totalObservations%batchSize==0:
                                        printWrite()
                        #print stuff to screen and save parameters to file when the Yahoo! dataset file ends
                        printWrite()
