'''
Models: random, CoLinUCB, LinUCB, GOBLin, Ensemble
'''
import numpy as np
from random import randint
from scipy.sparse import csgraph 
from scipy.linalg import sqrtm
from scipy.stats import beta


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

# data structure to store ctr    
class articleAccess():
    def __init__(self):
        self.accesses = 0.0 # times the article was chosen to be presented as the best articles
        self.clicks = 0.0     # of times the article was actually clicked by the user

    def getCTR(self):
        try:
            CTR = self.clicks / self.accesses
        except ZeroDivisionError: # if it has not been accessed
            CTR = -1
        return CTR

    def addRecord(self, click):
        self.clicks += click
        self.accesses += 1

# structure to save data from random strategy as mentioned in LiHongs paper
class randomStruct:
    def __init__(self):
        self.name = 'random'
        self.learn_stats = articleAccess()
        
    def pick(self, pool_articles, userID, article_chosen, click):
    	random_index = randint(0, len(pool_articles)-1)
    	article_picked = pool_articles[random_index][0]

    	if article_picked == article_chosen:
            self.learn_stats.addRecord(click)


# structure to save data from CoLinUCB strategy
class CoLinUCBStruct:
    def __init__(self, alpha, lambda_, d, userNum, userFeatureVectors):
        self.name = 'CoLinUCB'
        self.learn_stats = articleAccess()    

        self.alpha = alpha
        self.d = d
        self.userNum = userNum

        self.W = self.initializeW(userFeatureVectors)

        self.A = lambda_* np.identity(n = d*userNum)
        self.b = np.zeros(d*userNum)
        self.AInv = np.linalg.inv(self.A)

        self.theta = matrixize(np.dot(self.AInv , self.b), self.d)
        self.CoTheta = np.dot(self.theta, self.W)

        self.CCA = np.identity(n = d*userNum)
        self.BigW = np.kron(np.transpose(self.W), np.identity(n=d))

    # generate graph W according to similarity
    def initializeW(self, userFeatureVectors):
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

        return W.T
            
    def updateParameters(self, PickedfeatureVector, click, userID):
        X = vectorize(np.outer(PickedfeatureVector, self.W.T[userID])) 
        self.A += np.outer(X, X)    
        self.b += click*X
        self.AInv = np.linalg.inv(self.A)

        self.theta = matrixize(np.dot(self.AInv , self.b), self.d)
        self.CoTheta = np.dot(self.theta, self.W)
        self.CCA = np.dot(np.dot(self.BigW , self.AInv), np.transpose(self.BigW) )

    def getPta(self, featureVector, userID):
        featureVectorV = np.zeros(self.d*self.userNum)
        featureVectorV[int(userID*self.d):(int(userID)+1)*self.d] = np.asarray(featureVector)
        
        mean = np.dot(self.CoTheta.T[userID], featureVector)
        var = np.sqrt(np.dot(np.dot(featureVectorV, self.CCA) , featureVectorV))
    
        pta = mean + self.alpha*var
                
        return pta

    def pick(self, pool_articles, userID, article_chosen, click):
        ids = [a[0] for a in pool_articles]
        features = [np.array(a[1:6]) for a in pool_articles]
        pta = [self.getPta(f, userID) for f in features]
        
        # pick the one with max pta
        index = np.argmax(pta)
        article_picked = ids[index]

        if article_picked == article_chosen:
            self.learn_stats.addRecord(click)
            self.updateParameters(features[index], click, userID)

        return article_picked      


# structure to save data from LinUCB strategy
class LinUCBStruct:
    def __init__(self, alpha, lambda_,  d, userNum):
        self.name = 'LinUCB'
        self.alpha = alpha
        self.d = d
        self.A, self.b, self.theta = [], [], []
        for i in range(userNum):
             self.A.append(lambda_*np.identity(n = d))
             self.b.append(np.zeros(d))
             self.theta.append(np.zeros(d))

        self.learn_stats = articleAccess()

    def updateParameters(self, PickedfeatureVector, click, userID):
        self.A[userID] += np.outer(PickedfeatureVector, PickedfeatureVector)
        self.b[userID] += click*PickedfeatureVector
        self.theta[userID] = np.dot(np.linalg.inv(self.A[userID]), self.b[userID])

    def getPta(self, featureVector, userID):
        mean = np.dot(self.theta[userID], featureVector)
        var = np.sqrt(np.dot( np.dot(featureVector, np.linalg.inv(self.A[userID])) , featureVector))
        pta = mean + self.alpha*var
        return pta

    def pick(self, pool_articles, userID, article_chosen, click):
        ids = [a[0] for a in pool_articles]
        features = [np.array(a[1:6]) for a in pool_articles]
        pta = [self.getPta(f, userID) for f in features]
        
        # pick the one with max pta
        index = np.argmax(pta)
        article_picked = ids[index]

        if article_picked == article_chosen:
            self.learn_stats.addRecord(click)
            self.updateParameters(features[index], click,userID)

        return article_picked   



class GOBLinStruct:
    def __init__(self, alpha, lambda_, d,  Gepsilon, userNum, userFeatureVectors):
        self.name = 'GOBLin'
        self.learn_stats = articleAccess()    

        self.alpha = alpha
        self.d = d
        self.userNum= userNum

        self.W = self.initializeGW(userFeatureVectors, Gepsilon)

        self.A = lambda_* np.identity(n = d*self.userNum)
        self.b = np.zeros(d*self.userNum)
        self.AInv = np.linalg.inv(self.A)
        
        self.theta = np.dot(np.linalg.inv(self.A), self.b)   # Long vector
        self.STBigWInv = sqrtm( np.linalg.inv(np.kron(self.W, np.identity(n=d))) )
        self.STBigW = sqrtm(np.kron(self.W, np.identity(n=d)))

    def initializeGW(self, FeatureVectors, Gepsilon):
        n = len(FeatureVectors)
        W = np.zeros(shape = (n, n))
            
        for i in range(n):
            sSim = 0
            for j in range(n):
                sim = np.dot(FeatureVectors[i],FeatureVectors[j])
                print 'sim',sim
                if i == j:
                    sim += 1
                W[i][j] = sim
                sSim += sim
                
            W[i] /= sSim
            for a in range(n):
                print '%.3f' % W[i][a],
            print ''
        G = W
        L = csgraph.laplacian(G, normed = False)
        I = np.identity(n)
        GW = I + Gepsilon*L  # W is a double stochastic matrix
        print GW          
        return GW.T

    def updateParameters(self, PickedfeatureVector, click, userID):
        featureVectorV = np.zeros(self.d*self.userNum)
        featureVectorV[int(userID)*self.d:(int(userID)+1)*self.d] = np.asarray(PickedfeatureVector)

        CoFeaV = np.dot(self.STBigWInv, featureVectorV)
        self.A += np.outer(CoFeaV, CoFeaV)
        self.b += click * CoFeaV

        self.AInv = np.linalg.inv(self.A)

        self.theta = np.dot(self.AInv, self.b)

    def getPta(self, featureVector, userID):

        featureVectorV = np.zeros(self.d*self.userNum)
        featureVectorV[int(userID)*self.d:(int(userID)+1)*self.d] = np.asarray(featureVector)
        
        CoFeaV = np.dot(self.STBigWInv, featureVectorV)

        mean = np.dot(np.transpose(self.theta), CoFeaV)
    
        var = np.sqrt( np.dot( np.dot(CoFeaV, self.AInv) , CoFeaV))
        pta = mean + self.alpha * var
        return pta

    def pick(self, pool_articles, userID, article_chosen, click):
        ids = [a[0] for a in pool_articles]
        features = [np.array(a[1:6]) for a in pool_articles]
        pta = [self.getPta(f, userID) for f in features]
        
        # pick the one with max pta
        index = np.argmax(pta)
        article_picked = ids[index]

        if article_picked == article_chosen:
            self.learn_stats.addRecord(click)
            self.updateParameters(features[index], click,userID)

        return article_picked

# ensemble model
class Ensemble:
    def __init__(self, alpha, lambda_, d,  Gepsilon, userNum, userFeatureVectors):
        self.name = 'ensemble'
        self.learn_stats = articleAccess()

        self.model = []
        self.model.append(LinUCBStruct(alpha, lambda_, d, userNum))
        self.model.append(CoLinUCBStruct(alpha, lambda_ , d, userNum, userFeatureVectors))
        self.model.append(GOBLinStruct(alpha, lambda_, d, Gepsilon, userNum,userFeatureVectors))
        self.modelNum = len(self.model)

        self.alpha = np.zeros(self.modelNum)
        self.beta = np.zeros(self.modelNum)

    def pick(self, pool_articles, userID, article_chosen, click):
        r = [beta.rvs(1+self.model[i].learn_stats.clicks, 1+self.model[i].learn_stats.accesses-self.model[i].learn_stats.clicks) 
             for i in range(self.modelNum)]
        m = np.argmax(r) #choose model
        # print 'choose model: ' + self.model[m].name

        article_picked = self.model[m].pick(pool_articles, userID, article_chosen, click)

        if article_picked == article_chosen:
            self.learn_stats.addRecord(click)

        return article_picked

        
           











