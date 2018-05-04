from models import *
from conf import *
import time, datetime
from scipy.spatial import distance


# read centroids from file
def getClusters(fileNameWriteCluster):
    with open(fileNameWriteCluster, 'r') as f:
        clusters = []
        for line in f:
            vec = []
            line = line.split(' ')
            for i in range(len(line) - 1):
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
        f.write('data')  # the observation line starts with data;
        f.write(',' + str(tim))
        f.write(',' + ';'.join([str(x) for x in recordedStats]))
        f.write('\n')


if __name__ == '__main__':
    # regularly print stuff to see if everything is going alright.
    # this function is inside main so that it shares variables with main and I dont wanna have large number of function arguments
    def printWrite(fileNameWrite):
        randomLearnCTR = random_model.learn_stats.getCTR()
        CTR = model.learn_stats.getCTR()

        print totalObservations
        print 'random', randomLearnCTR, model.name, CTR

        recordedStats = [randomLearnCTR, CTR]
        # write to file
        save_to_file(fileNameWrite, recordedStats, tim)


    timeRun = datetime.datetime.now().strftime('_%m_%d_%H_%M')  # the current data time
    # dataDays = ['01', '02', '03', '04', '05', '06', '07', '08', '09', '10']
    dataDays = ['10']
    batchSize = 2000  # size of one batch

    d = 5  # feature dimension
    alpha = 0.3  # control how much to explore
    lambda_ = 0.2  # regularization used in matrix A
    epsilon = 0.3

    totalObservations = 0

    fileNameWriteCluster = os.path.join(data_address, 'kmeans_model_80.dat')
    userFeatureVectors = getClusters(fileNameWriteCluster)
    userNum = len(userFeatureVectors)

    random_model = randomStruct()
    model1 = CoLinUCBStruct(alpha, lambda_, d, userNum, userFeatureVectors)
    model2 = GOBLinStruct(alpha, lambda_, d, epsilon, userNum, userFeatureVectors)
    model3 = LinUCBStruct(alpha, lambda_, d, userNum)
    model = HyperLinUCB(alpha, lambda_, d, epsilon, userNum, userFeatureVectors)

    output = []

    for dataDay in dataDays:
        fileName = "./data" + "/ydata-fp-td-clicks-v1_0.200905" + dataDay
        # fileNameWrite = os.path.join('./results', model.name + '_20_' + dataDay + timeRun + '.csv')

        # put some new data in file for readability
        # with open(fileNameWrite, 'a+') as f:
        #     f.write('\nNew Run at  ' + datetime.datetime.now().strftime('%m/%d/%Y %H:%M:%S'))
        #     f.write('\n, Time,random;' + model.name + '\n')

        # print fileName, fileNameWrite

        with open(fileName, 'r') as f:
            # reading file line ie observations running one at a time
            for line in f:
                # if totalObservations > 2*batchSize:
                #     break
                totalObservations += 1

                tim, article_chosen, click, user_features, pool_articles = parseLine(line)
                currentUser_featureVector = user_features[:-1]

                currentUserID = getIDAssignment(np.asarray(currentUser_featureVector), userFeatureVectors)

                model1.pick(pool_articles, currentUserID, article_chosen, click)
                model2.pick(pool_articles, currentUserID, article_chosen, click)
                model3.pick(pool_articles, currentUserID, article_chosen, click)
                model.pick(pool_articles, currentUserID, article_chosen, click, currentUser_featureVector)
                random_model.pick(pool_articles, currentUserID, article_chosen, click)

                # if the batch has ended
                if totalObservations % batchSize == 0:
                    base = random_model.learn_stats.getCTR()
                    # print base
                    s = str(totalObservations) + ' ' + str(base) + ' ' + str(model1.learn_stats.getCTR()) + ' ' + str(
                        model2.learn_stats.getCTR()) + ' ' + str(model3.learn_stats.getCTR()) + ' ' + str(
                        model.learn_stats.getCTR())
                    print s
                    s = str(totalObservations) + ' ' + str(model1.learn_stats.getCTR() / base) + ' ' + str(
                        model2.learn_stats.getCTR() / base) + ' ' + str(model3.learn_stats.getCTR() / base) + ' ' + str(
                        model.learn_stats.getCTR() / base)
                    print s
                    output.append(s)

        # printWrite()
        with open('./results/plot_stat_80.csv', 'w') as f:
            f.write('\n'.join(output))






