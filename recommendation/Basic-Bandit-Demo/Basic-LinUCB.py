import numpy as np

class LinUCB:
    def __init__(self):
        self.alpha = 0.25
        self.r1 = 0.6
        self.r0 = -16
        self.d = 6  # dimension of user features
        self.Aa = {} # Aa : collection of matrix to compute disjoint part for each article a, d*d
        self.AaI = {}  # AaI : store the inverse of all Aa matrix

        self.ba = {}  # ba : collection of vectors to compute disjoin part, d*1
        self.theta = {}

        self.a_max = 0



        self.x = None
        self.xT = None


    def set_articles(self,art):
        for key in art:
            self.Aa[key] = np.identity(self.d) # 创建单位矩阵
            self.ba[key] = np.zeros((self.d,1))

            self.AaI[key] = np.identity(self.d)
            self.theta[key] = np.zeros((self.d,1))



    def update(self,reward):
        if reward == -1:
            pass
        elif reward == 1 or reward == 0:
            if reward == 1:
                r = self.r1
            else:
                r = self.r0

            self.Aa[self.a_max] += np.dot(self.x,self.xT)
            self.ba[self.a_max] += r * self.x
            self.AaI[self.a_max] = np.linalg.inv(self.Aa[self.a_max])
            self.theta[self.a_max] = np.dot(self.AaI[self.a_max],self.ba[self.a_max])

        else:
            # error
            pass


    def recommend(self,timestamp,user_features,articles):
        xaT = np.array([user_features]) # d * 1
        xa = np.transpose(xaT)

        AaI_tmp = np.array([self.AaI[article] for article in articles])
        theta_tmp = np.array([self.theta[article] for article in articles])
        art_max = articles[np.argmax(np.dot(xaT,theta_tmp) + self.alpha * np.sqrt(np.dot(np.dot(xaT,AaI_tmp),xa)))]

        self.x = xa
        self.xT = xaT

        self.a_max = art_max

        return self.a_max








