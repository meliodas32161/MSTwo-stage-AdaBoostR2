"""
Multi-source TrAdaBoostR2 algorithm

based on algorithm 3 in paper "Boosting for Regression Transfer".

"""

import numpy as np
import copy
from sklearn import tree
from sklearn.tree import DecisionTreeRegressor
from sklearn.gaussian_process import GaussianProcessRegressor 
from sklearn.gaussian_process.kernels import WhiteKernel, RBF, Matern
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import KFold


def _return_std(X, trees, predictions, min_variance):
    std = np.zeros(len(X))
    for tree in trees:
        var_tree = tree.tree_.impurity[tree.apply(X)]

        
        var_tree[var_tree < min_variance] = min_variance
        mean_tree = tree.predict(X)
        std += var_tree + mean_tree ** 2

    std /= len(trees)
    std -= predictions ** 2.0
    std[std < 0.0] = 0.0
    std = std ** 0.5
    return std
################################################################################
# H 测试样本分类结果
# Train_T 原训练样本 np数组  目标域数据
# Train_S 辅助训练样本  np数组   源域数据
# Label_T 原训练样本标签  np数组
# Label_S 辅助训练样本标签  np数组
# Test  测试样本
# N 迭代次数
# trans_S是一个列表，里面储存多个源域。算法初始化之前先读取trans_S的源域的数量。

class StageTwoAdaboost():
    def __init__(self,
                 base_estimator = DecisionTreeRegressor(max_depth=4),
                 learning_rate = 1.,
                 sample_size = None,   # 因为中间域的存在，因此源域和目标域的规模是变化的
                 loss = 'linear',
                 random_state = np.random.mtrand._rand):
        self.base_estimator = base_estimator
        self.learning_rate = learning_rate
        self.loss = loss
        self.random_state = random_state
        self.sample_size = sample_size  # sample_size代表本来的目标域的大小，不考虑中间域因素
        
        
    def fit(self, X, Y, sample_weight = None):
        
        if self.learning_rate <= 0:
            raise ValueError("learning_rate must be greater than zero")

        if sample_weight is None:
            # Initialize weights to 1 / n_samples
            sample_weight = np.empty(X.shape[0], dtype=np.float64)
            sample_weight[:] = 1. / X.shape[0]
        else:
            # Normalize existing weights
            sample_weight = sample_weight / sample_weight.sum(dtype=np.float64)
            # Check that the sample weights sum is positive
            if sample_weight.sum() <= 0:
                raise ValueError(
                      "Attempting to fit with a non-positive "
                      "weighted number of samples.")
                
        # Clear any previous fit results
        self.estimators_ = []
        self.estimator_weights_ = np.zeros(self.n_estimators, dtype=np.float64)
        self.estimator_errors_ = np.ones(self.n_estimators, dtype=np.float64)
        
        for iboost in range(self.n_estimators):
            # Boosting step
            sample_weight, estimator_weight, estimator_error = self._stage2_adaboostR2(
                    iboost,
                    X, Y,
                    sample_weight)
            
            
        
        
        def _stage2_adaboostR2(self,iboost,X,Y,sample_weight):
            self.middle_size = 0 # 初始化中间域规模
            estimator = copy.deepcopy(self.base_estimator)
            
            cdf = np.cumsum(sample_weight)  # 得到的是一个数组，最后一个值才是所有的值相加的和，所以是cdf[-1]
            cdf /= cdf[-1]  # 归一化
            uniform_samples = self.random_state.random_sample(X.shape[0]) # 生成数量为训练集数量的01之间的随机数
            
            bootstrap_idx = cdf.searchsorted(uniform_samples, side='right') # 把uniform_samples插入到升序的cdf中，返回插入元素的索引，返回一个scalar
            bootstrap_idx = np.array(bootstrap_idx,copy=False)
            
            # 在求cdf的时候，如果某个权重比较大，那么在cdf求和中，他和前一个值的距离就更大。在随机生成uniform_samples的时候就有更大的概率落在
            # 这个权重和上一个权重之间，即选择这个样本的概率更高。这就是样本赋予权重的意义
            estimator.fit(X[bootstrap_idx], Y[bootstrap_idx])
            y_predict = estimator.predict(X)
            error_vect = np.abs(y_predict - Y)
            error_max = error_vect.max()
            if error_max != 0.:
                error_vect /= error_max

            if self.loss == 'square':
                error_vect **= 2
            elif self.loss == 'exponential':
                error_vect = 1. - np.exp(- error_vect)
            # Calculate the average loss
            estimator_error = (sample_weight * error_vect).sum()
            if estimator_error <= 0:
            # Stop if fit is perfect
                return sample_weight, 1., 0.
            elif estimator_error >= 0.5:
               # Discard current estimator only if it isn't the only one
               if len(self.estimators_) > 1:
                   self.estimators_.pop(-1)
               return None, None, None
           
            beta = estimator_error / (1. - estimator_error)
            # avoid overflow of np.log(1. / beta)
            if beta < 1e-308:
                beta = 1e-308
            # 在AdaboostR2中，学习器的权重estimator_weight用ln（1/beta）表示    
            estimator_weight = self.learning_rate * np.log(1. / beta)  
            
#%%  设计中间域，根据error_vect的值，从源域中选择样本填充进中间域

            # 目标域规模不确定
            X_source,Y_source,source_weight, X_target, Y_target, target_weight = Middledomine(iboost, 
                                                                                              error_vect, 
                                                                                              X_source, y_source,
                                                                                              X_target, y_target,
                                                                                              bootstrap_idx)

            # Boost weight using AdaBoost.R2 alg except the weight of the source data
            # the weight of the source data are remained  源域权重保持不变
        
            source_weight_sum= np.sum(sample_weight[:-self.sample_size[-1]]) / np.sum(sample_weight)
            target_weight_sum = np.sum(sample_weight[-self.sample_size[-1]:]) / np.sum(sample_weight)
            
            if not iboost == self.n_estimators - 1:
                sample_weight[-self.sample_size[-1]:] *= np.power(
                        beta,
                        (1. - error_vect[-self.sample_size[-1]:]) * self.learning_rate)  # 更改目标域样本实例权重
                # make the sum weight of the source data not changing
                source_weight_sum_new = np.sum(sample_weight[:-self.sample_size[-1]]) / np.sum(sample_weight)
                target_weight_sum_new = np.sum(sample_weight[-self.sample_size[-1]:]) / np.sum(sample_weight)
                if source_weight_sum_new != 0. and target_weight_sum_new != 0.:
                    sample_weight[:-self.sample_size[-1]] = sample_weight[:-self.sample_size[-1]]*source_weight_sum/source_weight_sum_new
                    sample_weight[-self.sample_size[-1]:] = sample_weight[-self.sample_size[-1]:]*target_weight_sum/target_weight_sum_new
    
            return sample_weight, estimator_weight, estimator_error
        
        
        def Middledomine(self,iboost,
                         error_vect, 
                         X_source,Y_source,
                         X_target,Y_target,
                         sample_weight,
                         bootstrap_idx):
            """
                input: iboost 当前迭代数，判断是否需要从中间域拿出数据
                       error_vect 每个样本的误差
                       X_source/X_target 源域/目标域数据
                       Y_source/Y_target 源域/目标域标签
                       sample_weight 权重序列
                       bootstrap_idx 训练所用的数据的索引值
                      
                output:X_source/X_target 新的源域/目标域数据
                       Y_source/Y_target 新的源域/目标域标签 
                       index_sourcenew 新的源域数据索引
                       index_middle 中间域数据索引
                       index_targetnex 新的目标域数据索引
                  
            """
            
            len_source = X_source.shape[0]
            len_target = X_target.shape[0]
            error_s = []
            error_t = []
            index_s = []
            index_t = []
            index_m = []
            source_weight = sample_weight[:-(self.sample_size+self.middle_size)]
            target_weight = sample_weight[-(self.sample_size+self.middle_size):]
            # Y_middle = []
            # 将误差分为源域的误差和目标域的误差
            for i,error_ in enumerate(bootstrap_idx):
                sample_size = X_target.shape[0]  # 当前目标域的规模
                if error_ < len_source:
                    error_s.append((error_vect[i]))  # 同时存储误差和对应的索引
                    index_s.append(error_)
                elif error_ < (len_source + len_target) and error_ >= len_source:
                    error_t.append(error_vect[i])
                    index_t.append(error_)
            
            error_t_min = min(error_t) # 目标域最小误差
            error_t_max = max(error_t) # 目标域最大误差
            
            # 根据源域误差，选择现有中间域中不适合的数据放回源域(中间域数据放到目标域前面)
            if iboost != 0:
                for index, errors in enumerate(error_t[:-self.sample_size]):  # 保证了索引不会超出中间域的大小
                    if errors >= error_t_max:
                        middle_temp = X_target[index]  # 储存该数据以及其对应的权重
                        midweight_temp = target_weight[index]
                        np.delete(X_target, index)  # 从当前中间域删除该数据
                        np.delete(target_weight, index) # 从当前中间域权重删除该权重
                        np.append(X_source, middle_temp)  # 将该数据放到源域数据后面
                        np.append(source_weight, midweight_temp)  # 将该数据对应权重放回源域权重中
            # 根据源域误差，选择适合放进中间域的数据          
            for index, errors in enumerate(error_s):
                if errors <= error_t_min + 1e-30:
                    middle_temp = X_source[index]
                    midweight_temp = source_weight[index]
                    np.delete(X_source, index)    # 从当前源域删除该数据
                    np.delete(source_weight, index)
                    np.insert(X_target, 0, middle_temp)  # 将该数据放到中间域数据前面
                    np.insert(target_weight, 0, midweight_temp)  
                    
            
            # # 根据源域误差，选择现有中间域中不适合的数据放回源域
            # if iboost == 0:
            #     X_middle = np.zeros(len_source)
            #     # middle_weight = np.zeros(len_source)
            # else:      
            #     for errors in enumerate(error_s):#,index_s:
            #         if errors[1][0] >= error_t_max:
            #            np.delete(X_middle, errors[1][1])  # 从当前中间域删除该数据
            #            np.insert(X_source, errors[1][1], errors[1][0])  # 将该数据放回源域的相应位置
            #            np.insert(source_weight, errors[1][1],)  # 将该数据对应权重放回源域权重中
            #         # if errors >= error_t_max - 1e-30:
            #         #     np.delete(X_middle, index)
            #         #     np.delete(X_middle, index)
            #         #     np.insert(X_source, index, values)
            # # 根据源域误差，选择适合放进中间域的数据      
                
            # for errors in enumerate(error_s):
            #     if errors[1][0] <= error_t_min + 1e-30:
            #         np.delete(X_source, errors[1][1])    # 从当前源域删除该数据
            #         np.insert(X_middle, errors[1][1], errors[1][0])  # 将该数据放入当前中间域
                    
            
            
            # X_target = np.concatenate(X_middle,X_target)  # 将中间域与源域相结合
            
            # # index_middle = 
            # sample_size = len_target + X_middle     
            
            # 返回目标域数据
            return X_source,Y_source,source_weight, X_target, Y_target, target_weight 
        
        def predict(self, X):
            # Evaluate predictions of all estimators
            predictions = np.array([
                    est.predict(X) for est in self.estimators_[:len(self.estimators_)]]).T
    
            # Sort the predictions
            sorted_idx = np.argsort(predictions, axis=1)
    
            # Find index of median prediction for each sample
            weight_cdf = np.cumsum(self.estimator_weights_[sorted_idx], axis=1)
            median_or_above = weight_cdf >= 0.5 * weight_cdf[:, -1][:, np.newaxis]
            median_idx = median_or_above.argmax(axis=1)
    
            median_estimators = sorted_idx[np.arange(X.shape[0]), median_idx]
    
            # Return median predictions
            return predictions[np.arange(X.shape[0]), median_estimators]
    
    


class MSTradaBoost():
    def __init__(self,epoch=10,learner = GaussianProcessRegressor(),
                 sample_size = None,
                 fold = 5,
                 error_split = None,
                 learning_rate = 1.,
                 loss = 'linear',
                 random_state = np.random.mtrand._rand):
        self.epoch = epoch # 单个模型样本实例权重更新的迭代次数
        # self.steps = steps  # 单个模型样本实例权重更新的迭代次数
        self.learner = learner
        # self.sample_size = sample_size
        self.beta = None
        self.beta_Tree = None
        self.result = None
        self.row_T = None
        self.row_S = []
        self.models = []
        self.fold = fold
        self.error_split = error_split # 该参数用于选择M个模型中选择多少（比例）认为是好的模型
        
    def fit(self,trans_T, trans_S, label_T, label_S):
        self.sample_size = trans_T.shape[0]  # 目标域大小
        M = len(trans_S)  # number of source domain,trans_S应当是一个列表
        trans_data = []
        trans_label = []
        # target_weights = []  # 目标域样本实例的权重
        # source_weights = []  # 源域样本实例权重
        weights = []  # 样本实例权重
        self.weight_model = np.zeros([1, self.epoch])  # 模型权重初始化
        self.row_T = trans_T.shape[0]
        self.errors_ = []
        
        # 初始化：将M个源域数据与目标域数据结合生成M个训练数据，并初始化样本实例权重
        for i in range(M):
            trans_data.append(np.concatenate(trans_S[i],trans_T), axis=0)
            trans_label.append(np.concatenate((label_S[i], label_T), axis=0))
            self.row_S.append(trans_S[i].shape[0])
            weights.append(1. / (self.row_S[i] + self.row_T))  # 样本实例权重初始化
            
            self.beta = 1 / (1 + np.sqrt(2 * np.log(self.row_S[i] / self.epoch)))
            # self.row_S.append(trans_S[i].shape[0])
            
       
        for j in range(M):
            P = self.calculate_P(weights[j], trans_label)# 将样本实例权值转换为分布
            
            total = np.sum(weights[j]) #/(self.row_S[j]+self.row_T)
            sample_weights = np.asarray(weights[j]/total, order='C')
            self.learner.fit(trans_data[j], trans_label[j], sample_weight = P)
            self.models.append(self.learner)
            kf = KFold(n_splits = self.fold)
            error = []
            target_weight = P[-self.row_T:]
            source_weight = P[:-self.row_T]
            
            # k折交叉验证计算M个模型总体精度（MSE）存在self.errors_中
            for train,test in kf.split(trans_T):
                X_train = np.concatenate((trans_S[j],trans_T[train]))
                y_train = np.concatenate((label_S[j] ,label_T[train]))
                X_test = trans_T[test]
                y_test = trans_T[test]
                
                # make sure the sum weight of the target data do not change with CV's split sampling
                target_weight_train = target_weight[train]*np.sum(target_weight)/np.sum(target_weight[train])
                self.learner.fit(X_train, y_train, sample_weights = np.concatenate((source_weight, target_weight_train)))
                y_predict = self.learner.predict(X_test)
                error.append(mean_squared_error(y_predict, y_test))
            
            self.errors_.append(np.array(error).mean())
       
        # # 根据MSE计算每个模型的整体误差率，用于模型权重调整
        # error_rate = self.calculate_error_rate_model(self.errors_)
#%%计算模型权重        
        # 根据误差重新调整模型权值。误差小的权值变大，误差大的权值减小
        self.weight_model = self.calculate_P(1/np.exp(self.errors_)) # 数组从小到大排序,获取对应索引
        # 判断 error_split 是否合乎规则
        if self.error_split == None:
            self.error_split = np.floor(M/2)
        elif self.error_split >= M:
            raise ValueError(" error_split must be less then the number of sources")
        elif self.error_split%1 != 0:
            raise ValueError("error_split must be an integer")
        self.model_index = np.argsort(self.weight_model)[::-1][0:self.error_split]  # 找到了表现最好的(权重最大的)error_split个模型的索引,索引从大到小
        
        # # 只选择表现好的模型用于下面的迁移学习过程
        # self.models = self.models[model_index]
#%% 样本实例权重更新过程        
        self.model = []
        temp = 0
        for i in self.model_index:  
            X_source = trans_data[i][:-self.sample_size[-1]]
            y_source = trans_label[i][:-self.sample_size[-1]]
            X_target = trans_data[i][-self.sample_size[-1]:]
            y_target = trans_label[i][-self.sample_size[-1]:]    

            
            # target_weight = sample_weight[-self.sample_size[-1]:]
            # source_weight = sample_weight[:-self.sample_size[-1]]
            for j in range(self.epoch):
                kf = KFold(n_splits = self.fold)
                error = []
 
                P = self.calculate_P(weights[i])
                target_weight = P[-self.sample_size[-1]:]
                source_weight = P[:-self.sample_size[-1]]
                
                for train, test in kf.split(X_target):
                    sample_size = [self.sample_size[0], len(train)]
                    model = StageTwoAdaboost(self.base_estimator,
                                        sample_size = self.sample_size,
                                        n_estimators = self.n_estimators,
                                        learning_rate = self.learning_rate, 
                                        loss = self.loss,
                                        random_state = self.random_state)
                    X_train = np.concatenate((X_source, X_target[train]))
                    y_train = np.concatenate((y_source, y_target[train]))
                    X_test = X_target[test]
                    y_test = y_target[test]
                    # make sure the sum weight of the target data do not change with CV's split sampling 确保目标数据的总权重不随CV的分割采样而变化
                    target_weight_train = target_weight[train]*np.sum(target_weight)/np.sum(target_weight[train])
                    model.fit(X_train, y_train, sample_weight = np.concatenate((source_weight, target_weight_train)))
                    y_predict = model.predict(X_test)
                    error.append(mean_squared_error(y_predict, y_test))
                self.errors.append(np.array(error).mean())
                X = trans_data[i]
                Y = trans_label[i]
                sample_weight = self._twostage_adaboostR2(j, X, Y, sample_weight=P)
                
                
                
                # 调整源域样本权重
                for k in range(self.row_S[j]):
                    #beta = error_rate/(1-error_rate),wt+1 = wt*beta^(1-et)
                    # weights[j][k] =  weights[j][k]*np.power(self.beta_Tree[0,k],1 - np.abs(label_S_predict[k] - label_S[j][k]))
                    
                    #beta = error_rate/(1-error_rate),wt+1 = wt*beta^(exp(-et)+1-e-1)
                    weights[j][k] = weights[j][k]*np.power(self.beta_Tree[0,k],
                                                            np.exp(- np.abs(label_S_predict[k] - label_S[j][k]))+1-np.exp(-1))
                
                # 调整目标域样本权重
                for k in range(self.row_T):
                    weights[j][self.row_S[j] + k] = weights[j][self.row_S[j] + k]
                P = self.calculate_P(weights[j], trans_label)
        
            
        def calculate_P(self, weights, label):
            total = np.sum(weights)
            return np.asarray(weights / total, order='C')
        
        
        
        def calculate_error_rate(self,label_T_predict,label_T, weights):
            total = np.sum(weights)
            maximize = np.max(np.abs(label_T_predict - label_T))
            if maximize != 0 :
                epsilon = np.sum(weights[:, 0] / maximize * np.abs(label_T_predict - label_T))
            
            return epsilon
        
        def _twostage_adaboostR2(self, istep, X, y, sample_weight):

            estimator = copy.deepcopy(self.base_estimator) # some estimators allow for specifying random_state estimator = base_estimator(random_state=random_state)
    
            ## using sampling method to account for sample_weight as discussed in Drucker's paper
            # Weighted sampling of the training set with replacement
            cdf = np.cumsum(sample_weight) # 将数组按行或者列累加
            cdf /= cdf[-1]
            uniform_samples = self.random_state.random_sample(X.shape[0])
            bootstrap_idx = cdf.searchsorted(uniform_samples, side='right')
            # searchsorted returns a scalar
            bootstrap_idx = np.array(bootstrap_idx, copy=False)
    
            # Fit on the bootstrapped sample and obtain a prediction
            # for all samples in the training set
            estimator.fit(X[bootstrap_idx], y[bootstrap_idx])
            y_predict = estimator.predict(X)
    
    
            error_vect = np.abs(y_predict - y)
            error_max = error_vect.max()
    
            if error_max != 0.:
                error_vect /= error_max
    
            if self.loss == 'square':
                error_vect **= 2
            elif self.loss == 'exponential':
                error_vect = 1. - np.exp(- error_vect)
    
            # Update the weight vector
            beta = self._beta_binary_search(istep, sample_weight, error_vect, stp = 1e-80)
            
            # 在第一阶段只改变源域权重
    
            if not istep == self.steps - 1:
                sample_weight[:-self.sample_size[-1]] *= np.power(
                        beta,
                        (error_vect[:-self.sample_size[-1]]) * self.learning_rate)
            return sample_weight


        def _beta_binary_search(self, istep, sample_weight, error_vect, stp):
            # calculate the specified sum of weight for the target data
            n_target = self.sample_size[-1]
            n_source = np.array(self.sample_size).sum() - n_target
            theoretical_sum = n_target/(n_source+n_target) + istep/(self.steps-1)*(1-n_target/(n_source+n_target))
            # for the last iteration step, beta is 0.
            if istep == self.steps - 1:
                beta = 0.
                return beta
            # binary search for beta
            L = 0.
            R = 1.
            beta = (L+R)/2
            sample_weight_ = copy.deepcopy(sample_weight)
            sample_weight_[:-n_target] *= np.power(
                        beta,
                        (error_vect[:-n_target]) * self.learning_rate)# 更新源域实例权重 wt+1 = wt*beta^error_vect
            sample_weight_ /= np.sum(sample_weight_, dtype=np.float64)# 归一化总体实例权重
            updated_weight_sum = np.sum(sample_weight_[-n_target:], dtype=np.float64)# 目标域实例权重求和
            
    
            while np.abs(updated_weight_sum - theoretical_sum) > 0.01:# 要求当前目标域实例权重之和不能大于m/(m+n) + t/(s-1)*(1-t/(n+m))
            # 在目标域权重之和不大于设定的最大值的时候，就可以不断减小源域实例的权重，通过把beta放在左半部分（即降低Right，让right为中间值），直到无法再搜索为止
                if updated_weight_sum < theoretical_sum:
                    R = beta - stp
                    if R > L:
                        beta = (L+R)/2
                        sample_weight_ = copy.deepcopy(sample_weight)
                        sample_weight_[:-n_target] *= np.power(
                                    beta,
                                    (error_vect[:-n_target]) * self.learning_rate)
                        sample_weight_ /= np.sum(sample_weight_, dtype=np.float64)
                        updated_weight_sum = np.sum(sample_weight_[-n_target:], dtype=np.float64)
                    else:
                        print("At step:", istep+1)
                        print("Binary search's goal not meeted! Value is set to be the available best!")
                        print("Try reducing the search interval. Current stp interval:", stp)
                        break
    
                elif updated_weight_sum > theoretical_sum:
                    L = beta + stp
                    if L < R:
                        beta = (L+R)/2
                        sample_weight_ = copy.deepcopy(sample_weight)
                        sample_weight_[:-n_target] *= np.power(
                                    beta,
                                    (error_vect[:-n_target]) * self.learning_rate)
                        sample_weight_ /= np.sum(sample_weight_, dtype=np.float64)
                        updated_weight_sum = np.sum(sample_weight_[-n_target:], dtype=np.float64)
                    else:
                        print("At step:", istep+1)
                        print("Binary search's goal not meeted! Value is set to be the available best!")
                        print("Try reducing the search interval. Current stp interval:", stp)
                        break
            return beta
        
        # def update_model_weight(self,error_rate):
        #     return np.exp(error_rate - 1) * self.weight_model
        def predict(self,test_data):
            model = self.models
            self.row_Test = test_data.shape[0]
            
            results = []
            result = np.ones([self.row_Test, self.epoch])
