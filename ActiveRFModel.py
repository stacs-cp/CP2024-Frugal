import pandas as pd
import numpy as np

class ActiveRFModel:
    def __init__(self, learner, type, name, encoder, initial_data, pool_data, sample_weight = None):
        self.learner = learner
        self.model_type = type
        self.model_name = name
        self.encoder = encoder
        self.initial_train_data = initial_data
        self.pool_data = pool_data
        self.sample_weight = sample_weight

    def get_learner(self):
        return self.learner
    
    def get_model_type(self):
        return self.model_type
    
    def get_model_name(self):
        return self.model_name
    
    def get_encoder(self):
        return self.encoder
    
    def get_initial_train_data(self):
        return self.initial_train_data
    
    def get_pool_data(self):
        return self.pool_data
    
    def predict(self, X):
        return self.learner.predict(X)
    
    def teach(self, feature_columns, weight = "No_Weight"):
        if weight == "Weighted":
            self.learner.teach(X = self.initial_train_data[feature_columns], y = self.label_transform(), sample_weight=self.sample_weight, only_new=True)
        else:
            self.learner.teach(X = self.initial_train_data[feature_columns], y = self.label_transform(), only_new=True)

    def label_transform(self):
        y = self.initial_train_data[[*self.model_name]].idxmin(axis=1).values
        y = self.encoder.transform(y)

        return y

    def print(self):
        print("Model Name: ", self.model_name)
        print("Type: ", self.model_type)
        print("Encoder: ", self.encoder)
        print("Initial Data:", self.initial_train_data)
        print("Pool Data", self.pool_data)

    def update_initial_train_data_and_pool(self, index, weight = "No_Weight", timeout_limit = None, penalty = None):
        if weight == "Weighted":
            self.sample_weight = np.append(self.sample_weight, (self.pool_data.loc[index].clip(upper = timeout_limit).replace(timeout_limit, timeout_limit * penalty)[[*self.model_name][0]] - self.pool_data.loc[index].clip(upper = timeout_limit).replace(timeout_limit, timeout_limit * penalty)[[*self.model_name][1]]).abs())
        
        self.initial_train_data = pd.concat([self.initial_train_data, self.pool_data.loc[index].clip(upper = timeout_limit)], axis=0)
        self.pool_data.drop(index, inplace=True)

    def delete_from_pool(self, index):
        self.pool_data.drop(index, inplace=True)
