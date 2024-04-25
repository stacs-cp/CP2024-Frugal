from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score

class PassiveRFModel:
    def __init__(self, model, type, name, encoder, data, labels):
        self.model = model
        self.model_type = type
        self.model_name = name
        self.encoder = encoder
        self.data = data
        self.labels = labels

    def get_model(self):
        return self.model
    
    def get_model_type(self):
        return self.model_type
    
    def get_model_name(self):
        return self.model_name
    
    def get_encoder(self):
        return self.encoder
    
    def get_data(self):
        return self.data
    
    def get_labels(self):
        return self.labels
    
    def print(self):
        print("Model: ", self.model)
        print("Model Name: ", self.model_name)
        print("Encoder: ", self.encoder)
        print("Data: ", self.data)
        print("Labels: ", self.labels)

    def fit(self, feature_columns, weight = "No_Weight", timeout = None, penalty = None):
        if weight == "Weighted":
            diff = (self.data[self.model_name].replace(timeout, timeout * penalty)[self.model_name[0]] - self.data[self.model_name].replace(timeout, timeout * penalty)[self.model_name[1]]).abs()
            self.model.fit(self.data[feature_columns], self.labels, sample_weight=diff.values)
        else:
            self.model.fit(self.data[feature_columns], self.labels)
    
    def predict(self, X):
        return self.model.predict(X)
    
    def predict_proba(self, X):
        return self.model.predict_proba(X)
    
    def score(self, X, y):
        print("Accuracy: ", round(accuracy_score(y, self.model.predict(X)), 2))
        print("F1 Score: ", round(f1_score(y, self.model.predict(X)) , 2))
        print("Precision: ", round(precision_score(y, self.model.predict(X)) , 2))
        print("Recall: ", round(recall_score(y, self.model.predict(X)), 2))