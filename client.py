
import flwr as fl
from model import create_model
from data_preprocessing import load_data, preprocess_data, split_data
import tensorflow as tf
import numpy as np
from sklearn.metrics import roc_auc_score, precision_score, recall_score, f1_score, roc_curve
import matplotlib.pyplot as plt

# Define a function to distribute data among clients for simulation
def distribute_data(X, y, num_clients):
    '''Distribute data and labels to simulate a federated learning setup.'''
    X_split = np.array_split(X, num_clients)
    y_split = np.array_split(y, num_clients)
    return list(zip(X_split, y_split))

class FLClient(fl.client.NumPyClient):
    def __init__(self, model, X_train, y_train, X_test, y_test):
        self.model = model
        self.X_train, self.y_train = X_train, y_train
        self.X_test, self.y_test = X_test, y_test

    def get_parameters(self):
        return self.model.get_weights()

    def fit(self, parameters, config):
        self.model.set_weights(parameters)
        history = self.model.fit(self.X_train, self.y_train, epochs=2, batch_size=32, verbose=0)
        # Only return the last loss and accuracy values
        final_loss = history.history['loss'][-1]
        final_accuracy = history.history['accuracy'][-1]
        return self.model.get_weights(), len(self.X_train), {"loss": final_loss, "accuracy": final_accuracy}

    def evaluate(self, parameters, config):
        self.model.set_weights(parameters)
        loss, accuracy = self.model.evaluate(self.X_test, self.y_test, verbose=0)
        
        # Obtain model predictions
        y_pred_probs = self.model.predict(self.X_test).ravel()
        y_pred = (y_pred_probs > 0.5).astype(int)
        
        # Calculate metrics
        roc_auc = roc_auc_score(self.y_test, y_pred_probs)
        precision = precision_score(self.y_test, y_pred, zero_division=0)
        recall = recall_score(self.y_test, y_pred, zero_division=0)
        f1 = f1_score(self.y_test, y_pred, zero_division=0)
        
        # ROC Curve
        fpr, tpr, thresholds = roc_curve(self.y_test, y_pred_probs)
        plt.figure()
        plt.plot(fpr, tpr, label='ROC curve (area = %0.2f)' % roc_auc)
        plt.plot([0, 1], [0, 1], 'k--')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('Receiver Operating Characteristic')
        plt.legend(loc="lower right")
        plt.show()
        
        return loss, len(self.X_test), {
            "accuracy": accuracy, 
            "roc_auc": roc_auc,
            "precision": precision, 
            "recall": recall, 
            "f1_score": f1
        }

def main():
    # Load and preprocess data
    data = load_data('ORAB_Annotation_MIMIC.csv')
    X, y = preprocess_data(data)
    
    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = split_data(X, y, test_size=0.2)
    
    # Create a model
    model = create_model(input_shape=X_train.shape[1])
    
    # Distribute training data among clients (testing data remains the same for evaluation)
    num_clients = 3
    client_data = distribute_data(X_train, y_train, num_clients)
    
    # Start Flower client
    client_idx = 2  # For simulation purposes, if running this script multiple times, change the index
    fl.client.start_numpy_client("localhost:8080", client=FLClient(
        model,
        client_data[client_idx][0],
        client_data[client_idx][1],
        X_test,
        y_test,
    ))

if __name__ == "__main__":
    main()
