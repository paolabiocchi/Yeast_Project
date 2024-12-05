import pandas as pd
import numpy as np
from sklearn.model_selection import GridSearchCV, KFold
from sklearn.metrics import mean_squared_error
import os.path
from preprocessing import *
from skorch import NeuralNetRegressor
from torch import nn, optim
from skorch.callbacks import EarlyStopping
from skorch.callbacks import EpochScoring
import matplotlib.pyplot as plt

np.random.seed(10)

X_file = "data/X_matrix.csv"
Y_file = "data/Y_matrix.csv"

print("Loading the data...")
x_df = pd.read_csv(X_file)
y_df = pd.read_csv(Y_file)
x_data_f, y_data_f = preprocessed_data(x_df, y_df)

#x, y, x_test_try = preprocessed_data(path_train, path_cddd, path_test)


# Préparation des données
print("Préparation des données...")
x_data_f1 = x_data_f.drop(columns=["Yeast_ID"]).fillna(0)  # Remplacer les valeurs manquantes par 0 dans X
y_data_f1 = y_data_f["YPD_doublingtime"].fillna(y_data_f["YPD_doublingtime"].mean())  # Remplacer les valeurs manquantes par la moyenne dans Y

# Number of input features
n_input_features = x_data_f1.shape[1]

# Define an enhanced neural network
class EnhancedRegressionNet(nn.Module):
    def __init__(self, n_input_features, dropout_rate, n_neurons=128):
        super(EnhancedRegressionNet, self).__init__()
        self.fc1 = nn.Linear(n_input_features, n_neurons) #n_input_features
        self.relu1 = nn.ReLU()
        self.dropout1 = nn.Dropout(dropout_rate)
        self.fc2 = nn.Linear(n_neurons, n_neurons)
        self.relu2 = nn.ReLU()
        self.dropout2 = nn.Dropout(dropout_rate)
        self.fc3 = nn.Linear(n_neurons, 1)

    def forward(self, x):
        x = self.relu1(self.fc1(x))
        x = self.dropout1(x)
        x = self.relu2(self.fc2(x))
        x = self.dropout2(x)
        x = self.fc3(x)
        return x.squeeze()


# Define scoring callbacks for training and validation loss
train_loss = EpochScoring(scoring='neg_mean_squared_error', on_train=True, name='train_loss', lower_is_better=False)
valid_loss = EpochScoring(scoring='neg_mean_squared_error', name='valid_loss', lower_is_better=False)


#Neural Network Regressor
net = NeuralNetRegressor(
    module=EnhancedRegressionNet,
    module__n_input_features=n_input_features , #n_input_features
    criterion=nn.MSELoss,
    optimizer=optim.Adam,
    iterator_train__shuffle=True,
    callbacks=[EarlyStopping(patience=5)],
    verbose=1
)

#parameter grid
param_grid = {
    'module__dropout_rate': [0.008, 0.009, 0.01],
    'lr': [0.00014, 0.00015, 0.00016],
    'max_epochs': [69, 70, 71],
    'optimizer': [optim.Adam],
}



# GridSearchCV 
grid_search = GridSearchCV(net, param_grid=param_grid, cv=KFold(n_splits=6), scoring='neg_mean_squared_error', n_jobs=-1)
grid_search.fit(x_data_f1.values.astype(np.float32), y_data_f1.values.astype(np.float32))

# Get the best parameters from the grid search
best_params = grid_search.best_params_
print("Best Parameters:", best_params)

# Training of the model
best_net = NeuralNetRegressor(
    module=EnhancedRegressionNet,
    module__n_input_features=n_input_features,
    module__n_neurons=128,
    module__dropout_rate=best_params['module__dropout_rate'],
    criterion=nn.MSELoss,
    max_epochs=best_params['max_epochs'],
    optimizer=best_params['optimizer'],
    lr=best_params['lr'],
    iterator_train__shuffle=True,
    callbacks=[EarlyStopping(patience=5), train_loss, valid_loss],
    verbose=1
)
best_net.fit(x_data_f1.values.astype(np.float32), y_data_f1.values.astype(np.float32))

Y_pred = best_net.predict(x_data_f1.values.astype(np.float32))

# Evaluate the model for the train data
mse = mean_squared_error(y_data_f1, Y_pred)
print(f'Mean Squared Error: {mse}')

id_array = np.arange(1, len(Y_pred)+1)
final_df = pd.DataFrame({
    'ID': id_array,
    'division_rate': Y_pred.flatten()
})

# Save the new DataFrame to a CSV file
final_csv = final_df.to_csv("Data\\results_nn3.csv", index=False)


# Extract training and validation loss for a plot
train_losses = best_net.history[:, 'train_loss']
valid_losses = best_net.history[:, 'valid_loss']

plt.plot(train_losses, label='Training loss')
plt.plot(valid_losses, label='Validation loss')
plt.xlabel('Epochs')
plt.ylabel('Negative Mean Squared Error')
plt.title('Training and Validation Loss per Epoch')
plt.legend()
plt.savefig("Data\\NNplot_nn3.png")