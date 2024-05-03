import pickle
import matplotlib.pyplot as plt
import pickle
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Flatten, Conv1D, MaxPooling1D
from tensorflow.keras.utils import to_categorical
from sklearn.metrics import accuracy_score
from pyriemann.estimation import Covariances
from pyriemann.classification import MDM


def load_data(paths):
   data = []
   for path in paths:
       with open(path, 'rb') as f:
           data.append(pickle.load(f))
   return data

def evaluate_models(train_sig, test_sigs):
   fs = 1000
   t_pre = 1 * fs
   t_post = 2 * fs
   epochs, labels = process_signals(train_sig, t_pre, t_post)
  
   # Train models on training data
   mdm_riemann, mdm_euclid, ann_model, cnn_model = train_models(epochs, labels)
  
   # Evaluate models on testing data
   num_runs = 3  # Number of times to repeat testing for each subject
   all_accuracies = {}
  
   for i, test_sig in enumerate(test_sigs):
       subject_accuracies = []
       for _ in range(num_runs):
           epochs_test, labels_test = process_signals(test_sig, t_pre, t_post)
           accuracy = test_models(epochs_test, labels_test, mdm_riemann, mdm_euclid, ann_model, cnn_model)
           subject_accuracies.append(accuracy)
       all_accuracies[f"Subject {i+1}"] = subject_accuracies
      
   return all_accuracies

def process_signals(sig, t_pre, t_post):
   epochs = []
   labels = []
   events = np.where(sig['states'] != 4)[0]
   sample_size = int(0.01 * len(events))
   sampled_events = np.random.choice(events, size=sample_size, replace=False)
  
   for event in sampled_events:
       if event - t_pre >= 0 and event + t_post < sig['value'].shape[1]:
           epoch_states = sig['states'][event - t_pre:event + t_post]
           if np.unique(epoch_states).size == 1:
               epochs.append(sig['value'][:, event - t_pre:event + t_post])
               labels.append(epoch_states[0])
  
   return np.array(epochs), np.array(labels)

def train_models(epochs, labels):
   # Train MDM with Riemannian metric
   cov_estimator = Covariances(estimator='oas')
   cov_matrices = cov_estimator.fit_transform(epochs)
  
   mdm_riemann = MDM(metric='riemann', n_jobs=1)
   mdm_euclid = MDM(metric='euclid', n_jobs=1)
  
   X_train, y_train = cov_matrices, labels
   mdm_riemann.fit(X_train, y_train)
   mdm_euclid.fit(X_train, y_train)
  
   # Train ANN and CNN
   scaler = StandardScaler()
   epochs_reshaped = epochs.reshape(epochs.shape[0], -1)
   epochs_scaled = scaler.fit_transform(epochs_reshaped)
   epochs_scaled = epochs_scaled.reshape(epochs.shape)
   labels_categorical = to_categorical(labels)
  
   ann_model = train_ann(epochs_scaled, labels_categorical)
   cnn_model = train_cnn(epochs_scaled, labels_categorical)
  
   return mdm_riemann, mdm_euclid, ann_model, cnn_model

def test_models(epochs, labels, mdm_riemann, mdm_euclid, ann_model, cnn_model):
   accuracies = {}
  
   # Test MDM with Riemannian metric
   cov_estimator = Covariances(estimator='oas')
   cov_matrices = cov_estimator.transform(epochs)
  
   y_pred_riemann = mdm_riemann.predict(cov_matrices)
   y_pred_euclid = mdm_euclid.predict(cov_matrices)
  
   accuracies['MDM (Riemann)'] = accuracy_score(labels, y_pred_riemann)
   accuracies['MDM (Euclid)'] = accuracy_score(labels, y_pred_euclid)


   # Test ANN and CNN
   scaler = StandardScaler()
   epochs_reshaped = epochs.reshape(epochs.shape[0], -1)
   epochs_scaled = scaler.fit_transform(epochs_reshaped)
   epochs_scaled = epochs_scaled.reshape(epochs.shape)
   labels_categorical = to_categorical(labels)
  
   accuracies['ANN'] = ann_model.evaluate(epochs_scaled, labels_categorical)[1]
   accuracies['CNN'] = cnn_model.evaluate(epochs_scaled, labels_categorical)[1]
  
   return accuracies

def train_ann(X_train, y_train):
   ann_model = Sequential([
       Dense(128, activation='relu', input_shape=(X_train.shape[1], X_train.shape[2])),
       Dropout(0.5),
       Dense(64, activation='relu'),
       Dropout(0.5),
       Dense(32, activation='relu'),
       Flatten(),
       Dense(4, activation='softmax')
   ])
   ann_model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
   ann_model.fit(X_train, y_train, epochs=5, batch_size=1024, validation_split=0.1, verbose=1)
   return ann_model

def train_cnn(X_train, y_train):
   cnn_model = Sequential([
       Conv1D(filters=32, kernel_size=3, padding='same', activation='relu', input_shape=(X_train.shape[1], X_train.shape[2])),
       MaxPooling1D(pool_size=2),
       Conv1D(filters=64, kernel_size=3, padding='same', activation='relu'),
       MaxPooling1D(pool_size=2),
       Flatten(),
       Dense(64, activation='relu'),
       Dense(4, activation='softmax')
   ])
   cnn_model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
   cnn_model.fit(X_train, y_train, epochs=5, batch_size=1024, validation_split=0.1, verbose=1)
   return cnn_model

def number_to_state(number):
   state_correspondence = {
       0: 'asynch_MA',
       1: 'awake',
       2: 'slow_MA',
       3: 'slow_updown',
       4: 'unknown'
   }
   return state_correspondence.get(number, "unknown state")

list_subjects = [
"Subject2.pkl",
"Subject1.pkl",
"Subject3.pkl"
                ]


subjects_data = load_data(list_subjects)
train_sig = subjects_data[0]
test_sigs = subjects_data[1:]


accuracies = evaluate_models(train_sig, test_sigs)
print(accuracies)

