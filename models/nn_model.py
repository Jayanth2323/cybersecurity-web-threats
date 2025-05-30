from keras.models import Sequential
from keras.layers import Dense
from sklearn.preprocessing import StandardScaler


def build_nn_model(input_dim):
    model = Sequential([
        Dense(64, activation='relu', input_dim=input_dim),
        Dense(32, activation='relu'),
        Dense(1, activation='sigmoid')
    ])
    model.compile(
        optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    return model


def preprocess_and_train_nn(X, y):
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    model = build_nn_model(X.shape[1])
    model.fit(X_scaled, y, epochs=10, batch_size=16, verbose=0)
    return model, scaler
