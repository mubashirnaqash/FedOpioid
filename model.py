import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.optimizers import Adam
def create_model(input_shape):
    # Define a neural network model
    model = Sequential([
        Dense(128, activation='relu', input_shape=(input_shape,)),  # Correct input_shape usage
        Dropout(0.5),
        Dense(64, activation='relu'),
        Dropout(0.5),
        Dense(1, activation='sigmoid')  # Assuming binary classification. Adjust if needed.
    ])
    model.compile(optimizer=Adam(learning_rate=0.001), loss='binary_crossentropy', metrics=['accuracy'])
    return model

def main():
    # Example usage
    model = create_model(input_shape=1011)  # Replace 11 with the actual number of features after preprocessing
    model.summary()  # Display the model structure

if __name__ == "__main__":
    main()
