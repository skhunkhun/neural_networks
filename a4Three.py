from headers import *

def nnThree(plt):
    # Load the MNIST dataset
    (X_train, y_train), (X_test, y_test) = mnist.load_data()

    # Preprocess the data
    X_train = X_train.reshape(X_train.shape[0], -1) / 255.0
    X_test = X_test.reshape(X_test.shape[0], -1) / 255.0
    y_train = np.eye(10)[y_train]
    y_test = np.eye(10)[y_test]

    x_train_arr = []
    y_train_arr = []

    x_test_arr = []
    y_test_arr = []

    # Define the neural network class of three
    class NeuralNetorkThree:
        
        def __init__(self, input_dim, hidden_dim1, hidden_dim2, hidden_dim3, output_dim):
            # Initialize the weights
            self.W1 = np.random.randn(input_dim, hidden_dim1) * 0.01
            self.b1 = np.zeros((1, hidden_dim1))
            self.W2 = np.random.randn(hidden_dim1, hidden_dim2) * 0.01
            self.b2 = np.zeros((1, hidden_dim2))
            self.W3 = np.random.randn(hidden_dim2, hidden_dim3) * 0.01
            self.b3 = np.zeros((1, hidden_dim3))
            self.W4 = np.random.randn(hidden_dim3, output_dim) * 0.01
            self.b4 = np.zeros((1, output_dim))
        
        def relu(self, Z):
            return np.maximum(0, Z)
        
        def softmax(self, Z):
            expZ = np.exp(Z)
            return expZ / np.sum(expZ, axis=1, keepdims=True)
        
        def forward(self, X):
            # Compute the forward pass
            Z1 = np.dot(X, self.W1) + self.b1
            A1 = self.relu(Z1)
            Z2 = np.dot(A1, self.W2) + self.b2
            A2 = self.relu(Z2)
            Z3 = np.dot(A2, self.W3) + self.b3
            A3 = self.relu(Z3)
            Z4 = np.dot(A3, self.W4) + self.b4
            A4 = self.softmax(Z4)
            return A4, A3, A2, A1
        
        def backward(self, X, y, A4, A3, A2, A1):
            # Compute the gradients using backpropagation
            m = y.shape[0]
            dZ4 = A4 - y
            dW4 = np.dot(A3.T, dZ4) / m
            db4 = np.sum(dZ4, axis=0, keepdims=True) / m
            dZ3 = np.dot(dZ4, self.W4.T) * (A3 > 0)
            dW3 = np.dot(A2.T, dZ3) / m
            db3 = np.sum(dZ3, axis=0, keepdims=True) / m
            dZ2 = np.dot(dZ3, self.W3.T) * (A2 > 0)
            dW2 = np.dot(A1.T, dZ2) / m
            db2 = np.sum(dZ2, axis=0, keepdims=True) / m
            dZ1 = np.dot(dZ2, self.W2.T) * (A1 > 0)
            dW1 = np.dot(X.T, dZ1) / m
            db1 = np.sum(dZ1, axis=0, keepdims=True) / m
            return dW1, db1, dW2, db2, dW3, db3, dW4, db4
        
        def train(self, X, y, learning_rate=0.1, epochs=20, batch_size=32):
            # Split the data into train, test, and validation sets (60/20/20)
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
            X_val, X_test, y_val, y_test = train_test_split(X_test, y_test, test_size=0.25, random_state=42)

            # Loop over the epochs
            for epoch in range(epochs):
                # Loop over the batches
                for i in range(0, X_train.shape[0], batch_size):
                    X_batch = X_train[i:i+batch_size]
                    y_batch = y_train[i:i+batch_size]

                    # Forward pass
                    y_prediction, A3, A2, A1 = self.forward(X_batch)

                    # Compute the loss and accuracy
                    loss = -np.mean(y_batch * np.log(y_prediction))
                    acc = np.mean(np.argmax(y_batch, axis=1) == np.argmax(y_prediction, axis=1))

                    # Backward pass
                    dW1, db1, dW2, db2, dW3, db3, dW4, db4 = self.backward(X_batch, y_batch, y_prediction, A3, A2, A1)

                    # Update the weights
                    self.W1 -= learning_rate * dW1
                    self.b1 -= learning_rate * db1
                    self.W2 -= learning_rate * dW2
                    self.b2 -= learning_rate * db2
                    self.W3 -= learning_rate * dW3
                    self.b3 -= learning_rate * db3
                    self.W4 -= learning_rate * dW4
                    self.b4 -= learning_rate * db4

                # Compute the performance on the validation set
                y_valid_prediction, _, _, _ = self.forward(X_val)
                valid_loss = -np.mean(y_val * np.log(y_valid_prediction))
                valid_acc = np.mean(np.argmax(y_val, axis=1) == np.argmax(y_valid_prediction, axis=1))
                x_test_arr.append(epoch + 1)
                y_test_arr.append(valid_loss * 100)

                # Print the loss and accuracy for each epoch
                print(f"Epoch {epoch+1}/{epochs} - loss: {loss:.4f} - acc: {acc:.4f} - valid_loss: {valid_loss:.4f} - valid_acc: {valid_acc:.4f}")
                x_train_arr.append(epoch + 1)
                y_train_arr.append(loss * 100)

    nn = NeuralNetorkThree(784, 512, 256, 128, 10)

    # Train the neural network using the training data
    nn.train(X_train, y_train)

    # Evaluate the performance of the neural network on the test data
    y_prediction, _, _, _ = nn.forward(X_test)
    test_acc = np.mean(np.argmax(y_test, axis=1) == np.argmax(y_prediction, axis=1))
    print(f"Test accuracy: {test_acc:.4f}")

    # Add to graph
    plt.plot(x_train_arr, y_train_arr, label="Train 3")
    plt.plot(x_test_arr, y_test_arr, label="Test 3")

nnThree(plt)