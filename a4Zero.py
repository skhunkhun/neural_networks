from headers import *

def nnZero (plt):
    # Load the MNIST dataset
    (X_train, y_train), (X_test, y_test) = mnist.load_data()

    # Preprocess the data
    X_train = X_train.reshape(X_train.shape[0], -1) / 255.0
    X_test = X_test.reshape(X_test.shape[0], -1) / 255.0
    y_train = np.eye(10)[y_train]
    y_test = np.eye(10)[y_test]

    # Initialize the plot points for training and testing
    x_train_arr = []
    y_train_arr = []

    x_test_arr = []
    y_test_arr = []

    # Define the neural network class of zero
    class NeuralNetworkZero:
        
        def __init__(self, input_dim, output_dim):
            # Initialize the weights
            self.W = np.random.randn(input_dim, output_dim) * 0.01
            self.b = np.zeros((1, output_dim))
        
        def softmax(self, Z):
            expZ = np.exp(Z)
            return expZ / np.sum(expZ, axis=1, keepdims=True)
        
        def forward(self, X):
            # Compute the forward pass
            Z = np.dot(X, self.W) + self.b
            A = self.softmax(Z)
            return A
        
        def backward(self, X, y, A):
            # Compute the gradients using backpropagation
            m = y.shape[0]
            dZ = A - y
            dW = np.dot(X.T, dZ) / m
            db = np.sum(dZ, axis=0, keepdims=True) / m
            return dW, db
        
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
                    y_prediction = self.forward(X_batch)

                    # Compute the loss and accuracy
                    loss = -np.mean(y_batch * np.log(y_prediction))
                    acc = np.mean(np.argmax(y_batch, axis=1) == np.argmax(y_prediction, axis=1))

                    # Backward pass
                    dW, db = self.backward(X_batch, y_batch, y_prediction)

                    # Update the weights
                    self.W -= learning_rate * dW
                    self.b -= learning_rate * db

                y_valid_prediction = self.forward(X_val)
                valid_loss = -np.mean(y_val * np.log(y_valid_prediction))
                valid_acc = np.mean(np.argmax(y_val, axis=1) == np.argmax(y_valid_prediction, axis=1))
                x_test_arr.append(epoch + 1)
                y_test_arr.append(valid_loss * 100)

                # Print the loss and accuracy for each epoch
                print(f"Epoch {epoch+1}/{epochs} - loss: {loss:.4f} - acc: {acc:.4f} - valid_loss: {valid_loss:.4f} - valid_acc: {valid_acc:.4f}")
                x_train_arr.append(epoch + 1)
                y_train_arr.append(loss * 100)

    nn = NeuralNetworkZero(784, 10)

    # Train the neural network using the training data
    nn.train(X_train, y_train)

    # Evaluate the performance of the neural network on the test data
    y_prediction = nn.forward(X_test)
    test_acc = np.mean(np.argmax(y_test, axis=1) == np.argmax(y_prediction, axis=1))
    print(f"Test accuracy: {test_acc:.4f}")

    # Add to graph
    plt.plot(x_train_arr, y_train_arr, label="Train 0")
    plt.plot(x_test_arr, y_test_arr, label="Test 0")

nnZero(plt)