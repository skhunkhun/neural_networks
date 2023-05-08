Sunveer Khunkhun
April 13 2023

<!-- HOW TO RUN CODE -->

    -To run all of the code and see the graph type in the command line 'python3 main.py' and everything should run at once (it could take up to 15 minutes).
     - If main.py is run, the graph that will be displayed uses 'matplotlib.use('TkAgg')', so it is best to make sure 'TkAgg' is installed.

    -The programs can also be run individually:
        - 'python3 a4Zero.py' for 0 hidden layers NN
        - 'python3 a4One.py' for 1 hidden layers NN
        - 'python3 a4Two.py' for 2 hidden layers NN
        - 'python3 a4Three.py' for 3 hidden layers NN

    ** Running the programs will output a line after each epoch and then the test accuracy at the end:
        - e.g
            - 'Epoch 1/20 - loss: 0.0314 - acc: 0.8750 - valid_loss: 0.0232 - valid_acc: 0.9334' ... 'Epoch 20/20 - loss: 0.0355 - acc: 0.9062 - valid_loss: 0.0275 - valid_acc: 0.9231'
            - 'Test accuracy: 0.9248'

<!-- IMPLEMENTATION -->

    ** All training was done with an epoch value of 20

    - a4Zero.py 
        - contains an NN with 0 hidden layers, but still contains a softmax function for decent accuracy
        - input: 784
        - output: 10
        - Validation accuracy: 0.9247
    
    - a4One.py
        - contains an NN with 1 hidden layer. Uses the relu and softmax functions
        - input: 784
        - HL1: 128
        - output: 10
        - Validation accuracy: 0.9769

    - a4Two.py
        - contains an NN with 2 hidden layers. Uses relu and softmax functions
        - input: 784
        - HL1: 256
        - HL2: 128
        - output 10
        - Validation accuracy: 0.9801

    - a4Three.py
        - contains an NN with 3 hidden layers. Uses relu and softmax functions
        - input: 784
        - HL1: 512
        - HL2: 256
        - HL3: 128
        - output 10
        - Validation accuracy: 0.9766

<!-- DISCUSSION -->

    - Architecture 1 (0 hidden layers)
        - With no hidden layers, the network is essentially just a linear regression model with a softmax inclusion.
        As a result, it had the lowest accuracy compared to all of the other neural networks since minimal training was actually done

    - Architecture 2 (1 hidden layer)
        - With one hidden layer, the network started to learn more complex patterns in the data. However, the hidden layer was too small, 
            and the network may not have been able to capture all the relevant information in the dataset. The result was 5% in accuracy over architecture 1

    - Architecture 3 (2 hidden layers)
        - With two hidden layers, the network learned even more complex patterns in the data. The result was a slight increase in accuracy over architecture 2

    - Architecture 4 (3 hidden layers)
        - With three hidden layers, the network had the potential to learn even more complex patterns in the data, but with this dataset, there was overfitting,
        which lead to inconsitent accuracies and an overall lower accuracies than both architecture 1 and architecture 2

    Overall, the best architecture for a neural network on the MNIST dataset was with 2 hidden layers. It had the best accuracies and the ability to learn complex patterns in the data without being
    prone to the overfitting that architecture 4 experienced. 
    



    