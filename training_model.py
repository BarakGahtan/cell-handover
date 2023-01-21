# Measure our neural network by mean square error
import torch
from torch import optim


def train(model, training_count, data_set_train, output):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print('A {} device was detected.'.format(device))

    x = torch.tensor(data_set_train, dtype=torch.float, device=device)
    y = torch.tensor(output, dtype=torch.float, device=device)
    criterion = torch.nn.MSELoss()
    # Train our network with a simple SGD optimizer
    optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9)
    # Train our network a using the entire dataset 5 times
    for epoch in range(training_count):
        totalLoss = 0
        for i in range(len(x)):
            # Single Forward Pass
            ypred = model(x[i])

            # Measure how well the model predicted vs the actual value
            loss = criterion(ypred, y[i])

            # Track how well the model predicted (called loss)
            totalLoss += loss.item()

            # Update the neural network
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        # Print out our loss after each training iteration
        print("Total Loss: ", totalLoss)
