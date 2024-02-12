
import torch
import matplotlib.pyplot as plt
import argparse
from torchvision import transforms
from model import eeg_model
from custom_dataset import custom_dataset
import torch.nn.functional as F
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix


def test(model, loss_fn, test_loader, device):
    # test the model
    model.eval()  # set the model to evaluation mode
    test_loss = 0
    correct = 0
    predictions = []
    targets = []
    with torch.no_grad():
        for values, labels in test_loader:
            values = values.float().to(device)
            labels = labels.float().to(device)
            # labels = F.one_hot(labels, num_classes=2).float()
            outputs = model(values)
            outputs = outputs.squeeze(1)
            test_loss += loss_fn(outputs, labels).item()
            pred = outputs.round()
            correct += pred.eq(labels.view_as(pred)).sum().item()
            predictions.extend(pred.cpu().numpy())
            targets.extend(labels.cpu().numpy())
    accuracy = 100. * correct / len(test_loader.dataset)
    print('Test set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)'
            .format(test_loss/len(test_loader), correct, len(test_loader.dataset), accuracy)
    )
    confusion = confusion_matrix(targets, predictions)
    print('Confusion Matrix:')
    print(confusion)
    return accuracy


def train(n_epochs, optimizer, model, loss_fn, train_loader, test_loader, scheduler, device, output_plot_path, param_file):
    print ('training ...')
    model.train()      # keep track of gradient for backtracking
    losses_train = []
    for epoch in range(1, n_epochs+1):
        print('epoch ', epoch)
        loss_train = 0.0
        for values, labels in train_loader:
            # values = values[0]
            values = values.float().to(device)
            labels = labels.float().to(device)  # assuming labels are of type LongTensor
            # labels = F.one_hot(labels, num_classes=2).float()  # One-hot encode the labels
            outputs = model(values)
            outputs = outputs.squeeze(1)
            loss = loss_fn(outputs, labels)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            loss_train += loss.item()

        scheduler.step()

        losses_train += [loss_train/len(train_loader)] # update value of losses
        # validation_accuracy = test(model, loss_fn, test_loader, device)
        print('Epoch: {}, Training loss: {}'
              .format(epoch, loss_train/len(train_loader))
        )

    # plot output of training and save to output_plot_path file
    torch.save(model.state_dict(), param_file)
    plt.plot(losses_train)
    plt.savefig(output_plot_path)
    # Save the model as a .pth file
    # filename = 'model' + str(model_number) + '.pth'


# class Focus:
#     def __init(self, sensor_values, is_focused):
#         self.sensor_values = sensor_values
#         self.is_focused = is_focused

# Define the main method
def main():
    parser = argparse.ArgumentParser(description='Train a neural network model.')
    parser.add_argument('-e', type=int, default=25, help='Argument for e')
    parser.add_argument('-b', type=int, default=2, help='Argument for b')
    parser.add_argument('-p', type=str, default="output/output_path.png", help='Argument for p')

    args = parser.parse_args()

    # Check if all required arguments are provided
    if args.e is None or args.b is None or args.p is None:
        parser.print_help()
        return

    # Use the arguments to train the model
    batch_size = args.b
    n_epochs = args.e
    output_plot_path = args.p
    model_output_path = "output/model.pth"

    # train_transform = transforms.Compose([transforms.ToTensor()])
    dataset = custom_dataset("./Data")

    # Split the dataset into training and testing set
    train_data, test_data = train_test_split(dataset, test_size=0.2)

    # Create data loaders for both sets
    train_loader = torch.utils.data.DataLoader(train_data, batch_size=batch_size, shuffle=True)
    test_loader = torch.utils.data.DataLoader(test_data, batch_size=batch_size, shuffle=False)

    # Create an instance of the model, optimizer, and scheduler
    model = eeg_model()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-5)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.5)

    # implement loss function for binary classification
    loss_fn = torch.nn.BCELoss()

    device = torch.device("cpu")
    # Assuming you have the necessary model, optimizer, loss_fn, train_loader, scheduler, and device defined

    # train(n_epochs, optimizer, model, loss_fn, train_loader, test_loader, scheduler, device, output_plot_path, model_output_path)
    # accuracy = test(model, loss_fn, test_loader, device)

    train(n_epochs, optimizer, model, loss_fn, train_loader, test_loader, scheduler, device, output_plot_path, "model.pth")
    accuracy = test(model, loss_fn, test_loader, device)
    print(f"Accuracy: {accuracy}")

# Execute the main method if this script is run directly
if __name__ == '__main__':
    main()
