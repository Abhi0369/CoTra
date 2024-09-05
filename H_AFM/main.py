from corr_vit import ViT
from helper_functions import save_checkpoint,save_experiment
import torch
from torch import nn
import torch.optim as optim
from Data import train_loader,test_loader
from configs import config


class Trainer:
    """
    The simple trainer.
    """

    def __init__(self, model, optimizer, loss_fn, exp_name, device):
        self.model = model.to(device)
        self.optimizer = optimizer
        self.loss_fn = loss_fn
        self.exp_name = exp_name
        self.device = device


    def train(self, trainloader, testloader, epochs, save_model_every_n_epochs=0):
        """
        Train the model for the specified number of epochs.
        """
        # Keep track of the losses and accuracies
        train_losses, test_losses, accuracies= [], [], []
        # Train the model
        for i in range(epochs):
            train_loss,labels,drp= self.train_epoch(trainloader)
            accuracy, test_loss,preds,lab = self.evaluate(testloader)

            train_losses.append(train_loss)
            test_losses.append(test_loss)
            accuracies.append(accuracy)
            print(f"Epoch: {i+1}, Train loss: {train_loss:.4f}, Test loss: {test_loss:.4f}, Accuracy: {accuracy:.4f}")
            if save_model_every_n_epochs > 0 and (i+1) % save_model_every_n_epochs == 0 and i+1 != epochs:
                print('\tSave checkpoint at epoch', i+1)
                save_checkpoint(self.exp_name, self.model, i+1)

        save_experiment(self.exp_name, config, self.model, train_losses, test_losses, accuracies)

    def train_epoch(self, trainloader):
        """
        Train the model for one epoch.
        """
        self.model.train()
        all_drp =[]
        total_loss = 0
        for batch in trainloader:
            # Move the batch to the device
            batch = [t.to(self.device) for t in batch]
            images, labels = batch
            # Zero the gradients
            self.optimizer.zero_grad()
            # Calculate the loss
            logits, _,_ = self.model(images)

            loss =self.loss_fn(logits,labels)  

            l1_loss = self.model.output.norm(1)


            gamma = config["gamma"]
            loss += gamma * l1_loss
            loss.backward(retain_graph=False)

            self.optimizer.step()

            total_loss += loss.item() * len(images) 
    
            all_drp=[0]
        return total_loss / len(trainloader.dataset),labels,all_drp

    @torch.no_grad()
    def evaluate(self, testloader):
        self.model.eval()
        total_loss = 0
        correct = 0

        with torch.no_grad():
            for batch in testloader:
                # Move the batch to the device
                batch = [t.to(self.device) for t in batch]
                images, labels = batch

                # Get predictions
                logits, attn,_ = self.model(images)
   
                loss = self.loss_fn(logits,labels)

    
                l1_loss = self.model.output.norm(1)

                gamma = config["gamma"]
                loss += gamma * l1_loss


                total_loss += loss.item() * len(images) 
                # Calculate the accuracy
                predictions = torch.argmax(logits, dim=1)
                correct += torch.sum(predictions == labels).item()

    #                 print(predictions,labels)
        accuracy = correct / len(testloader.dataset)
        avg_loss = total_loss / len(testloader.dataset)
        return accuracy, avg_loss, predictions,labels


def main():

    seed = config['seed']
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    batch_size = config['batch_size']
    epochs = config['epochs']
    lr = config['lr']


    save_model_every_n_epochs = 2000
    # Load the dataset

    trainloader,testloader=train_loader, test_loader  
                                                             
    # Create the model, optimizer, loss function and trainer
    model = ViT(config)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.CrossEntropyLoss()

    device= "cuda" if torch.cuda.is_available() else "cpu"


    trainer = Trainer(model, optimizer, loss_fn, 'my_exp', device=device)
    trainer.train(trainloader, testloader, epochs, save_model_every_n_epochs=save_model_every_n_epochs)

    
if __name__ == "__main__":
    main()

    