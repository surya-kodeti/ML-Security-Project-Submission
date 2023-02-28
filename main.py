from models import ConvNet, MlleaksMLP
import torch.optim as optim
import torch
import torch.nn as nn
from train_eval import train, eval_model, train_attacker, eval_attacker
from custom_dataloader import dataloader
import os
import argparse




parser = argparse.ArgumentParser()
parser.add_argument('--dataset', default='cifar', help='The dataset of choice between "cifar" and "mnist".')
parser.add_argument('--batch_size', default=64, type=int, help='The batch size used for training.')
parser.add_argument('--epoch', default=20, type=int, help='Number of epochs for shadow and target model.')
parser.add_argument('--attack_epoch', default=50, type=int, help='Number of epochs for attack model.')
parser.add_argument('--only_eval', default=False, type=bool, help='If true, only evaluate trained loaded models.')
parser.add_argument('--save_new_models', default=False, type=bool, help='If true, trained models will be saved.')
args = parser.parse_args()


def main():
    dataset = args.dataset
    shadow_path, target_path, attack_path = "./models/shadow_" + str(dataset) + ".pth", \
                                            "./models/target_" + str(dataset) + ".pth", \
                                            "./models/attack_" + str(dataset) + ".pth"

   
    if dataset == "cifar":
        input_size = 3
    elif dataset == "mnist":
        input_size = 1

    n_epochs = args.epoch
    attack_epochs = args.attack_epoch
    batch_size = args.batch_size

    
    shadow_train_loader = dataloader(dataset=dataset, batch_size_train=batch_size, batch_size_test=1000,
                                     split_dataset="shadow_train")
    shadow_out_loader = dataloader(dataset=dataset, batch_size_train=batch_size, batch_size_test=1000,
                                   split_dataset="shadow_out")
    target_train_loader = dataloader(dataset=dataset, batch_size_train=batch_size, batch_size_test=1000,
                                     split_dataset="target_train")
    target_out_loader = dataloader(dataset=dataset, batch_size_train=batch_size, batch_size_test=1000,
                                   split_dataset="target_out")

    testloader = dataloader(dataset=dataset, batch_size_train=batch_size, batch_size_test=1000,
                            split_dataset="test")

    
    target_net = shadow_net = ConvNet(input_size=input_size)

    
    target_loss = shadow_loss = nn.CrossEntropyLoss()
    target_optim = optim.Adam(target_net.parameters(), lr=0.001)
    shadow_optim = optim.Adam(shadow_net.parameters(), lr=0.001)

    
    attack_net = MlleaksMLP()
    
    attack_loss = nn.BCELoss()
    attack_optim = optim.Adam(attack_net.parameters(), lr=0.001)

   

    if os.path.exists(shadow_path):
        print("Load shadow model")
        shadow_net.load_state_dict(torch.load(shadow_path))
    
    if not args.only_eval:
        print("start training shadow model: ")
        for epoch in range(n_epochs):
            loss_train_shadow = train(shadow_net, shadow_train_loader, shadow_loss, shadow_optim, verbose=False)
            
            if (epoch+1) % 5 == 0:
                accuracy_train_shadow = eval_model(shadow_net, shadow_train_loader, report=False)
                accuracy_test_shadow = eval_model(shadow_net, testloader, report=True)
                print("Shadow model: epoch[%d/%d] Train loss: %.5f training set accuracy: %.5f  test set accuracy: %.5f"
                      % (epoch + 1, n_epochs, loss_train_shadow, accuracy_train_shadow, accuracy_test_shadow))
            if args.save_new_models:
                if not os.path.exists("./models"):
                    os.mkdir("./models")  
                
                torch.save(shadow_net.state_dict(), "./models/shadow_" + str(dataset) + ".pth")

    if os.path.exists(target_path):
        print("Load target model")
        target_net.load_state_dict(torch.load(target_path))
    
    if not args.only_eval:
        print("start training target model: ")
        for epoch in range(n_epochs):
            loss_train_target = train(target_net, target_train_loader, target_loss, target_optim, verbose=False)
            
            if (epoch + 1) % 5 == 0:
                accuracy_train_target = eval_model(target_net, target_train_loader, report=False)
                accuracy_test_target = eval_model(target_net, testloader, report=True)
                print("Target model: epoch[%d/%d] Train loss: %.5f training set accuracy: %.5f  test set accuracy: %.5f"
                      % (epoch + 1, n_epochs, loss_train_target, accuracy_train_target, accuracy_test_target))
            if args.save_new_models:
                
                if not os.path.exists("./models"):
                    os.mkdir("./models")  
                torch.save(target_net.state_dict(), target_path)

    if os.path.exists(attack_path):
        print("Load attack model")
        attack_net.load_state_dict(torch.load(attack_path))
    
    if not args.only_eval:
        print("start training attacker model")
        for epoch in range(attack_epochs):
            loss_attack = train_attacker(attack_net, shadow_net, shadow_train_loader, shadow_out_loader, attack_optim, attack_loss, num_posterior=3, verbose=False)
            
            if (epoch+1) % 1 == 0:
                max_accuracy = eval_attacker(attack_net, target_net, target_train_loader, target_out_loader, num_posterior=3)
                print("Attack model: epoch[%d/%d]  Train loss: %.5f  Accuracy on target set: %.5f"
                      % (epoch + 1, attack_epochs, loss_attack, max_accuracy))
                if args.save_new_models:
                    if not os.path.exists("./models"):
                        os.mkdir("./models") 
                    torch.save(attack_net.state_dict(), attack_path)

    
    if args.only_eval:
        print("Classification Report Shadow Net:")
        eval_model(shadow_net, testloader, report=True)
        print("Classification Report Target Net:")
        eval_model(target_net, testloader, report=True)
        print("Report of Attack Net")
        eval_attacker(attack_net, target_net, target_train_loader, target_out_loader, num_posterior=3)


if __name__ == '__main__':
    main()
