from torchvision import datasets
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from torch.utils.data import SubsetRandomSampler
import sys




def subsetloader(ls_indices, start, end, trainset, batch_size):
   
    ids = ls_indices[start:end]
    sampler = SubsetRandomSampler(ids)
    loader = DataLoader(trainset, batch_size=batch_size, sampler=sampler)
    return loader





def dataloader(dataset="cifar", batch_size_train=8, batch_size_test=1000, split_dataset="shadow_train"):
   
    try:
        if dataset == "cifar":

            transform = transforms.Compose(
                [transforms.ToTensor(),
                 transforms.Normalize((0.4914, 0.4822, 0.4465), (0.247, 0.243, 0.261))])
            trainset = datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
            testset = datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)
            testloader = DataLoader(testset, batch_size=batch_size_test, shuffle=False)

        elif dataset == "mnist":

            transform = transforms.Compose([transforms.ToTensor(),
                                            transforms.Normalize((0.1307,), (0.3081,))])
            trainset = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
            testset = datasets.MNIST(root="./data", train=False, download=True, transform=transform)
            testloader = DataLoader(testset, batch_size=batch_size_test, shuffle=False)
        else:
            raise NotAcceptedDataset

    except NotAcceptedDataset:
        print('Dataset Error. Choose "cifar" or "mnist"')
        sys.exit()

    total_size = len(trainset)
    split1 = total_size // 4
    split2 = split1 * 2
    split3 = split1 * 3

    indices = [*range(total_size)]

    if split_dataset == "shadow_train":
        return subsetloader(indices, 0, split1, trainset, batch_size_train)

    elif split_dataset == "shadow_out":
        return subsetloader(indices, split1, split2, trainset, batch_size_train)

    elif split_dataset == "target_train":
        return subsetloader(indices, split2, split3, trainset, batch_size_train)

    elif split_dataset == "target_out":
        return subsetloader(indices, split3, total_size, trainset, batch_size_train)

    else:
        return testloader





class NotAcceptedDataset(Exception):
    
    pass


