import torch
import torch.nn.functional as F
from tqdm import tqdm
from sklearn.metrics import classification_report
import numpy as np



def train(model, data_loader, criterion, optimizer, verbose=True):
    
    running_loss = 0
    model.train()
    for step, (batch_img, batch_label) in enumerate(tqdm(data_loader)):
        optimizer.zero_grad()  
        output = model(batch_img)  
        loss = criterion(output, batch_label)
        loss.backward()  
        optimizer.step()  
        running_loss += loss
        
        if verbose:
            print("[%d/%d] loss = %f" % (step, len(data_loader), loss.item()))
    return running_loss


def eval_model(model, test_loader, report=True):
    
    total = 0
    correct = 0
    gt = []
    preds = []
    with torch.no_grad():  
        model.eval()
        for step, (batch_img, batch_label) in enumerate(tqdm(test_loader)):

            output = model(batch_img)

            predicted = torch.argmax(output, dim=1)
            preds.append(predicted)

            gt.append(batch_label)

            total += batch_img.size(0)
            correct += torch.sum(batch_label == predicted)

    accuracy = 100 * (correct/total)
    if report:
        gt = torch.cat(gt, 0)
        preds = torch.cat(preds, 0)
        print(classification_report(gt.cpu(), preds.cpu()))

    return accuracy




def train_attacker(attack_net, shadow, shadow_train, shadow_out, optimizer, criterion, num_posterior,verbose=True):

    
    shadow_net = shadow
    shadow_net.eval()

    total = 0
    correct = 0
    running_loss = 0

    
    for step, ((train_img, _), (out_img, _)) in enumerate(tqdm(zip(shadow_train, shadow_out))):

       
        if train_img.shape[0] != out_img.shape[0]:
            break

        minibatch_size = train_img.shape[0]

      
        train_posteriors = F.softmax(shadow_net(train_img.detach()), dim=1)
        out_posteriors = F.softmax(shadow_net(out_img.detach()), dim=1)

        
        train_sort, _ = torch.sort(train_posteriors, descending=True)
        out_sort, _ = torch.sort(out_posteriors, descending=True)

        
        train_top_k = train_sort[:, :num_posterior].clone()
        out_top_k = out_sort[:, :num_posterior].clone()

        train_labels = torch.ones(minibatch_size)
        out_labels = torch.zeros(minibatch_size)

        optimizer.zero_grad()

        train_predictions = torch.squeeze(attack_net(train_top_k))
        out_predictions = torch.squeeze(attack_net(out_top_k))

        
        loss_train = criterion(train_predictions, train_labels)
        loss_out = criterion(out_predictions, out_labels)

        loss = (loss_train + loss_out) / 2

        loss.backward()  
        optimizer.step()  
        running_loss += loss

        if verbose:
            correct += (F.sigmoid(train_predictions) >= 0.5).sum().item()
            correct += (F.sigmoid(out_predictions) < 0.5).sum().item()
            total += train_predictions.size(0) + out_predictions.size(0)
            accuracy = 100 * correct / total

            print("[%d/%d] loss = %.2f, accuracy = %.2f" % (step, len(shadow_train), loss.item(), accuracy))

        return running_loss


def eval_attacker(attack_model, target_model, target_train, target_out, num_posterior):
    

    with torch.no_grad():

        target_model.eval()
        attack_model.eval()

        precisions = []
        recalls = []
        accuracies = []

        thresholds = np.arange(0.50, 0.80, 0.01)  
        total = np.zeros(len(thresholds))
        correct = np.zeros(len(thresholds))
        true_positives = np.zeros(len(thresholds))
        false_positives = np.zeros(len(thresholds))
        false_negatives = np.zeros(len(thresholds))

        for step, ((train_img, _), (out_img, _)) in enumerate(tqdm(zip(target_train, target_out))):

            
            train_posteriors = F.softmax(target_model(train_img.detach()), dim=1)
            out_posteriors = F.softmax(target_model(out_img.detach()), dim=1)

            
            train_sort, _ = torch.sort(train_posteriors, descending=True)
            train_top_k = train_sort[:, :num_posterior].clone()

            out_sort, _ = torch.sort(out_posteriors, descending=True)
            out_top_k = out_sort[:, :num_posterior].clone()

            
            train_predictions = F.sigmoid(torch.squeeze(attack_model(train_top_k)))
            out_predictions = F.sigmoid(torch.squeeze(attack_model(out_top_k)))

            
            for i, t in enumerate(thresholds):
                
                true_positives[i] += (train_predictions >= t).sum().item()
                
                false_positives[i] += (out_predictions >= t).sum().item()
                
                false_negatives[i] += (train_predictions < t).sum().item()

                correct[i] += (train_predictions >= t).sum().item()
                correct[i] += (out_predictions < t).sum().item()
                total[i] += train_predictions.size(0) + out_predictions.size(0)

        
        for i, t in enumerate(thresholds):
            accuracy = 100 * correct[i] / total[i]

            
            if true_positives[i] + false_positives[i] != 0:
                precision = true_positives[i] / (true_positives[i] + false_positives[i])
            else:
                precision = 0
            if true_positives[i] + false_negatives[i] != 0:
                recall = true_positives[i] / (true_positives[i] + false_negatives[i])
            else:
                recall = 0
            accuracies.append(accuracy)
            precisions.append(precision)
            recalls.append(recall)

            print(
                "threshold = %.4f, accuracy = %.2f, precision = %.2f, recall = %.2f" % (t, accuracy, precision, recall))

        return max(accuracies)
