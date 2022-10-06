import os.path
from sklearn.utils import shuffle
import torch
from sklearn.metrics import classification_report
from torch.utils.data import Dataset, DataLoader,TensorDataset,random_split,SubsetRandomSampler, ConcatDataset
from torchvision import transforms
import torch.nn.init
import pandas as pd
from torch import nn
import numpy as np
import torch.optim as optim
import torch.nn.init
from sklearn.model_selection import KFold
import copy
from utils import *
from models import *
from dataloader import *


def train_model_binary_classification(model, train_dataloaders, val_dataloaders, criterion, optimizer, num_epochs=5):
    best_acc = 0.0
    best_loss = 0.0
    best_f1 = 0.0
    best_precision = 0.0
    best_recall = 0.0
    best_epoch_num = 0

    for epoch in range(num_epochs):
        print('Fold {} Epoch {}/{}'.format(fold+1, epoch+1, num_epochs))
        print('-' * 10)

        ##################### train ##########################
        model.train()
        running_loss = 0.0
        running_corrects = 0

        for i, (_, input_ids, attention_masks, image, labels) in enumerate(train_dataloaders):
            input_ids = input_ids.to(device)
            attention_masks = attention_masks.to(device)
            labels = labels.to(device)
            image = image.to(device)

            logits = model(input_ids, attention_masks, image)
            loss = criterion(logits, labels)
            outputs = torch.sigmoid(logits)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            preds = outputs.reshape(-1).round()
            running_loss += loss.item() * input_ids.size(0)
            running_corrects += torch.sum(preds == labels.reshape(-1))

        epoch_loss = running_loss / len(train_dataset)
        epoch_acc = running_corrects.double() / len(train_dataset)
        print('train loss: {:.4f}, acc: {:.4f}'.format(epoch_loss, epoch_acc))

        ##################### validation ##########################
        model.eval()
        running_loss = 0.0
        running_corrects = 0
        predicted_labels = []
        predicted_probs = []
        predicted_text_ids = []
        gold_labels = []

        with torch.no_grad():
            for i, (text_ids, input_ids, attention_masks, image, labels) in enumerate(val_dataloaders):
                input_ids = input_ids.to(device)
                attention_masks = attention_masks.to(device)
                labels = labels.to(device)
                image = image.to(device)

                logits = model(input_ids, attention_masks, image)
                loss = criterion(logits, labels)
                outputs = torch.sigmoid(logits)

                preds = outputs.reshape(-1).round()

                running_loss += loss.item() * input_ids.size(0)
                running_corrects += torch.sum(preds == labels.reshape(-1))

                predicted_text_ids += list(text_ids)
                predicted_labels += preds.detach().cpu().tolist()
                predicted_probs += outputs.reshape(-1).detach().cpu().tolist()
                gold_labels += labels.reshape(-1).detach().cpu().tolist()

        epoch_loss = running_loss / len(val_dataset)
        # epoch_acc = running_corrects.double() / len(val_dataset)

        epoch_metrics = classification_report(gold_labels, predicted_labels, output_dict=True, digits=4)
        epoch_f1 = epoch_metrics["1.0"]['f1-score']
        epoch_precision = epoch_metrics["1.0"]['precision']
        epoch_recall = epoch_metrics["1.0"]['recall']
        epoch_acc = epoch_metrics["accuracy"]

        is_best_epoch = False
        if  best_f1 <= epoch_f1:
            best_f1 = epoch_f1
            best_acc = epoch_acc
            best_loss = epoch_loss
            best_precision = epoch_precision
            best_recall = epoch_recall
            best_epoch_num = epoch

            is_best_epoch = True
            predict_df = pd.DataFrame({"ids":predicted_text_ids, "gold_labels":gold_labels, "predicted_labels":predicted_labels, "probabilities": predicted_probs})
            predict_df.to_csv(os.path.join(args.exp_dir, f"fold_{fold}_results.csv"), index=False)
        if args.save_checkpoint == 1:
            checkpoint_name = os.path.join(args.exp_dir, f'fold_{fold}_model_epoch_{epoch+1}.pth.tar')
            save_checkpoint(args, {
                'epoch': epoch + 1,
                'state_dict': model.state_dict(),
                'best_f1': best_f1,
                'optimizer': optimizer.state_dict(),
            }, fold=fold, filename=checkpoint_name, is_best=is_best_epoch, save_best_only=True)

        print('val loss: {:.4f}, acc: {:.4f}, f1: {:.4f}, precision: {:.4f}, recall: {:.4f}'.format(epoch_loss, epoch_acc, epoch_f1, epoch_precision, epoch_recall))
        print('best loss: {:.4f}, acc: {:.4f}, f1: {:.4f}, precision: {:.4f}, recall: {:.4f}, epoch {}'.format(best_loss, best_acc, best_f1, best_precision, best_recall, best_epoch_num+1))
        print(classification_report(gold_labels, predicted_labels, digits=4))

    return best_f1, best_precision, best_recall, best_acc

def train_model_multi_classification(model, train_dataloaders, val_dataloaders, criterion, optimizer, num_epochs=5):
    best_acc = 0.0
    best_loss = 0.0
    best_f1 = 0.0
    best_precision = 0.0
    best_recall = 0.0
    best_epoch_num = 0

    for epoch in range(num_epochs):
        print('Fold {} Epoch {}/{}'.format(fold+1, epoch+1, num_epochs))
        print('-' * 10)

        ##################### train ##########################
        model.train()
        running_loss = 0.0
        running_corrects = 0

        for i, (_, input_ids, attention_masks, image, labels) in enumerate(train_dataloaders):
            input_ids = input_ids.to(device)
            attention_masks = attention_masks.to(device)
            labels = labels.to(device)
            image = image.to(device)

            logits = model(input_ids, attention_masks, image)
            loss = criterion(logits, labels)
            outputs = torch.softmax(logits, dim=-1)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            _, preds = torch.max(outputs, 1)
            running_loss += loss.item() * input_ids.size(0)
            running_corrects += torch.sum(preds == labels.data)

        epoch_loss = running_loss / len(train_dataset)
        epoch_acc = running_corrects.double() / len(train_dataset)
        print('train loss: {:.4f}, acc: {:.4f}'.format(epoch_loss, epoch_acc))

        ##################### validation ##########################
        model.eval()
        running_loss = 0.0
        running_corrects = 0
        predicted_labels = []
        predicted_probs = []
        predicted_text_ids = []
        gold_labels = []

        with torch.no_grad():
            for i, (text_ids, input_ids, attention_masks, image, labels) in enumerate(val_dataloaders):
                input_ids = input_ids.to(device)
                attention_masks = attention_masks.to(device)
                labels = labels.to(device)
                image = image.to(device)

                logits = model(input_ids, attention_masks, image)
                loss = criterion(logits, labels)
                outputs = torch.softmax(logits, dim=-1)

                _, preds = torch.max(outputs, 1)

                running_loss += loss.item() * input_ids.size(0)
                running_corrects += torch.sum(preds == labels.data)

                predicted_text_ids += list(text_ids)
                predicted_labels += preds.detach().cpu().tolist()
                predicted_probs += outputs.detach().cpu().tolist()
                gold_labels += labels.detach().cpu().tolist()

        epoch_loss = running_loss / len(val_dataset)
        # epoch_acc = running_corrects.double() / len(val_dataset)

        epoch_metrics = classification_report(gold_labels, predicted_labels, output_dict=True, digits=4)
        epoch_f1 = epoch_metrics["macro avg"]['f1-score']
        epoch_precision = epoch_metrics["macro avg"]['precision']
        epoch_recall = epoch_metrics["macro avg"]['recall']
        epoch_acc = epoch_metrics["accuracy"]

        is_best_epoch = False
        if  best_f1 <= epoch_f1:
            best_f1 = epoch_f1
            best_acc = epoch_acc
            best_loss = epoch_loss
            best_precision = epoch_precision
            best_recall = epoch_recall
            best_epoch_num = epoch

            is_best_epoch = True
            predict_df = pd.DataFrame({"ids":predicted_text_ids, "gold_labels":gold_labels, "predicted_labels":predicted_labels, "probabilities": predicted_probs})
            predict_df.to_csv(os.path.join(args.exp_dir, f"fold_{fold}_results.csv"), index=False)
        if args.save_checkpoint == 1:
            checkpoint_name = os.path.join(args.exp_dir, f'fold_{fold}_model_epoch_{epoch+1}.pth.tar')
            save_checkpoint(args, {
                'epoch': epoch + 1,
                'state_dict': model.state_dict(),
                'best_f1': best_f1,
                'optimizer': optimizer.state_dict(),
            }, fold=fold, filename=checkpoint_name, is_best=is_best_epoch, save_best_only=True)

        print('val loss: {:.4f}, acc: {:.4f}, f1: {:.4f}, precision: {:.4f}, recall: {:.4f}'.format(epoch_loss, epoch_acc, epoch_f1, epoch_precision, epoch_recall))
        print('best loss: {:.4f}, acc: {:.4f}, f1: {:.4f}, precision: {:.4f}, recall: {:.4f}, epoch {}'.format(best_loss, best_acc, best_f1, best_precision, best_recall, best_epoch_num+1))
        print(classification_report(gold_labels, predicted_labels, digits=4))

    return best_f1, best_precision, best_recall, best_acc


def train_model_multi_label_classification(model, train_dataloaders, val_dataloaders, criterion, optimizer, num_epochs=5):
    best_acc = 0.0
    best_loss = 0.0
    best_f1 = 0.0
    best_precision = 0.0
    best_recall = 0.0
    best_epoch_num = 0

    for epoch in range(num_epochs):
        print('Fold {} Epoch {}/{}'.format(fold+1, epoch+1, num_epochs))
        print('-' * 10)

        ##################### train ##########################
        model.train()
        running_loss = 0.0
        running_corrects = 0

        for i, (_, input_ids, attention_masks, image, labels) in enumerate(train_dataloaders):
            input_ids = input_ids.to(device)
            attention_masks = attention_masks.to(device)
            labels = labels.to(device)
            image = image.to(device)

            logits = model(input_ids, attention_masks, image)
            loss = criterion(logits, labels)
            outputs = torch.sigmoid(logits)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            preds = outputs.round()
            running_loss += loss.item() * input_ids.size(0)
            running_corrects += torch.sum(preds == labels)

        epoch_loss = running_loss / len(train_dataset)
        epoch_acc = running_corrects.double() / len(train_dataset) / 3 # one hot is 3 dim
        print('train loss: {:.4f}, acc: {:.4f}'.format(epoch_loss, epoch_acc))

        ##################### validation ##########################
        model.eval()
        running_loss = 0.0
        running_corrects = 0
        predicted_labels = []
        predicted_probs = []
        predicted_text_ids = []
        gold_labels = []

        with torch.no_grad():
            for i, (text_ids, input_ids, attention_masks, image, labels) in enumerate(val_dataloaders):
                input_ids = input_ids.to(device)
                attention_masks = attention_masks.to(device)
                labels = labels.to(device)
                image = image.to(device)

                logits = model(input_ids, attention_masks, image)
                loss = criterion(logits, labels)
                outputs = torch.sigmoid(logits)

                preds = outputs.round()

                running_loss += loss.item() * input_ids.size(0)
                running_corrects += torch.sum(preds == labels)

                predicted_text_ids += list(text_ids)
                predicted_labels += preds.detach().cpu().tolist()
                predicted_probs += outputs.detach().cpu().tolist()
                gold_labels += labels.detach().cpu().tolist()

        epoch_loss = running_loss / len(val_dataset)
        epoch_acc = running_corrects.double() / len(val_dataset) / 3 # one hot is 3 dim

        epoch_metrics = classification_report(gold_labels, predicted_labels, output_dict=True, digits=4)
        epoch_f1 = epoch_metrics["macro avg"]['f1-score']
        epoch_precision = epoch_metrics["macro avg"]['precision']
        epoch_recall = epoch_metrics["macro avg"]['recall']

        is_best_epoch = False
        if  best_f1 <= epoch_f1:
            best_f1 = epoch_f1
            best_acc = epoch_acc
            best_loss = epoch_loss
            best_precision = epoch_precision
            best_recall = epoch_recall
            best_epoch_num = epoch

            is_best_epoch = True
            predict_df = pd.DataFrame({"ids":predicted_text_ids, "gold_labels":gold_labels, "predicted_labels":predicted_labels, "probabilities": predicted_probs})
            predict_df.to_csv(os.path.join(args.exp_dir, f"fold_{fold}_results.csv"), index=False)
        if args.save_checkpoint == 1:
            checkpoint_name = os.path.join(args.exp_dir, f'fold_{fold}_model_epoch_{epoch+1}.pth.tar')
            save_checkpoint(args, {
                'epoch': epoch + 1,
                'state_dict': model.state_dict(),
                'best_f1': best_f1,
                'optimizer': optimizer.state_dict(),
            }, fold=fold, filename=checkpoint_name, is_best=is_best_epoch, save_best_only=True)

        print('val loss: {:.4f}, acc: {:.4f}, f1: {:.4f}, precision: {:.4f}, recall: {:.4f}'.format(epoch_loss, epoch_acc, epoch_f1, epoch_precision, epoch_recall))
        print('best loss: {:.4f}, acc: {:.4f}, f1: {:.4f}, precision: {:.4f}, recall: {:.4f}, epoch {}'.format(best_loss, best_acc, best_f1, best_precision, best_recall, best_epoch_num+1))
        print(classification_report(gold_labels, predicted_labels, digits=4))

    return best_f1, best_precision, best_recall, 0


if __name__ == '__main__':

    train_transform = transforms.Compose([
        transforms.Resize((224,224)),
        # transforms.RandomResizedCrop((224, 224)),
        # transforms.RandomAffine(0, shear=10, scale=(0.8,1.2)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
    ])
    val_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
    ])

    args = get_argparser().parse_args()
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(args.seed)

    # create experiment dirs
    exp_name = get_exp_name(args)
    args.exp_dir = f"./experiments/{exp_name}"
    make_dir(args.exp_dir)
    sys.stdout = Logger(os.path.join(args.exp_dir, "train.log"), sys.stdout)
    sys.stderr = Logger(os.path.join(args.exp_dir, "error.log"), sys.stderr)

    # initial model and optimizer
    if args.exp_mode == 1:
        # multi-class classification
        if args.img_model == 0:
            init_model = MultiModelResnet50(out_dim=6)
        elif args.img_model == 1:
            init_model = MultiModelResnet101(out_dim=6)
        else:
            init_model = MultiModelVGG16(out_dim=6)
        criterion = nn.CrossEntropyLoss()
    elif args.exp_mode == 3:
        # multi-label classification
        if args.img_model == 0:
            init_model = MultiModelResnet50(out_dim=3)
        elif args.img_model == 1:
            init_model = MultiModelResnet101(out_dim=3)
        else:
            init_model = MultiModelVGG16(out_dim=3)
        criterion = nn.BCEWithLogitsLoss()
    else:
        # binary classification
        if args.img_model == 0:
            init_model = MultiModelResnet50(out_dim=1)
        elif args.img_model == 1:
            init_model = MultiModelResnet101(out_dim=1)
        else:
            init_model = MultiModelVGG16(out_dim=1)
        criterion = nn.BCEWithLogitsLoss()

    init_optimizer = optim.Adam(init_model.parameters(), lr=args.lr)

    # results
    f1_list = []
    precision_list = []
    recall_list = []
    acc_list = []

    df = pd.read_csv(os.path.join(args.data_dir, 'gun_control_annotation.csv'), index_col=0)
    if args.exp_mode == 3 or args.exp_mode == 4 or args.exp_mode == 5 or args.exp_mode == 6:
        df = df[df["persuasion_mode"].apply(lambda x: len(str(x)) > 4)]

    df = shuffle(df, random_state=args.seed)

    kfold = KFold(n_splits=args.kfold, shuffle=True, random_state=args.seed)
    for fold, (train_idx, val_idx) in enumerate(kfold.split(df)):
        print('Running fold {}...'.format(fold + 1))

        train_annotation = df.iloc[train_idx].reset_index()
        val_annotation = df.iloc[val_idx].reset_index()
        train_dataset = ImageTextDataset(args, annotation=train_annotation, root_dir=os.path.join(args.data_dir, 'images'), transform=train_transform)
        val_dataset = ImageTextDataset(args, annotation=val_annotation, root_dir=os.path.join(args.data_dir, 'images'), transform=val_transform)
        train_dataloaders = DataLoader(train_dataset, collate_fn=collate_fn, batch_size=args.batch_size)
        val_dataloaders = DataLoader(val_dataset, collate_fn=collate_fn, batch_size=args.batch_size)

        model = copy.deepcopy(init_model)
        model.to(device)
        optimizer = copy.deepcopy(init_optimizer)

        if args.exp_mode == 1:
            fold_f1, fold_precision, fold_recall, fold_acc = train_model_multi_classification(model, train_dataloaders, val_dataloaders, criterion, optimizer, args.num_epochs)
        elif args.exp_mode == 3:
            fold_f1, fold_precision, fold_recall, fold_acc = train_model_multi_label_classification(model, train_dataloaders, val_dataloaders, criterion, optimizer, args.num_epochs)
        else:
            fold_f1, fold_precision, fold_recall, fold_acc = train_model_binary_classification(model, train_dataloaders, val_dataloaders, criterion, optimizer, args.num_epochs)

        f1_list.append(fold_f1)
        precision_list.append(fold_precision)
        recall_list.append(fold_recall)
        acc_list.append(fold_acc)

    m_f1, m_precision, m_recall, m_acc = np.round(np.mean(f1_list),4), np.round(np.mean(precision_list),4), np.round(np.mean(recall_list),4), np.round(np.mean(acc_list),4)
    print(f"{exp_name} {args.kfold} fold validation...")
    print(f"{'f1' : <10}{'precision' : <10}{'recall' : <10}{'acc' : <10}")
    print(f"{m_f1 : <10}{m_precision : <10}{m_recall : <10}{m_acc : <10}")

    with open(os.path.join(args.exp_dir, "report.txt"), "w") as f:
        f.write(f"{exp_name} {args.kfold} fold validation...\n")
        f.write(f"{'f1' : <10}{'precision' : <10}{'recall' : <10}{'acc' : <10}\n")
        f.write(f"{m_f1 : <10}{m_precision : <10}{m_recall : <10}{m_acc : <10}\n")