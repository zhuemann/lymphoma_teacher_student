import os
import torch
import torch.nn as nn
from transformers import AutoTokenizer, RobertaModel, BertModel
from torch.utils.data import Dataset, DataLoader
import gc
import numpy as np
from tqdm import tqdm
import torchvision.transforms as transforms
from sklearn import model_selection
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
import pandas as pd
from sklearn import metrics

from util import compute_metrics, hamming_score, loss_fn, truncate_left_text_dataset
from five_class_setup import five_class_image_text_label
from BertBase import BERTClass
from VitBase import ViTBase16
from efficientnet_pytorch import EfficientNet
from dataloader_text_image import TextImageDataset
from vision_model import Vision_Model

import ssl

ssl.SSLContext.verify_mode = ssl.VerifyMode.CERT_OPTIONAL

def train_vision_model(seed, batch_size=8, epoch=1, dir_base="/home/zmh001/r-fcb-isilon/research/Bradshaw/",
                              n_classes=2):
    # model specific global variables
    # IMG_SIZE = 224
    IMG_SIZE = 384
    # IMG_SIZE = 600
    BATCH_SIZE = batch_size
    LR = 8e-5 #8e-5  # 1e-4 was for efficient #1e-06 #2e-6 1e-6 for transformer 1e-4 for efficientnet
    GAMMA = 0.7
    N_EPOCHS = epoch  # 8
    N_CLASS = n_classes
    seed = seed
    # TOKENIZERS_PARALLELISM=True

    # creates the label, text, and image names in a dataframe for 2 class
    # df = get_text_id_labels(dir_base=dir_base)
    # df = df.set_index('id')

    dataframe_location = os.path.join(dir_base, 'Zach_Analysis/lymphoma_data/lymphoma_df.xlsx')
    save_labels = False

    if save_labels:
        # gets the labels and saves it off to the location

        # creates the label, text, and image names in a dataframe for 5 class
        df = five_class_image_text_label(dir_base=dir_base)
        #print(df)
        df.to_excel(dataframe_location, index=False)
    else:
        # reads in the dataframe as it doesn't really change to save time
        df = pd.read_excel(dataframe_location, engine='openpyxl')
        print(df)
        df.set_index("image_id", inplace=True)
        print(df)


    # creates the path to the roberta model used from the bradshaw drive and loads the tokenizer and roberta model
    # roberta_path = os.path.join(dir_base, 'Zach_Analysis/roberta_large/')
    # using bert for now
    roberta_path = os.path.join(dir_base, 'Zach_Analysis/models/bert/')
    # roberta_path = os.path.join(dir_base, 'Zach_Analysis/models/bio_clinical_bert/')

    tokenizer = AutoTokenizer.from_pretrained(roberta_path)
    # roberta_model = RobertaModel.from_pretrained(roberta_path)
    roberta_model = BertModel.from_pretrained(roberta_path)

    # takes just the last 512 tokens if there are more than 512 tokens in the text
    df = truncate_left_text_dataset(df, tokenizer)

    # Splits the data into 80% train and 20% valid and test sets
    train_df, test_valid_df = model_selection.train_test_split(
        df, test_size=0.2, random_state=seed, stratify=df.label.values
    )
    # Splits the test and valid sets in half so they are both 10% of total data
    test_df, valid_df = model_selection.train_test_split(
        test_valid_df, test_size=0.5, random_state=seed, stratify=test_valid_df.label.values
    )

    #print(test_df)
    #num_0_labels = (test_df[["label"]] == 0).any(axis=1)
    #num_1_labels = (test_df[["label"]] == 1).any(axis=1)
    #num_2_labels = (test_df[["label"]] == 2).any(axis=1)
    #num_3_labels = (test_df[["label"]] == 3).any(axis=1)
    #num_4_labels = (test_df[["label"]] == 4).any(axis=1)
    #print("num_1: " + str(num_0_labels.sum()))
    #print("num_2: " + str(num_1_labels.sum()))
    #print("num_3: " + str(num_2_labels.sum()))
    #print("num_4: " + str(num_3_labels.sum()))
    #print("num_5: " + str(num_4_labels.sum()))

    # save_filepath = os.path.join(dir_base, '/UserData/Zach_Analysis/Redacted_Reports/petlymph_names.xlsx')

    # test_df.to_excel(save_filepath, index=False)
    # print("after save")
    # save_filepath = os.path.join(dir_base, '/Zach_Analysis/lymphoma_data/test_vision_df.xlsx')
    # test_df.to_excel(save_filepath, index=True)

    # create image augmentations
    transforms_train = transforms.Compose(
        [
            transforms.Resize((IMG_SIZE, IMG_SIZE)),
            transforms.RandomHorizontalFlip(p=0.3),
            transforms.RandomVerticalFlip(p=0.3),
            transforms.RandomAffine(degrees=10, translate=(.1, .1), scale=None, shear=None),
            # transforms.RandomResizedCrop(IMG_SIZE),
            transforms.ToTensor(),
            transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
        ]
    )

    transforms_valid = transforms.Compose(
        [
            transforms.Resize((IMG_SIZE, IMG_SIZE)),
            transforms.ToTensor(),
            transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
        ]
    )

    # should be able to delete these don't use the MIPDataset
    # train_dataset = MIPSDataset(train_df, transforms=transforms_train)
    # valid_dataset = MIPSDataset(valid_df, transforms=transforms_valid)
    # test_dataset = MIPSDataset(test_df, transforms=transforms_valid)

    training_set = TextImageDataset(train_df, tokenizer, 512, mode="train", transforms=transforms_train,
                                    dir_base=dir_base)
    valid_set = TextImageDataset(valid_df, tokenizer, 512, transforms=transforms_valid, dir_base=dir_base)
    test_set = TextImageDataset(test_df, tokenizer, 512, transforms=transforms_valid, dir_base=dir_base)

    train_loader = torch.utils.data.DataLoader(
        dataset=training_set,
        batch_size=BATCH_SIZE,
        sampler=None,
        drop_last=True,
        num_workers=8,
    )

    valid_loader = torch.utils.data.DataLoader(
        dataset=valid_set,
        batch_size=BATCH_SIZE,
        sampler=None,
        drop_last=True,
        num_workers=8,
    )

    test_loader = torch.utils.data.DataLoader(
        dataset=test_set,
        batch_size=BATCH_SIZE,
        sampler=None,
        drop_last=True,
        num_workers=8,
    )

    # probably can delete these
    criterion = nn.CrossEntropyLoss()
    #criterion = nn.MSELoss()
    # criterion = nn.SmoothL1Loss()
    # criterion = nn.BCEWithLogitsLoss()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print("FULL Dataset: {}".format(df.shape))
    print("TRAIN Dataset: {}".format(train_df.shape))
    print("TEST Dataset: {}".format(test_df.shape))
    print("VALID Dataset: {}".format(valid_df.shape))

    train_params = {'batch_size': BATCH_SIZE,
                    'shuffle': True,
                    'num_workers': 4
                    }

    test_params = {'batch_size': BATCH_SIZE,
                   'shuffle': True,
                   'num_workers': 4
                   }


    # training_set = TextImageDataset(train_df, tokenizer, 512, mode="train", transforms = transforms_train, dir_base = dir_base)
    training_loader = DataLoader(training_set, **train_params)

    # valid_set = TextImageDataset(valid_df, tokenizer, 512, transforms = transforms_valid, dir_base = dir_base)
    valid_loader = DataLoader(valid_set, **test_params)

    # test_set = TextImageDataset(test_df, tokenizer, 512, transforms = transforms_valid, dir_base = dir_base)
    test_loader = DataLoader(test_set, **test_params)

    # creates the vit model which gets passed to the multimodal model class
    # vit_model = ViTBase16(n_classes=N_CLASS, pretrained=True, dir_base=dir_base)

    #vis_model = Vision_Model(n_classes=N_CLASS, n_latient=768, pretrained=True, dir_base=dir_base)


    vis_model = EfficientNet.from_pretrained('efficientnet-b0', num_classes=1000)
    #model_path = os.path.join(dir_base, 'Zach_Analysis/models/teacher_student/efficientnet-b0')

    #torch.save(vis_model.state_dict(), model_path)

    use_1000ep_model = False
    if use_1000ep_model:
        model_path = os.path.join(dir_base,
                                  'Zach_Analysis/models/teacher_student/pretrained_student_vision_model_1000ep')
        vis_model = EfficientNet.from_name('efficientnet-b0')
        #vis_model = Vision_Model(model=vis_model, n_latient=1000, n_classes=768, pretrained=False)
        vis_model.load_state_dict(torch.load(model_path))
        #vis_model = Vision_Model(model=vis_model, n_latient=768, n_classes=5, pretrained=False)
    else:
        #model_path = os.path.join(dir_base,
        #                          'Zach_Analysis/models/teacher_student/pretrained_student_vision_model')
        #vis_model = EfficientNet.from_name('efficientnet-b0')
        vis_model = Vision_Model(model=vis_model, n_latient=1000, n_classes=768, pretrained=False)
        #vis_model.load_state_dict(torch.load(model_path))
        vis_model = Vision_Model(model=vis_model, n_latient=768, n_classes=5, pretrained=False)

    #model.fc = nn.Linear(1000, 5)
    #vis_model = model
    #vis_model = model.load_state_dict(torch.load(model_path))

    #if dir_base == "Z:/":
    #vis_model = EfficientNet.from_pretrained('efficientnet-b0', num_classes=5)  # num_classes=2
    #else:
    #vis_model = Vision_Model(n_classes=N_CLASS, pretrained=True, dir_base=dir_base)


    # creates the language model which gets passed to the multimodal model class
    #language_model = BERTClass(roberta_model, n_class=N_CLASS, n_nodes=1024)

    #for param in language_model.parameters():
    #    param.requires_grad = False

    for index, param in enumerate(vis_model.parameters()):
        print(param.size())
        param.requires_grad = True
        #print(index)
        if index < 3:
            param.require_grad = True

    for name, child in vis_model.named_children():
        for x, y in child.named_children():
            print(name, x)

    print(vis_model)
    #print(vis_model._blocks)
    print("break")
    #for name, layer in enumerate(vis_model):
    #    if

    # creates the multimodal modal from the langauge and vision model and moves it to device
    #model_obj = MyEnsemble(language_model, vit_model, n_classes=N_CLASS)

    #language_model.to(device)

    model_obj = vis_model
    model_obj.to(device)



    # defines which optimizer is being used
    # print(model_obj.parameters)
    optimizer = torch.optim.Adam(params=model_obj.parameters(), lr=LR)

    best_acc = -1
    for epoch in range(1, N_EPOCHS + 1):
        model_obj.train()
        gc.collect()
        fin_targets = []
        fin_outputs = []
        confusion_matrix = [[0, 0, 0, 0, 0], [0, 0, 0, 0, 0], [0, 0, 0, 0, 0], [0, 0, 0, 0, 0], [0, 0, 0, 0, 0]]

        #if epoch > 25:
        #    for param in model_obj.parameters():
        #        param.requires_grad = True
        #    for learning_rate in optimizer.param_groups:
        #        learning_rate['lr'] = 5e-6  # 1e-6 for roberta

        for _, data in tqdm(enumerate(training_loader, 0)):
            ids = data['ids'].to(device, dtype=torch.long)
            mask = data['mask'].to(device, dtype=torch.long)
            token_type_ids = data['token_type_ids'].to(device, dtype=torch.long)
            targets = data['targets'].to(device, dtype=torch.long)
            images = data['images'].to(device)

            outputs = model_obj(images)

            fin_targets.extend(targets.cpu().detach().numpy().tolist())
            fin_outputs.extend(torch.sigmoid(outputs).cpu().detach().numpy().tolist())
            # fin_outputs.extend(outputs.cpu().detach().numpy().tolist())
            # targets = torch.nn.functional.one_hot(input = targets.long(), num_classes = n_classes)

            optimizer.zero_grad()
            # loss = loss_fn(outputs[:, 0], targets)
            loss = criterion(outputs, targets)
            if _ % 50 == 0:
                print(f'Epoch: {epoch}, Loss:  {loss.item()}')

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            #print(outputs.shape[0])
            for i in range(0, outputs.shape[0]):
                actual = targets[i].detach().cpu().data.numpy()
                predicted = outputs.argmax(dim=1)[i].detach().cpu().data.numpy()
                #print(actual)
                #print(predicted)
                confusion_matrix[predicted][actual] += 1

        # get the final score
        # if N_CLASS > 2:
        final_outputs = np.copy(fin_outputs)
        # final_outputs = np.round(final_outputs, decimals=0)
        # final_outputs = (final_outputs == final_outputs.max(axis=1)[:,None]).astype(int)
        final_outputs = np.argmax(final_outputs, axis=1)
        # else:
        #    final_outputs = np.array(fin_outputs) > 0.5

        # print(final_outputs.tolist())
        # print(fin_targets)
        accuracy = accuracy_score(np.array(fin_targets), np.array(final_outputs))
        print(f"Train Accuracy = {accuracy}")
        print(confusion_matrix)

        # each epoch, look at validation data
        model_obj.eval()
        fin_targets = []
        fin_outputs = []
        confusion_matrix = [[0, 0, 0, 0, 0], [0, 0, 0, 0, 0], [0, 0, 0, 0, 0], [0, 0, 0, 0, 0], [0, 0, 0, 0, 0]]

        with torch.no_grad():
            gc.collect()
            for _, data in tqdm(enumerate(valid_loader, 0)):
                ids = data['ids'].to(device, dtype=torch.long)
                mask = data['mask'].to(device, dtype=torch.long)
                token_type_ids = data['token_type_ids'].to(device, dtype=torch.long)
                targets = data['targets'].to(device, dtype=torch.long)
                images = data['images'].to(device)

                outputs = model_obj(images)

                fin_targets.extend(targets.cpu().detach().numpy().tolist())
                fin_outputs.extend(torch.sigmoid(outputs).cpu().detach().numpy().tolist())  # for two class
                # fin_outputs.extend(outputs.cpu().detach().numpy().tolist())

                for i in range(0, outputs.shape[0]):
                    actual = targets[i].detach().cpu().data.numpy()
                    predicted = outputs.argmax(dim=1)[i].detach().cpu().data.numpy()
                    confusion_matrix[predicted][actual] += 1

            # get the final score
            # if N_CLASS > 2:
            final_outputs = np.copy(fin_outputs)
            # final_outputs = np.round(final_outputs, decimals=0)
            final_outputs = np.argmax(final_outputs, axis=1)
            # final_outputs = (final_outputs == final_outputs.max(axis=1)[:,None]).astype(int)
            # else:
            #    final_outputs = np.array(fin_outputs) > 0.5

            # final_outputs = np.array(fin_outputs) > 0.5
            # final_outputs = np.copy(fin_outputs)
            # final_outputs = (final_outputs == final_outputs.max(axis=1)[:,None]).astype(int)
            val_hamming_loss = metrics.hamming_loss(fin_targets, final_outputs)
            val_hamming_score = hamming_score(np.array(fin_targets), np.array(final_outputs))

            accuracy = accuracy_score(np.array(fin_targets), np.array(final_outputs))
            print(f"valid Hamming Score = {val_hamming_score}\nValid Accuracy = {accuracy}")

            print(f"Epoch {str(epoch)}, Vision Validation Hamming Score = {val_hamming_score}")
            print(f"Epoch {str(epoch)}, Vision Validation Hamming Loss = {val_hamming_loss}")
            print(confusion_matrix)
            if accuracy >= best_acc:
                best_acc = accuracy
                save_path = os.path.join(dir_base, 'Zach_Analysis/models/teacher_student/best_vision_model')
                # torch.save(model_obj.state_dict(), '/home/zmh001/r-fcb-isilon/research/Bradshaw/Zach_Analysis/models/vit/best_multimodal_modal')
                torch.save(model_obj.state_dict(), save_path)

    model_obj.eval()
    fin_targets = []
    fin_outputs = []
    row_ids = []
    confusion_matrix = [[0, 0, 0, 0, 0], [0, 0, 0, 0, 0], [0, 0, 0, 0, 0], [0, 0, 0, 0, 0], [0, 0, 0, 0, 0]]
    saved_path = os.path.join(dir_base, 'Zach_Analysis/models/teacher_student/best_vision_model')
    # model_obj.load_state_dict(torch.load('/home/zmh001/r-fcb-isilon/research/Bradshaw/Zach_Analysis/models/vit/best_multimodal_modal'))
    model_obj.load_state_dict(torch.load(saved_path))

    with torch.no_grad():
        for _, data in tqdm(enumerate(test_loader, 0)):
            ids = data['ids'].to(device, dtype=torch.long)
            mask = data['mask'].to(device, dtype=torch.long)
            token_type_ids = data['token_type_ids'].to(device, dtype=torch.long)
            targets = data['targets'].to(device, dtype=torch.long)
            images = data['images'].to(device)

            outputs = model_obj(images)
            row_ids.extend(data['row_ids'])
            fin_targets.extend(targets.cpu().detach().numpy().tolist())
            fin_outputs.extend(torch.sigmoid(outputs).cpu().detach().numpy().tolist())  # for two class
            # fin_outputs.extend(outputs.cpu().detach().numpy().tolist())

            for i in range(0, outputs.shape[0]):
                actual = targets[i].detach().cpu().data.numpy()
                predicted = outputs.argmax(dim=1)[i].detach().cpu().data.numpy()
                confusion_matrix[predicted][actual] += 1

        # get the final score
        # if N_CLASS > 2:
        final_outputs = np.copy(fin_outputs)
        final_outputs = np.argmax(final_outputs, axis=1)
        # final_outputs = np.round(final_outputs, decimals=0)
        # final_outputs = (final_outputs == final_outputs.max(axis=1)[:,None]).astype(int)
        # else:
        #    final_outputs = np.array(fin_outputs) > 0.5

        test_hamming_score = hamming_score(np.array(fin_targets), np.array(final_outputs))
        accuracy = accuracy_score(np.array(fin_targets), np.array(final_outputs))
        print(f"Test Vision Hamming Score = {test_hamming_score}\nTest Vision Accuracy = {accuracy}")
        print(confusion_matrix)

        return accuracy, confusion_matrix
        # print(f"Test Hamming Score = {test_hamming_score}\nTest Accuracy = {accuracy}\n{model_type[model_selection] + save_name_extension}")

        # create a dataframe of the prediction, labels, and which ones are correct
        # if N_CLASS > 2:
        #    df_test_vals = pd.DataFrame(list(zip(row_ids, np.argmax(fin_targets, axis=1).astype(int).tolist(), np.argmax(final_outputs, axis=1).astype(int).tolist())), columns=['id', 'label', 'prediction'])
        # else:
        #    df_test_vals = pd.DataFrame(list(zip(row_ids, list(map(int, fin_targets)), final_outputs[:,0].astype(int).tolist())), columns=['id', 'label', 'prediction'])
        # df_test_vals['correct'] = df_test_vals['label'].equals(df_test_vals['prediction'])
        # df_test_vals['correct'] = np.where( df_test_vals['label'] == df_test_vals['prediction'], 1, 0)
        # df_test_vals = df_test_vals.sort_values('id')
        # df_test_vals = df_test_vals.set_index('id')