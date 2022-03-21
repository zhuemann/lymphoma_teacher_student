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
from torch.optim.lr_scheduler import MultiStepLR

from util import compute_metrics, hamming_score, loss_fn, truncate_left_text_dataset
from five_class_setup import five_class_image_text_label
from BertBase import BERTClass
from VitBase import ViTBase16
from efficientnet_pytorch import EfficientNet
from dataloader_text_image import TextImageDataset
from vision_model import Vision_Model

import ssl
ssl.SSLContext.verify_mode = ssl.VerifyMode.CERT_OPTIONAL


def teacher_student_train(seed, batch_size=8, epoch=1, dir_base="/home/zmh001/r-fcb-isilon/research/Bradshaw/",
                              n_classes=2):
    # model specific global variables
    # IMG_SIZE = 224
    IMG_SIZE = 384
    # IMG_SIZE = 600
    BATCH_SIZE = batch_size
    LR = 1e-3 #8e-5  # 1e-4 was for efficient #1e-06 #2e-6 1e-6 for transformer 1e-4 for efficientnet
    GAMMA = 0.7
    N_EPOCHS = epoch  # 8
    N_CLASS = n_classes
    seed = seed
    # TOKENIZERS_PARALLELISM=True
    ssl.SSLContext.verify_mode = ssl.VerifyMode.CERT_OPTIONAL

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
    #roberta_path = os.path.join(dir_base, 'Zach_Analysis/roberta_large/')
    # using bert for now
    roberta_path = os.path.join(dir_base, 'Zach_Analysis/models/bert/')
    #roberta_path = os.path.join(dir_base, 'Zach_Analysis/models/bio_clinical_bert/')
    #roberta_path = os.path.join(dir_base, 'Zach_Analysis/models/roberta_pretrained_v4/')

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


    # save_filepath = os.path.join(dir_base, '/UserData/Zach_Analysis/Redacted_Reports/petlymph_names.xlsx')

    # test_df.to_excel(save_filepath, index=False)
    # print("after save")

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

    training_set = TextImageDataset(train_df, tokenizer, 512, mode="train", transforms=transforms_train,
                                    dir_base=dir_base)

    # probably can delete these
    # criterion = nn.CrossEntropyLoss()
    criterion = nn.MSELoss()
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

    # training_set = TextImageDataset(train_df, tokenizer, 512, mode="train", transforms = transforms_train, dir_base = dir_base)
    training_loader = DataLoader(training_set, **train_params)


    # creates the vit model which gets passed to the multimodal model class
    # vit_model = ViTBase16(n_classes=N_CLASS, pretrained=True, dir_base=dir_base)

    latient_layer = 768

    #vis_model = EfficientNet.from_pretrained('efficientnet-b0', num_classes=1000)  # num_classes=2
    #vis_model = Vision_Model(model=vis_model, n_latient=1000, n_classes=768, pretrained=False)

    random_initialize = False
    if random_initialize:
        vis_model = EfficientNet.from_name('efficientnet-b0')
        vis_model = Vision_Model(model = vis_model, n_latient = 1000, n_classes= 768, pretrained=False)
    else:
        vis_model = EfficientNet.from_pretrained('efficientnet-b0', num_classes=1000)  # num_classes=2
        vis_model = Vision_Model(model=vis_model, n_latient=1000, n_classes=768, pretrained=False)

    # creates the language model which gets passed to the multimodal model class

    #for param in roberta_model.paramters():


    language_model = BERTClass(roberta_model, n_class=N_CLASS, n_nodes=latient_layer)
    language_path = os.path.join(dir_base, 'Zach_Analysis/models/language_teacher_model')

    language_model.load_state_dict(torch.load(language_path))

    for param in language_model.parameters():
        param.requires_grad = False

    for param in vis_model.parameters():
        param.requires_grad = True


    # creates the langauge and vision models
    language_model.to(device)

    model_obj = vis_model
    model_obj.to(device)


    # defines which optimizer is being used
    # print(model_obj.parameters)
    optimizer = torch.optim.Adam(params=model_obj.parameters(), lr=LR)
    #scheduler = MultiStepLR(optimizer, milestones=[1, 2, 3, 4, 5, 6, 7, 8, 9, 15, 20, 30], gamma=0.9)
    scheduler = MultiStepLR(optimizer, milestones=[1, 8, 9, 15, 20, 30, 50, 100, 200, 300], gamma=0.95)

    best_acc = -1
    for epoch in range(1, N_EPOCHS + 1):
        model_obj.train()
        gc.collect()

        loss_list = []

        for _, data in tqdm(enumerate(training_loader, 0)):
            ids = data['ids'].to(device, dtype=torch.long)
            mask = data['mask'].to(device, dtype=torch.long)
            token_type_ids = data['token_type_ids'].to(device, dtype=torch.long)
            targets = data['targets'].to(device, dtype=torch.long)
            images = data['images'].to(device)

            output, lang_outputs = language_model(ids, mask, token_type_ids)
            vis_outputs = model_obj(images)

            optimizer.zero_grad()
            # loss = loss_fn(outputs[:, 0], targets)
            # loss = criterion(outputs, targets)
            loss = criterion(vis_outputs, lang_outputs)

            loss_list.append(loss.cpu().detach().numpy().tolist())
            #print(loss)
            if _ % 50 == 0:
                print(f'Epoch: {epoch}, Loss:  {loss.item()}')

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        #print("list")
        #print(loss_list)
        scheduler.step()
        #test = np.asarray(loss_list)
        #print(test)
        #print(type(test))

        print(f"average loss: {np.mean(np.asarray(loss_list))}")

    save_path = os.path.join(dir_base, 'Zach_Analysis/models/teacher_student/pretrained_student_vision_model')
    # torch.save(model_obj.state_dict(), '/home/zmh001/r-fcb-isilon/research/Bradshaw/Zach_Analysis/models/vit/best_multimodal_modal')
    torch.save(model_obj.state_dict(), save_path)
