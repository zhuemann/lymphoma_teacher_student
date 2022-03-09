# This is a sample Python script.

# Press Shift+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.
import pandas as pd
import os

from teacher_student_train import teacher_student_train
from train_vision_model import train_vision_model



# Press the green button in the gutter to run the script.
if __name__ == '__main__':

    local = False
    if local == True:
        directory_base = "Z:/"
    else:
        #directory_base = "/home/zmh001/r-fcb-isilon/research/Bradshaw/"
        directory_base = "/UserData/"

    seeds = [117, 295, 714, 892, 1023, 2756, 3425]
    accuracy_list = []

    for seed in seeds:
        teacher_student_train(seed=seed, batch_size=16, epoch=1, dir_base=directory_base, n_classes=5)
        acc, matrix = train_vision_model(seed=seed, batch_size=16, epoch=1, dir_base=directory_base, n_classes=5)

        accuracy_list.append(acc)
        df = pd.DataFrame(matrix)

        ## save to xlsx file
        filepath = os.path.join(directory_base, '/UserData/Zach_Analysis/result_logs_teacher_student/bio_bert_20ep/confusion_matrix_seed' + str(seed) + '.xlsx')

        df.to_excel(filepath, index=False)

        print(accuracy_list)

# See PyCharm help at https://www.jetbrains.com/help/pycharm/
