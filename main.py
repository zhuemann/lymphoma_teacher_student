# This is a sample Python script.

# Press Shift+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.

from teacher_student_train import teacher_student_train



# Press the green button in the gutter to run the script.
if __name__ == '__main__':

    local = False
    if local == True:
        directory_base = "Z:/"
    else:
        #directory_base = "/home/zmh001/r-fcb-isilon/research/Bradshaw/"
        directory_base = "/UserData/"

    seeds = [1]

    for seed in seeds:
        acc, matrix = teacher_student_train(seed=seed, batch_size=3, epoch=100, dir_base=directory_base, n_classes=5)

# See PyCharm help at https://www.jetbrains.com/help/pycharm/
