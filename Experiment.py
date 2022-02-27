import os
import random
from statistics import mean

from matplotlib import pyplot as plt
import edge_detection as e

def runTest(folder,data):
    #Get the number of files in each folder
    l1 = os.listdir("res/test/00021510")
    l2 = os.listdir("res/test/00023966")
    number_files_l1 = len(l1)
    number_files_l2 = len(l2)
    # print(number_files_l1)
    # print(number_files_l2)
    #pack images for edge detection

    if folder == 1:
        # get random num bounded by num files  1
        index = random.randrange(0, number_files_l1, 1)
        print(index)
        master = data[index][0]
    if folder == 2:
        # get random num bounded by num files  2
        index = random.randrange(number_files_l1, number_files_l2+number_files_l1, 1)
        print(index)
        master = data[index][0]

    master = e.process_image(master)
    results = []
    for i in range(len(data)):
        img = data[i][0]
        img = e.process_image(img)
        result = e.match_images(master, img)
        results.append((result, i))
    accuracy = 0
    if folder == 1:
        results = sorted(results, key=lambda tup: tup[0], reverse=True)
        results = results[0:number_files_l1]
        results = [x for x in results if x[1] < number_files_l1]
        accuracy = len(results)/number_files_l1
    elif folder == 2:
        results = sorted(results, key=lambda tup: tup[0], reverse=True)
        results = results[0:number_files_l2]
        results = [x for x in results if x[1] > number_files_l1]
        accuracy = len(results)/number_files_l2
    # print(results)
    # print(len(results))
    # print(accuracy)
    return accuracy


if __name__ == '__main__':
    print("Start experiment")
    folder1 = []
    folder2 = []
    n = 30
    data = e.BasicImageDataset("res/test")
    for i in range(n):
        temp_acc = runTest(1,data)
        folder1.append(temp_acc)
    for i in range(n):
        temp_acc = runTest(2,data)
        folder2.append(temp_acc)
    folder1 = sorted(folder1)
    folder2 = sorted(folder2)
    print(folder1)
    print(folder2)
    finalResult = sorted(folder1+folder2)
    mean1 = mean(finalResult)
    print("Mean accuracy: ", mean1)
    plt.plot(range(n+n), finalResult, label="Accuracy over " + str(n+n) + " runs",linewidth=5,color='green')
    plt.xlabel('Accuracy %')
    plt.ylabel('Runs')
    plt.legend()
    plt.show()