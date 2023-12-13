import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import os


def main():
    #starttodo
    # load the "results.csv" file from the mia-results directory
    # read the data into a list
    # plot the Dice coefficients per label (i.e. white matter, gray matter, hippocampus, amygdala, thalamus)
    # in a boxplot

    work_directory = os.path.dirname(os.path.realpath(__file__)) + '/mia-result'
    def all_subdirs_of(b=work_directory):
        result = []
        for d in os.listdir(b):
            bd = os.path.join(b, d)
            if os.path.isdir(bd): result.append(bd)
        return result

    all_subdirs = all_subdirs_of()
    for subdirs in all_subdirs:
        if os.path.exists(os.path.join(subdirs, "Dice_allL.png")):
            continue

        directory = subdirs
        #foldername = "2023-12-06-09-58-32.078T1W_C_I_G_" # must be adjusted
        #directory = os.path.join('mia-result', foldername)
        for i in range(7):
            print('-' * 5, 'Testing...')

            test_loop_parameter = ""
            if i == 1:
                test_loop_parameter = "_gaussian_300"
            elif i == 2:
                test_loop_parameter = "_gaussian_1000"
            elif i == 3:
                test_loop_parameter = "_gaussian_2000"
            elif i == 4:
                test_loop_parameter = "_salt_pepper_001"
            elif i == 5:
                test_loop_parameter = "_salt_pepper_002"
            elif i == 6:
                test_loop_parameter = "_salt_pepper_005"

            filename = "results"+test_loop_parameter+".csv"
            filepath = os.path.join(directory, filename)
            data = pd.read_csv(filepath, sep=';', header=0, index_col=["SUBJECT"])

            # Plot Parameter
            diceName = 'DICE'
            diceYlim = [0, 1]
            HDRFDSTName = 'HDRFDST'
            HDRFDSTYlim = [0, 20] # upper limit probably must be adjusted
            accuracyName = 'ACURCY'
            accuracyYlim = [0, 1]
            subjectName = 'Subject'
            labelName = 'Label'

            # DICE, HDRFDST, ACCURACY per subject (over all labels)
            Dice_allS = data.boxplot(by='SUBJECT', column=[diceName], rot=45, grid=False).get_figure()
            plt.xlabel(subjectName)
            plt.ylabel(diceName)
            plt.ylim(diceYlim)
            plt.close()
            HDRFDST_allS = data.boxplot(by='SUBJECT', column=[HDRFDSTName], rot=45, grid=False).get_figure()
            plt.xlabel(subjectName)
            plt.ylabel(HDRFDSTName)
            plt.ylim(HDRFDSTYlim)
            plt.close()
            try:
                Accuracy_allS = data.boxplot(by='SUBJECT', column=[accuracyName], rot=45, grid=False).get_figure()
                plt.xlabel(subjectName)
                plt.ylabel(accuracyName)
                plt.ylim(accuracyYlim)
                plt.close()
            except Exception as e:
                print(f"An error occurred: {e}")

            # DICE, HDRFDST, ACCURACY per label (over all subjects)
            Dice_allL = data.boxplot(by='LABEL', column=[diceName], grid=False).get_figure()
            plt.xlabel(labelName)
            plt.ylabel(diceName)
            plt.ylim(diceYlim)
            plt.close()
            HDRFDST_allL = data.boxplot(by='LABEL', column=[HDRFDSTName], grid=False).get_figure()
            plt.xlabel(labelName)
            plt.ylabel(HDRFDSTName)
            plt.ylim(HDRFDSTYlim)
            plt.close()
            try:
                Accuracy_allL = data.boxplot(by='LABEL', column=[accuracyName], grid=False).get_figure()
                plt.xlabel(labelName)
                plt.ylabel(accuracyName)
                plt.ylim(accuracyYlim)
                plt.close()
            except Exception as e:
                print(f"An error occurred: {e}")

            # save figures
            Dice_allS.tight_layout()
            Dice_allS.savefig(os.path.join(directory, 'Dice_allS'+test_loop_parameter+'.png'), dpi=600)
            HDRFDST_allS.tight_layout()
            HDRFDST_allS.savefig(os.path.join(directory, 'HDRFDST_allS'+test_loop_parameter+'.png'), dpi=600)
            try:
                Accuracy_allS.tight_layout()
                Accuracy_allS.savefig(os.path.join(directory, 'Accuracy_allS'+test_loop_parameter+'.png'), dpi=600)
            except Exception as e:
                print(f"An error occurred: {e}")
            Dice_allL.tight_layout()
            Dice_allL.savefig(os.path.join(directory, 'Dice_allL'+test_loop_parameter+'.png'), dpi=600)
            HDRFDST_allL.tight_layout()
            HDRFDST_allL.savefig(os.path.join(directory, 'HDRFDST_allL'+test_loop_parameter+'.png'), dpi=600)
            try:
                Accuracy_allL.tight_layout()
                Accuracy_allL.savefig(os.path.join(directory, 'Accuracy_allL'+test_loop_parameter+'.png'), dpi=600)
            except Exception as e:
                print(f"An error occurred: {e}")
    #plt.show()

    # alternative: instead of manually loading/reading the csv file you could also use the pandas package
    # but you will need to install it first ('pip install pandas') and import it to this file ('import pandas as pd')
    # pass  # pass is just a placeholder if there is no other code

    #endtodo

if __name__ == '__main__':
    main()
