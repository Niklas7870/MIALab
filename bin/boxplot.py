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

    directory = max(all_subdirs_of(), key=os.path.getmtime)

    #foldername = "2023-11-07-14-12-44" # must be adjusted
    #directory = os.path.join('mia-result', foldername)
    filename = "results.csv"
    filepath = os.path.join(directory, filename)
    data = pd.read_csv(filepath, sep=';', header=0, index_col=["SUBJECT"])

    # Plot Parameter
    diceName = 'DICE'
    diceYlim = [0, 1]
    HDRFDSTName = 'HDRFDST'
    HDRFDSTYlim = [0, 20] # upper limit probably must be adjusted
    subjectName = 'Subject'
    labelName = 'Label'

    # dice and housdorffdistance per subject (over all labels)
    Dice_allS = data.boxplot(by='SUBJECT', column=['DICE'], rot=45, grid=False).get_figure()
    plt.xlabel(subjectName)
    plt.ylabel(diceName)
    plt.ylim(diceYlim)
    HDRFDST_allS = data.boxplot(by='SUBJECT', column=['HDRFDST'], rot=45, grid=False).get_figure()
    plt.xlabel(subjectName)
    plt.ylabel(HDRFDSTName)
    plt.ylim(HDRFDSTYlim)

    # dice and housdorffdistance per label (over all subjects)
    Dice_allL = data.boxplot(by='LABEL', column=['DICE'], grid=False).get_figure()
    plt.xlabel(labelName)
    plt.ylabel(diceName)
    plt.ylim(diceYlim)
    HDRFDST_allL = data.boxplot(by='LABEL', column=['HDRFDST'], grid=False).get_figure()
    plt.xlabel(labelName)
    plt.ylabel(HDRFDSTName)
    plt.ylim(HDRFDSTYlim)

    # save figures
    Dice_allS.tight_layout()
    Dice_allS.savefig(os.path.join(directory, 'Dice_allS.png'), dpi=600)
    HDRFDST_allS.tight_layout()
    HDRFDST_allS.savefig(os.path.join(directory, 'HDRFDST_allS.png'), dpi=600)
    Dice_allL.tight_layout()
    Dice_allL.savefig(os.path.join(directory, 'Dice_allL.png'), dpi=600)
    HDRFDST_allL.tight_layout()
    HDRFDST_allL.savefig(os.path.join(directory, 'HDRFDST_allL.png'), dpi=600)

    #plt.show()

    # alternative: instead of manually loading/reading the csv file you could also use the pandas package
    # but you will need to install it first ('pip install pandas') and import it to this file ('import pandas as pd')
    # pass  # pass is just a placeholder if there is no other code

    #endtodo

if __name__ == '__main__':
    main()
