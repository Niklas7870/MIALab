import os
import pandas as pd
import matplotlib.pyplot as plt

def main():
    # evaluation of different randomforest parameters --> later used as baseline (before adding features)
    # as well as comparison of results for different feature extractions
    # --> calculates the mean of the evaluation metric over all subjects and labels

    foldername = 'mia-result'
    filename = 'results.csv'
    data = pd.DataFrame()

    # concat of all results into one dataframe
    for subdir, dirs, files in os.walk(foldername):
        for dir in dirs:
            filepath = os.path.join(foldername, dir, filename)
            df = pd.read_csv(filepath, sep=';', header=0)
            df['FOLDER'] = [dir] * (df.shape[0])
            data = pd.concat([data, df])

    data = data.set_index('FOLDER')

    # Plot Parameter
    diceName = 'DICE'
    diceYlim = [0, 1]
    HDRFDSTName = 'HDRFDST'
    HDRFDSTYlim = [0, 50]  # upper limit probably must be adjusted
    folderName = 'Folder'

    # dice and housdorffdistance per folder (random forest parameter settings) (over all subject & labels)
    Dice_mean = data.boxplot(by='FOLDER', column=['DICE'], rot=90, grid=False).get_figure()
    plt.xlabel(folderName)
    plt.ylabel(diceName)
    plt.ylim(diceYlim)
    HDRFDST_mean = data.boxplot(by='FOLDER', column=['HDRFDST'], rot=90, grid=False).get_figure()
    plt.xlabel(folderName)
    plt.ylabel(HDRFDSTName)
    plt.ylim(HDRFDSTYlim)

    # save figures
    #Dice_mean.set_size_inches(16, 9)
    Dice_mean.tight_layout()
    Dice_mean.savefig(os.path.join(foldername, 'Dice_mean.png'), dpi=600)
    HDRFDST_mean.tight_layout()
    #HDRFDST_mean.set_size_inches(16, 9)
    HDRFDST_mean.savefig(os.path.join(foldername, 'HDRFDST_allS.png'), dpi=600)

    #plt.show()



if __name__ == '__main__':
    main()