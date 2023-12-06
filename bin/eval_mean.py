import os
import pandas as pd
import matplotlib.pyplot as plt

def main():
    # evaluation of different randomforest parameters --> later used as baseline (before adding features)
    # as well as comparison of results for different feature extractions
    # --> calculates the mean of the evaluation metric over all subjects and labels

    foldername = os.path.dirname(os.path.realpath(__file__)) + '/mia-result'
    #foldername = 'mia-result'
    filename = 'results.csv'
    dataLS = pd.DataFrame() # holds eval value for individual subjects and labels
    dataS = pd.DataFrame() # holds eval value for individual subjects (label mean value)

    # concat of all results into two dataframes (once all subject and label / once all subject with mean over all label)
    for subdir, dirs, files in os.walk(foldername):
        for dir in dirs:
            # individual subjects and labels
            filepath = os.path.join(foldername, dir, filename)
            dfLS = pd.read_csv(filepath, sep=';', header=0)
            dfLS['FOLDER'] = [dir] * (dfLS.shape[0])
            dataLS = pd.concat([dataLS, dfLS])

            # individual subjects (label mean value)
            subjects = dfLS['SUBJECT'].unique()
            for subject in subjects:
                dfsubject = dfLS[dfLS['SUBJECT'] == subject]
                meanDICE = dfsubject['DICE'].mean()
                meanHDRFDST = dfsubject['HDRFDST'].mean()
                meanACCURACY = dfsubject['ACURCY'].mean()
                dfS = pd.DataFrame([[subject, meanDICE, meanHDRFDST, meanACCURACY, dir]], columns=['SUBJECT', 'DICE', 'HDRFDST', 'ACURCY', 'FOLDER'])
                dataS = pd.concat([dataS, dfS])


    dataLS = dataLS.set_index('FOLDER')
    dataS = dataS.set_index('FOLDER')

    # Plot Parameter
    diceName = 'DICE'
    diceYlimLS = [0, 1]
    diceYlimS = [0.5, 0.75]
    HDRFDSTName = 'HDRFDST'
    HDRFDSTYlimLS = [0, 20]
    HDRFDSTYlimS = [5, 10]
    ACCURACYName = 'ACURCY'
    ACCURACYYlimLS = [0, 1]
    ACCURACYYlimS = [0, 1]
    folderName = 'Folder'

    # dice and hausdorffdistance per folder (over all subject & labels)
    Dice_meanLS = dataLS.boxplot(by='FOLDER', column=['DICE'], rot=0, grid=False).get_figure()
    plt.xlabel(folderName)
    plt.ylabel(diceName)
    plt.ylim(diceYlimLS)
    HDRFDST_meanLS = dataLS.boxplot(by='FOLDER', column=['HDRFDST'], rot=0, grid=False).get_figure()
    plt.xlabel(folderName)
    plt.ylabel(HDRFDSTName)
    plt.ylim(HDRFDSTYlimLS)
    ACCURACY_meanLS = dataLS.boxplot(by='FOLDER', column=['ACURCY'], rot=0, grid=False).get_figure()
    plt.xlabel(folderName)
    plt.ylabel(ACCURACYName)
    plt.ylim(ACCURACYYlimLS)

    # dice and hausdorffdistance per folder (over all subject (label mean value))
    Dice_meanS = dataS.boxplot(by='FOLDER', column=['DICE'], rot=0, grid=False).get_figure()
    plt.xlabel(folderName)
    plt.ylabel(diceName)
    plt.ylim(diceYlimS)
    HDRFDST_meanS = dataS.boxplot(by='FOLDER', column=['HDRFDST'], rot=0, grid=False).get_figure()
    plt.xlabel(folderName)
    plt.ylabel(HDRFDSTName)
    plt.ylim(HDRFDSTYlimS)
    ACCURACY_meanS = dataS.boxplot(by='FOLDER', column=['ACURCY'], rot=0, grid=False).get_figure()
    plt.xlabel(folderName)
    plt.ylabel(ACCURACYName)
    plt.ylim(ACCURACYYlimS)

    # save figures
    # Dice_meanLS.set_size_inches(16, 9)
    Dice_meanLS.tight_layout()
    Dice_meanLS.savefig(os.path.join(foldername, 'Dice_meanLS.png'), dpi=600)
    # HDRFDST_meanLS.set_size_inches(16, 9)
    HDRFDST_meanLS.tight_layout()
    HDRFDST_meanLS.savefig(os.path.join(foldername, 'HDRFDST_meanLS.png'), dpi=600)
    # ACCURACY_meanLS.set_size_inches(16, 9)
    ACCURACY_meanLS.tight_layout()
    ACCURACY_meanLS.savefig(os.path.join(foldername, 'Accuracy_meanLS.png'), dpi=600)
    # Dice_meanS.set_size_inches(16, 9)
    Dice_meanS.tight_layout()
    Dice_meanS.savefig(os.path.join(foldername, 'Dice_meanS.png'), dpi=600)
    # HDRFDST_meanS.set_size_inches(16, 9)
    HDRFDST_meanS.tight_layout()
    HDRFDST_meanS.savefig(os.path.join(foldername, 'HDRFDST_meanS.png'), dpi=600)
    # ACCURACY_meanS.set_size_inches(16, 9)
    ACCURACY_meanS.tight_layout()
    ACCURACY_meanS.savefig(os.path.join(foldername, 'Accuracy_meanS.png'), dpi=600)

    #plt.show()



if __name__ == '__main__':
    main()