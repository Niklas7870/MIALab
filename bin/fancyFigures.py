import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import pandas as pd
import os

workdirectery = 'mia-result/Results_all/'
folderNames = ["00_C", "01_T1W_C_I", "02_T1W_C_G", "03_T1W_C_NH", "04_T1W_C_I_G_NH", "05_T2W_C_I", "06_T2W_C_G", "07_T2W_C_NH",
               "08_T2W_C_I_G_NH", "09_T1W_T2W_C_I_G_NH"]

fileNameNoise = ['_gaussian_300', '_gaussian_1000', '_gaussian_2000', '_gaussian_5000', '_salt_pepper_001',
                 '_salt_pepper_002', '_salt_pepper_005']

color1 = '#ED7D31'
color2 = '#70AD47'
color3 = 'DarkBlue'
colorGray = '#D9D9D9'

flierprop1 = dict(markeredgecolor=color1)
flierprop2 = dict(markeredgecolor=color2)

def plotGraphs(foldername1, foldername2):

    fileNameOrigin = 'results'

    dataT1 = pd.read_csv(workdirectery + foldername1 + '/' + fileNameOrigin + '.csv', sep=';', header=0, index_col=["SUBJECT"])
    dataT2 = pd.read_csv(workdirectery + foldername2 + '/' + fileNameOrigin + '.csv', sep=';', header=0, index_col=["SUBJECT"])

    noiseDataT1 = []
    noiseDataT2 = []

    for fileAppendix in fileNameNoise:
        noiseDataT1.append(pd.read_csv(workdirectery + foldername1 + '/' + fileNameOrigin + fileAppendix + '.csv',
                                       sep=';', header=0, index_col=["SUBJECT"]))
        noiseDataT2.append(pd.read_csv(workdirectery + foldername2 + '/' + fileNameOrigin + fileAppendix + '.csv',
                                       sep=';', header=0, index_col=["SUBJECT"]))

    # Plot Parameter
    diceName = 'DICE'
    diceYlim = [0, 1]
    hausdorffName = 'HDRFDST'
    HDRFDSTYlim = [0, 20]  # upper limit probably must be adjusted
    accuracyName = 'ACURCY'
    accuracyYlim = [0, 1]
    subjectName = 'Subject'
    labelName = 'LABEL'

    # DICE, HDRFDST, ACCURACY per subject (over all labels

    #figure, axis1 = plt.subplot()
    #axis2 = axis1.twinx()

    fig, axes = plt.subplots(nrows=2, figsize=(10, 7), gridspec_kw={'height_ratios': [4, 1], 'hspace': 0.4})
    ax2 = axes[0].twinx()

    rotation = 0

    mean = []
    labels = dataT1[labelName].unique()
    for label in labels:
        dfLabel = dataT1[dataT1[labelName] == label]
        mean.append(dfLabel[hausdorffName].mean())

    df = pd.DataFrame([[1, mean[0]], [2, mean[1]], [3, mean[2]], [4, mean[3]], [5, mean[4]]],
                      columns=[labelName, hausdorffName])
    df.plot.scatter(x=labelName, y=hausdorffName, color=color3, ax=ax2, s=5)


    # add DataFrames to subplots
    dataT1.boxplot(ax=axes[0], by=labelName, column=[diceName], rot=rotation, grid=False, color=color1, flierprops=flierprop1)
    dataT2.boxplot(ax=axes[0], by=labelName, column=[diceName], rot=rotation, grid=False, color=color2, flierprops=flierprop2)
    # dots.plot(ax = ax2, by=labelName, column=[hausdorffName])
    axes[0].set_ylim(0, 1)
    axes[0].set_ylabel(diceName)
    ax2.set_ylim(0, 20)
    ax2.set_yticks(np.arange(0, 20, 4))
    axes[0].grid(axis='y', which='both', color=colorGray, linestyle='-', linewidth=1)
    #axes[0].legend(['bla', 'blu', 'bla', 'blu', 'bla', 'blu', 'bla', 'blu'])
    patchT1 = mpatches.Patch(color=color1, label='T1w')
    patchT2 = mpatches.Patch(color=color2, label='T2w')
    patchDots = mpatches.Patch(color=color3, label='hausdorff')
    axes[0].legend(handles=[patchT1, patchT2, patchDots], loc='lower left')
    axes[0].set_title('comparison of dice and hausdorff form T1 and T2')

    totalNoiseDataT1 = noiseDataT1[0]
    totalNoiseDataT2 = noiseDataT2[0]

    bla = noiseDataT1[0].set_index(labelName).subtract(dataT1.set_index(labelName))

    for index in range(1, len(fileNameNoise)):
        totalNoiseDataT1 = pd.concat([totalNoiseDataT1, dataT1.set_index(labelName).subtract(noiseDataT1[index].set_index(labelName))])

    for index in range(1, len(fileNameNoise)):
        totalNoiseDataT2 = pd.concat([totalNoiseDataT2, dataT2.set_index(labelName).subtract(noiseDataT2[index].set_index(labelName))])

    meanNoiseT1 = []
    maxNoiseT1 = []
    minNoiseT1 = []
    labels = totalNoiseDataT1[labelName].unique()
    counter = 0
    for label in labels:
        counter += 1
        if counter >= 6:
            break
        dfLabelT1 = totalNoiseDataT1[totalNoiseDataT1[labelName] == label]
        meanNoiseT1.append(dfLabelT1[diceName].mean())
        maxNoiseT1.append(dfLabelT1[diceName].max())
        minNoiseT1.append(dfLabelT1[diceName].min())

    meanNoiseT2 = []
    maxNoiseT2 = []
    minNoiseT2 = []
    labels = totalNoiseDataT2[labelName].unique()

    counter = 0

    for label in labels:
        counter += 1
        if counter >= 6:
            break
        dfLabelT2 = totalNoiseDataT2[totalNoiseDataT2[labelName] == label]
        meanNoiseT2.append(dfLabelT2[diceName].mean())
        maxNoiseT2.append(dfLabelT2[diceName].max())
        minNoiseT2.append(dfLabelT2[diceName].min())

    maxErrorT1 = np.subtract(maxNoiseT1, meanNoiseT1)
    minErrorT1 = np.subtract(minNoiseT1, meanNoiseT1)

    maxErrorT2 = np.subtract(maxNoiseT2, meanNoiseT2)
    minErrorT2 = np.subtract(minNoiseT2, meanNoiseT2)

    elinewidth = 1
    capsize = 4
    capthick = 1
    markersize = 4

    labels = ['Amygdala', 'GreyMatter', 'Hippocampus', 'Thalamus', 'WhiteMatter']

    error = [abs(minErrorT1), abs(maxErrorT1)]
    axes[1].errorbar(labels, meanNoiseT1, yerr=error, fmt='o', elinewidth=elinewidth, capsize=capsize,
                     capthick=capthick, ms=markersize, color=color1)

    error = [abs(minErrorT2), abs(maxErrorT2)]
    axes[1].errorbar(labels, meanNoiseT2, yerr=error, fmt='o', elinewidth=elinewidth, capsize=capsize,
                     capthick=capthick, ms=markersize, color=color2)
    axes[1].set_ylim(0, 1)
    axes[1].set_yticks(np.arange(0, 1, 0.3))
    axes[1].set_xlim(-0.5, 4.5)
    axes[1].set_ylabel('relative ' + diceName)
    axes[1].set_xlabel(labelName)
    axes[1].set_title('relative dice deviation from noise test data')
    axes[1].grid(axis='y', which='both', color=colorGray, linestyle='-', linewidth=1)
    axes[1].legend(handles=[patchT1, patchT2], loc='center right', bbox_to_anchor=(1.12, 0.5))

    fig.suptitle("performance of features " + foldername1 + ' and ' + foldername2, fontsize=16)

    fig.savefig(os.path.join(workdirectery, foldername1 + '_and_' + foldername2 + '.png'), dpi=600)

    #plt.show()


def plotWeightedDice():

    fileName = 'weightedDiceScore'
    fileNameResults = 'results'

    dfWDice = pd.DataFrame()
    dfDice = pd.DataFrame()
    dataDice = pd.DataFrame()

    for folder in folderNames:
        dfBufferDice = pd.read_csv(workdirectery + folder + '/' + fileName + '.csv',
                                       sep=',', header=0, index_col=["SUBJECT"])
        dfBufferDice['FOLDER'] = [folder] * (dfBufferDice.shape[0])
        dfWDice = pd.concat([dfWDice, dfBufferDice])


        dfBuffer = pd.read_csv(workdirectery + folder + '/' + fileNameResults + '.csv',
                                   sep=';', header=0)
        dfBuffer['FOLDER'] = [folder] * (dfBuffer.shape[0])
        dfDice = pd.concat([dfWDice, dfBuffer])

        # individual subjects (label mean value)
        subjects = dfBuffer['SUBJECT'].unique()
        for subject in subjects:
            dfsubject = dfBuffer[dfBuffer['SUBJECT'] == subject]
            meanDice = dfsubject['DICE'].mean()
            dfS = pd.DataFrame([[subject, meanDice, folder]],
                               columns=['SUBJECT', 'DICE', 'FOLDER'])
            dataDice = pd.concat([dataDice, dfS])

    rotation = 30

    ax = dfWDice.boxplot(by='FOLDER', column=['DICE'], rot=rotation, grid=False, color=color1, flierprops=flierprop1)
    dataDice.boxplot(ax=ax, by='FOLDER', column=['DICE'], rot=rotation, grid=False, color=color2, flierprops=flierprop2)
    # dots.plot(ax = ax2, by=labelName, column=[hausdorffName])
    plt.ylim(0, 1)
    plt.ylabel('DICE')
    plt.grid(axis='y', which='both', color=colorGray, linestyle='-', linewidth=1)
    # axes[0].legend(['bla', 'blu', 'bla', 'blu', 'bla', 'blu', 'bla', 'blu'])
    plt.title('weighted dice Score')
    patchWD = mpatches.Patch(color=color1, label='weighted dice')
    patchD = mpatches.Patch(color=color2, label='dice')
    plt.legend(handles=[patchWD, patchD], loc='lower left')

    plt.show()

    plt.savefig(os.path.join(workdirectery, 'diceStuff.png'), dpi=600)


def main():
    #starttodo
    # load the "results.csv" file from the mia-results directory
    # read the data into a list
    # plot the Dice coefficients per label (i.e. white matter, gray matter, hippocampus, amygdala, thalamus)
    # in a boxplot


    foldername1 = "01_T1W_C_I" # must be adjusted
    foldername2 = "05_T2W_C_I" # must be adjusted

    plotGraphs(foldername1, foldername2)

    foldername1 = "02_T1W_C_G" # must be adjusted
    foldername2 = "06_T2W_C_G" # must be adjusted

    plotGraphs(foldername1, foldername2)

    foldername1 = "03_T1W_C_NH" # must be adjusted
    foldername2 = "07_T2W_C_NH" # must be adjusted

    plotGraphs(foldername1, foldername2)

    foldername1 = "04_T1W_C_I_G_NH" # must be adjusted
    foldername2 = "08_T2W_C_I_G_NH" # must be adjusted

    plotGraphs(foldername1, foldername2)

    plotWeightedDice()






    # def all_subdirs_of(b=work_directory):
    #     result = []
    #     for d in os.listdir(b):
    #         bd = os.path.join(b, d)
    #         if os.path.isdir(bd): result.append(bd)
    #     return result

    #all_subdirs = all_subdirs_of()
    # for subdirs in all_subdirs:
    #     if os.path.exists(os.path.join(subdirs, "Dice_allL.png")):
    #         continue
    #
    #     directory = subdirs
    #     #foldername = "2023-12-06-09-58-32.078T1W_C_I_G_" # must be adjusted
    #     #directory = os.path.join('mia-result', foldername)
    #
    #     test_loop_parameter = ["", "_gaussian_300", "_gaussian_1000", "_gaussian_2000", "_gaussian_5000",
    #                            "_salt_pepper_001", "_salt_pepper_002", "_salt_pepper_005"]
    #
    #     for test_str in test_loop_parameter:
    #         filename = "results" + test_str + ".csv"
    #         filepath = os.path.join(directory, filename)
    #         data = pd.read_csv(filepath, sep=';', header=0, index_col=["SUBJECT"])
    #
    #         # Plot Parameter
    #         diceName = 'DICE'
    #         diceYlim = [0, 1]
    #         HDRFDSTName = 'HDRFDST'
    #         HDRFDSTYlim = [0, 20] # upper limit probably must be adjusted
    #         accuracyName = 'ACURCY'
    #         accuracyYlim = [0, 1]
    #         subjectName = 'Subject'
    #         labelName = 'Label'
    #
    #         # DICE, HDRFDST, ACCURACY per subject (over all labels)
    #         Dice_allS = data.boxplot(by='SUBJECT', column=[diceName], rot=45, grid=False).get_figure()
    #         plt.xlabel(subjectName)
    #         plt.ylabel(diceName)
    #         plt.ylim(diceYlim)
    #         plt.close()
    #         HDRFDST_allS = data.boxplot(by='SUBJECT', column=[HDRFDSTName], rot=45, grid=False).get_figure()
    #         plt.xlabel(subjectName)
    #         plt.ylabel(HDRFDSTName)
    #         plt.ylim(HDRFDSTYlim)
    #         plt.close()
    #         Accuracy_allS = data.boxplot(by='SUBJECT', column=[accuracyName], rot=45, grid=False).get_figure()
    #         plt.xlabel(subjectName)
    #         plt.ylabel(accuracyName)
    #         plt.ylim(accuracyYlim)
    #         plt.close()
    #
    #         # DICE, HDRFDST, ACCURACY per label (over all subjects)
    #         Dice_allL = data.boxplot(by='LABEL', column=[diceName], grid=False).get_figure()
    #         plt.xlabel(labelName)
    #         plt.ylabel(diceName)
    #         plt.ylim(diceYlim)
    #         plt.close()
    #         HDRFDST_allL = data.boxplot(by='LABEL', column=[HDRFDSTName], grid=False).get_figure()
    #         plt.xlabel(labelName)
    #         plt.ylabel(HDRFDSTName)
    #         plt.ylim(HDRFDSTYlim)
    #         plt.close()
    #         Accuracy_allL = data.boxplot(by='LABEL', column=[accuracyName], grid=False).get_figure()
    #         plt.xlabel(labelName)
    #         plt.ylabel(accuracyName)
    #         plt.ylim(accuracyYlim)
    #         plt.close()
    #
    #         # save figures
    #         Dice_allS.tight_layout()
    #         Dice_allS.savefig(os.path.join(directory, 'Dice_allS' + test_str + '.png'), dpi=600)
    #         HDRFDST_allS.tight_layout()
    #         HDRFDST_allS.savefig(os.path.join(directory, 'HDRFDST_allS' + test_str + '.png'), dpi=600)
    #         Accuracy_allS.tight_layout()
    #         Accuracy_allS.savefig(os.path.join(directory, 'Accuracy_allS' + test_str + '.png'), dpi=600)
    #         Dice_allL.tight_layout()
    #         Dice_allL.savefig(os.path.join(directory, 'Dice_allL' + test_str + '.png'), dpi=600)
    #         HDRFDST_allL.tight_layout()
    #         HDRFDST_allL.savefig(os.path.join(directory, 'HDRFDST_allL' + test_str + '.png'), dpi=600)
    #         Accuracy_allL.tight_layout()
    #         Accuracy_allL.savefig(os.path.join(directory, 'Accuracy_allL' + test_str + '.png'), dpi=600)
    #
    #         filename = "weightedDiceScore" + test_str + ".csv"
    #         filepath = os.path.join(directory, filename)
    #         data = pd.read_csv(filepath, sep=',', header=0, index_col=["SUBJECT"])
    #
    #         diceName = 'weightedDice'
    #
    #         # dice and housdorffdistance per subject (over all labels)
    #         weightedDice = data.boxplot(by='SUBJECT', column=['DICE'], rot=45, grid=True).get_figure()
    #         plt.xlabel(subjectName)
    #         plt.ylabel(diceName)
    #         plt.ylim(diceYlim)
    #         weightedDice.tight_layout()
    #         weightedDice.savefig(os.path.join(directory, 'WEIGHTEDDICE_allS' + test_str + '.png'), dpi=600)

            #plt.show()

            # alternative: instead of manually loading/reading the csv file you could also use the pandas package
            # but you will need to install it first ('pip install pandas') and import it to this file ('import pandas as pd')
            # pass  # pass is just a placeholder if there is no other code

            #endtodo

if __name__ == '__main__':
    main()
