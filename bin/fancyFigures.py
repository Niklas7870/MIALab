import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import pandas as pd
import os

workdirectery = 'mia-result/'
folderNames = ["00_C", "01_T1W_C_I", "02_T1W_C_G", "03_T1W_C_NH", "04_T1W_C_I_G_NH", "05_T2W_C_I", "06_T2W_C_G",
               "07_T2W_C_NH", "08_T2W_C_I_G_NH", "09_T1W_T2W_C_I_G_NH"]

plotLabels = ["0_C", "1_T1-I", "2_T1-G", "3_T1-NH", "4_T1-I-G-NH", "5_T2-C-I", "6_T2-G", "7_T2-NH", "8_T2-I-G-NH",
                  "9_T1-T2-I-G-NH"]

fileNameNoise = ['_gaussian_300', '_gaussian_1000', '_gaussian_2000', '_gaussian_5000', '_salt_pepper_001',
                 '_salt_pepper_002', '_salt_pepper_005']

axisLabels = ['Amygdala', 'GreyMatter', 'Hippocampus', 'Thalamus', 'WhiteMatter']

color1 = '#ED7D31'
color2 = '#70AD47'
color31 = '#F4B183'
color32 = '#A9D18E'
colorGray = '#D9D9D9'

fontsizeLabels = 12

flierprop1 = dict(markeredgecolor=color1)
flierprop2 = dict(markeredgecolor=color2)

def plotGraphs(foldername1, foldername2, graphTitle, exeption):

    fileNameOrigin = 'results'

    dataT1 = pd.read_csv(workdirectery + foldername1 + '/' + fileNameOrigin + '.csv', sep=';', header=0, index_col=["SUBJECT"])
    if foldername2 != None:
        dataT2 = pd.read_csv(workdirectery + foldername2 + '/' + fileNameOrigin + '.csv', sep=';', header=0, index_col=["SUBJECT"])

    noiseDataT1 = []
    noiseDataT2 = []

    for fileAppendix in fileNameNoise:
        noiseDataT1.append(pd.read_csv(workdirectery + foldername1 + '/' + fileNameOrigin + fileAppendix + '.csv',
                                       sep=';', header=0, index_col=["SUBJECT"]))
        if foldername2 != None:
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

    fig, axes = plt.subplots(nrows=2, figsize=(10, 7), gridspec_kw={'height_ratios': [4, 1], 'hspace': 0.45})
    ax2 = axes[0].twinx()

    rotation = 0
    dotSize = 25

    mean = []
    labels = dataT1[labelName].unique()
    for label in labels:
        dfLabel = dataT1[dataT1[labelName] == label]
        mean.append(dfLabel[hausdorffName].mean())

    dfT1 = pd.DataFrame([[0.8, mean[0]], [1.8, mean[1]], [2.8, mean[2]], [3.8, mean[3]], [4.8, mean[4]]],
                      columns=[labelName, 'hausdorff distance'])
    dfT1.plot.scatter(x=labelName, y='hausdorff distance', color=color31, ax=ax2, s=dotSize, edgecolors='#7F7F7F')

    if foldername2 != None:
        mean = []
        labels = dataT2[labelName].unique()
        for label in labels:
            dfLabel = dataT2[dataT2[labelName] == label]
            mean.append(dfLabel[hausdorffName].mean())

        dfT2 = pd.DataFrame([[1.2, mean[0]], [2.2, mean[1]], [3.2, mean[2]], [4.2, mean[3]], [5.2, mean[4]]],
                            columns=[labelName, hausdorffName])
        dfT2.plot.scatter(x=labelName, y=hausdorffName, color=color32, ax=ax2, s=dotSize, edgecolors='#7F7F7F')

    positionsT1 = np.array(range(1, len(labels) + 1)) - 0.2
    positionsT2 = np.array(range(1, len(labels) + 1)) + 0.2

    widths = 0.3

    # add DataFrames to subplots
    dataT1.boxplot(ax=axes[0], by=labelName, column=[diceName], rot=rotation, grid=False, color=color1,
                   flierprops=flierprop1, positions=positionsT1, widths=widths, labels=None)
    if foldername2 != None:
        dataT2.boxplot(ax=axes[0], by=labelName, column=[diceName], rot=rotation, grid=False, color=color2,
                       flierprops=flierprop2, positions=positionsT2, widths=widths, labels=None)

    axes[0].set_ylim(0, 1)
    axes[0].set_ylabel('Dice Score', fontsize=fontsizeLabels)
    ax2.set_ylim(0, 25)
    ax2.set_yticks(np.arange(0, 25, 5))
    ax2.set_ylabel('Hausdorff Distance', fontsize=fontsizeLabels)
    axes[0].grid(axis='y', which='both', color=colorGray, linestyle='-', linewidth=1)
    label1 = 'Dice T1w'
    label2 = 'Dice T2w'
    labelDot1 = 'Hausdorff T1w'
    labelDot2 = 'Hausdorff T2w'
    if exeption == True:
        label1 = 'Dice T1w C-I-G-NH'
        label2 = 'Dice T1w T2w C-G-I-NH'
        labelDot1 = 'Hausdorff T1w C-I-G-NH'
        labelDot2 = 'Hausdorff T1w T2w C-G-I-NH'
    patchT1 = mpatches.Patch(color=color1, label=label1)
    patchT2 = mpatches.Patch(color=color2, label=label2)
    patchDotsT1 = mpatches.Patch(color=color31, label=labelDot1)
    patchDotsT2 = mpatches.Patch(color=color32, label=labelDot2)
    if foldername2 != None:
        axes[0].legend(handles=[patchT1, patchT2, patchDotsT1, patchDotsT2], loc='lower left')
    else:
        axes[0].legend(handles=[patchT1, patchDotsT2], loc='lower left')

    axes[0].set_title('Dice and Hausdorff - T1w vs T2w', fontsize=fontsizeLabels)

    if exeption == True:
        axes[0].set_title('Dice and Hausdorff', fontsize=fontsizeLabels)

    cols = dataT1.columns.difference(['LABEL'])
    totalNoiseDataT1 = noiseDataT1[0]
    totalNoiseDataT1[cols] = totalNoiseDataT1[cols].sub(dataT1[cols])
    totalNoiseDataT2 = []
    if foldername2 != None:
        totalNoiseDataT2 = noiseDataT2[0]
        totalNoiseDataT2[cols] = totalNoiseDataT2[cols].sub(dataT2[cols])

    for index in range(1, len(fileNameNoise)):
        NoiseBuffer = noiseDataT1[index]
        NoiseBuffer[cols] = NoiseBuffer[cols].sub(dataT1[cols])
        totalNoiseDataT1 = pd.concat([totalNoiseDataT1, NoiseBuffer])

    if foldername2 != None:
        for index in range(1, len(fileNameNoise)):
            NoiseBuffer = noiseDataT2[index]
            NoiseBuffer[cols] = NoiseBuffer[cols].sub(dataT2[cols])
            totalNoiseDataT2 = pd.concat([totalNoiseDataT2, NoiseBuffer])

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

    if foldername2 != None:
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

    positionPlot1 = [-0.05, 0.95, 1.95, 2.95, 3.95]
    positionPlot2 = [0.05, 1.05, 2.05, 3.05, 4.05]

    error = [abs(minErrorT1), abs(maxErrorT1)]
    axes[1].plot(axisLabels, [-1, -1, -1, -1, -1])
    axes[1].errorbar(positionPlot1, meanNoiseT1, yerr=error, fmt='o', elinewidth=elinewidth, capsize=capsize,
                     capthick=capthick, ms=markersize, color=color1)

    error = [abs(minErrorT2), abs(maxErrorT2)]
    if foldername2 != None:
        axes[1].errorbar(positionPlot2, meanNoiseT2, yerr=error, fmt='o', elinewidth=elinewidth, capsize=capsize,
                        capthick=capthick, ms=markersize, color=color2)
    axes[1].set_ylim(-0.6, 0.1)
    axes[1].set_yticks(np.arange(-0.6, 0.1, 0.2))

    axes[1].set_xlim(-0.7, 4.7)
    #axes[1].set_xticks(labels)
    axes[1].set_ylabel('Relative Dice', fontsize=fontsizeLabels)
    axes[1].set_xlabel('Label', fontsize=fontsizeLabels)
    axes[1].set_title('Relative Dice (Original vs Noisy Testing-Data) - T1w vs T2w', fontsize=fontsizeLabels)
    if exeption == True:
        axes[1].set_title('Relative Dice (Original vs Noisy Testing-Data)', fontsize=fontsizeLabels)
    axes[1].grid(axis='y', which='both', color=colorGray, linestyle='-', linewidth=1)

    axes[0].set_xlabel('Labels', fontsize=fontsizeLabels)
    axes[0].set_xticks(range(1, 6))
    axes[0].set_xticklabels(axisLabels)
    axes[0].plot(range(1, 6), [-1, -1, -1, -1, -1])

    if foldername2 != None:
        #axes[1].legend(handles=[patchT1, patchT2], loc='lower right', bbox_to_anchor=(1.05, 0.97))
        #fig.suptitle("performance of features " + foldername1 + ' and ' + foldername2, fontsize=16)
        fig.suptitle(graphTitle, fontsize=16)
        fig.savefig(os.path.join(workdirectery, foldername1 + '_and_' + foldername2 + '.png'), dpi=600)
    else:
        #axes[1].legend(handles=[patchT1], loc='lower right', bbox_to_anchor=(1.12, 0.5))
        #fig.suptitle("performance of features " + foldername1, fontsize=16)
        fig.suptitle(graphTitle, fontsize=16)
        fig.savefig(os.path.join(workdirectery, foldername1 + '.png'), dpi=600)

    if exeption == True:
        axes[0].set_ylim(0.3, 0.9)
        fig.savefig(os.path.join(workdirectery, foldername1 + '_and_' + foldername2 + '_zoomed.png'), dpi=600)


def plotWeightedDice():

    fileName = 'weightedDiceScore'
    fileNameResults = 'results'

    dfWDice = pd.DataFrame()
    dfDice = pd.DataFrame()
    dataDice = pd.DataFrame()

    for i in range(0, len(folderNames)):
        dfBufferDice = pd.read_csv(workdirectery + folderNames[i] + '/' + fileName + '.csv',
                                       sep=',', header=0, index_col=["SUBJECT"])
        dfBufferDice['FOLDER'] = [plotLabels[i]] * (dfBufferDice.shape[0])
        dfWDice = pd.concat([dfWDice, dfBufferDice])

        dfBuffer = pd.read_csv(workdirectery + folderNames[i] + '/' + fileNameResults + '.csv',
                                   sep=';', header=0)
        dfBuffer['FOLDER'] = [plotLabels[i]] * (dfBuffer.shape[0])
        #dfDice = pd.concat([dfDice, dfBuffer])

        # individual subjects (label mean value)
        subjects = dfBuffer['SUBJECT'].unique()
        for subject in subjects:
            dfsubject = dfBuffer[dfBuffer['SUBJECT'] == subject]
            meanDice = dfsubject['DICE'].mean()
            dfS = pd.DataFrame([[subject, meanDice, plotLabels[i]]],
                               columns=['SUBJECT', 'DICE', 'FOLDER'])
            dataDice = pd.concat([dataDice, dfS])

    rotation = 90

    figureSize = (12, 8)

    #positionsW = np.array(range(1, 11)) - 0.2
    #positionsNW = np.array(range(1, 11)) + 0.2

    widths = 0.5

    ax = dfWDice.boxplot(by='FOLDER', column=['DICE'], rot=rotation, grid=False, color=color1, flierprops=flierprop1,
                         widths=widths)
    dataDice.boxplot(ax=ax, by='FOLDER', column=['DICE'], rot=rotation, grid=False, color=color2, flierprops=flierprop2,
                     widths=widths)
    # dots.plot(ax = ax2, by=labelName, column=[hausdorffName])
    plt.ylim(0, 1)
    plt.ylabel('Dice Score', fontsize=fontsizeLabels)
    plt.xlabel("Feature Combinations", fontsize=fontsizeLabels)
    plt.grid(axis='y', which='both', color=colorGray, linestyle='-', linewidth=1)
    plt.title('', fontsize=fontsizeLabels)
    patchWD = mpatches.Patch(color=color1, label='Weighted-Dice')
    patchD = mpatches.Patch(color=color2, label='Dice')
    plt.legend(handles=[patchWD, patchD], loc='lower left')
    plt.suptitle('Dice vs Weighted-Dice', fontsize=16)
    plt.tight_layout()

    #plt.show()

    # ax1 = plt.axes()
    # x_axis = ax1.xaxis
    # ax1.set_xticks(range(1, 6))
    # ax1.set_xticklabels(axisLabels)
    # ax1.plot(range(1, 11), np.ones(10)*(-1))

    plt.savefig(os.path.join(workdirectery, 'diceStuff.png'), dpi=600)

    plt.ylim(0.4, 0.75)

    plt.savefig(os.path.join(workdirectery, 'diceStuffZoomed.png'), dpi=600)

def plotNoiseData():

    fileNameResults = 'results'

    folderNameNoiseTraining = ["10_T1W_C_I_GA_SP", "11_T1W_C_G_GA_SP", "12_T1W_C_NH_GA_SP"]

    dfNDice = pd.DataFrame()
    dfDice = pd.DataFrame()
    dataDice = pd.DataFrame()

    for i in range(0, len(folderNameNoiseTraining)):

        dfBufferDice = pd.read_csv(workdirectery + folderNameNoiseTraining[i] + '/' + fileNameResults + '.csv',
                                       sep=';', header=0)
        dfBufferDice['FOLDER'] = [plotLabels[i+1]] * (dfBufferDice.shape[0])
        dfNDice = pd.concat([dfNDice, dfBufferDice])

        dfBuffer = pd.read_csv(workdirectery + folderNames[i+1] + '/' + fileNameResults + '.csv',
                                   sep=';', header=0)
        dfBuffer['FOLDER'] = [plotLabels[i+1]] * (dfBuffer.shape[0])
        dfDice = pd.concat([dfDice, dfBuffer])

    rotation = 90

    positions = np.array(range(1, 4)) - 0.2
    positionsN = np.array(range(1, 4)) + 0.2

    widths = 0.3

    ax = dfDice.boxplot(by='FOLDER', column=['DICE'], rot=rotation, grid=False, color=color1, flierprops=flierprop1,
                        positions=positions, widths=widths)
    dfNDice.boxplot(by='FOLDER', column=['DICE'], rot=rotation, grid=False, color=color2, flierprops=flierprop2,
                    positions=positionsN, widths=widths, ax=ax)
    # dots.plot(ax = ax2, by=labelName, column=[hausdorffName])
    plt.ylim(0, 1)
    plt.ylabel('Dice Score', fontsize=fontsizeLabels)
    plt.xlabel("Feature Combinations", fontsize=fontsizeLabels)
    plt.suptitle('Noisy Training-Data vs Original Training-Data', fontsize=16)
    plt.grid(axis='y', which='both', color=colorGray, linestyle='-', linewidth=1)
    plt.title('')
    patchWD = mpatches.Patch(color=color1, label='Original Train-Data')
    patchD = mpatches.Patch(color=color2, label='Noisy Training-Data')
    plt.legend(handles=[patchWD, patchD], loc='lower left')
    plt.tight_layout()

    #plt.show()

    plt.savefig(os.path.join(workdirectery, 'noisyStuff.png'), dpi=600)


def main():
    #starttodo
    # load the "results.csv" file from the mia-results directory
    # read the data into a list
    # plot the Dice coefficients per label (i.e. white matter, gray matter, hippocampus, amygdala, thalamus)
    # in a boxplot

    #foldername1 = "00_C"
    #foldername2 = "09_T1W_T2W_C_I_G_NH"

    #plotGraphs(foldername1, foldername2)

    # foldername1 = "04_T1W_C_I_G_NH"
    # foldername2 = "09_T1W_T2W_C_I_G_NH"
    #
    # dfBuffer = pd.read_csv(workdirectery + "04_T1W_C_I_G_NH" + '/' + 'results' + '.csv',
    #                            sep=';', header=0)
    #
    # dataDice = pd.DataFrame()
    #
    #   # individual subjects (label mean value)
    # subjects = dfBuffer['LABEL'].unique()
    # for subject in subjects:
    #     dfsubject = dfBuffer[dfBuffer['LABEL'] == subject]
    #     meanDice = dfsubject['DICE'].mean()
    #     dfS = pd.DataFrame([[subject, meanDice]],
    #                            columns=['LABEL', 'DICE'])
    #     dataDice = pd.concat([dataDice, dfS])


    foldername1 = "01_T1W_C_I" # must be adjusted
    foldername2 = "05_T2W_C_I" # must be adjusted
    graphTitle = "Segmentation Quality - Intensity-Feature"

    plotGraphs(foldername1, foldername2, graphTitle, False)

    foldername1 = "02_T1W_C_G" # must be adjusted
    foldername2 = "06_T2W_C_G" # must be adjusted
    graphTitle = "Segmentation Quality - Gradient-Feature"

    plotGraphs(foldername1, foldername2, graphTitle, False)

    foldername1 = "03_T1W_C_NH" # must be adjusted
    foldername2 = "07_T2W_C_NH" # must be adjusted
    graphTitle = "Segmentation Quality - Neighborhood-Feature"

    plotGraphs(foldername1, foldername2, graphTitle, False)

    foldername1 = "04_T1W_C_I_G_NH" # must be adjusted
    foldername2 = "08_T2W_C_I_G_NH" # must be adjusted
    graphTitle = "Segmentation Quality - Intensity-, Gradient- and Neighborhood-Feature"

    plotGraphs(foldername1, foldername2, graphTitle, False)

    #foldername1 = "09_T1W_T2W_C_I_G_NH"
    #foldername2 = None

    #plotGraphs(foldername1, foldername2)

    # foldername1 = "01_T1W_C_I"
    # foldername2 = "10_T1W_C_I_GA_SP"
    # graphTitle = "Quality of segmentation of the Intensity-Feature, trained with and without noise"
    #
    # plotGraphs(foldername1, foldername2, graphTitle, True)
    #
    # foldername1 = "02_T1W_C_G"
    # foldername2 = "11_T1W_C_G_GA_SP"
    # graphTitle = "Quality of segmentation of the Gradient-Feature, trained with and without noise"
    #
    # plotGraphs(foldername1, foldername2, graphTitle, True)
    #
    # foldername1 = "03_T1W_C_NH"
    # foldername2 = "12_T1W_C_NH_GA_SP"
    # graphTitle = "Quality of segmentation of the Neighborhood-Feature, trained with and without noise"
    #
    # plotGraphs(foldername1, foldername2, graphTitle, True)

    foldername1 = "04_T1W_C_I_G_NH"
    foldername2 = "09_T1W_T2W_C_I_G_NH"
    graphTitle = "Segmentation Quality - Top Two Results"

    plotGraphs(foldername1, foldername2, graphTitle, True)

    plotWeightedDice()

    plotNoiseData()



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
