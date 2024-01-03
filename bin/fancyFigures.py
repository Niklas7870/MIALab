import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import pandas as pd
import os


# generates plots used in final presentation and report (dice, weighted dice (boxplots), robustness)
# after training and evaluating the models the individual result folder names or the list folderNames must be adjusted
# to have the same name, otherwise this script will not work.

# path and names for data handling
workdirectery = 'mia-result/'
folderNames = ["00_C", "01_T1W_C_I", "02_T1W_C_G", "03_T1W_C_NH", "04_T1W_C_I_G_NH", "05_T2W_C_I", "06_T2W_C_G",
               "07_T2W_C_NH", "08_T2W_C_I_G_NH", "09_T1W_T2W_C_I_G_NH"]

plotLabels = ["0_C", "1_T1-I", "2_T1-G", "3_T1-NH", "4_T1-I-G-NH", "5_T2-C-I", "6_T2-G", "7_T2-NH", "8_T2-I-G-NH",
                  "9_T1-T2-I-G-NH"]

fileNameNoise = ['_gaussian_300', '_gaussian_1000', '_gaussian_2000', '_gaussian_5000', '_salt_pepper_001',
                 '_salt_pepper_002', '_salt_pepper_005']

axisLabels = ['Amygdala', 'GreyMatter', 'Hippocampus', 'Thalamus', 'WhiteMatter']

# general color codes
color1 = '#ED7D31'
color2 = '#70AD47'
color31 = '#F4B183'
color32 = '#A9D18E'
colorGray = '#D9D9D9'

# design definitions for plotting
fontsizeLabels = 12
flierprop1 = dict(markeredgecolor=color1)
flierprop2 = dict(markeredgecolor=color2)

def plotGraphs(foldername1, foldername2, graphTitle, exeption):

    fileNameOrigin = 'results'

    # read in data
    dataT1 = pd.read_csv(workdirectery + foldername1 + '/' + fileNameOrigin + '.csv', sep=';', header=0, index_col=["SUBJECT"])
    dataT2 = pd.read_csv(workdirectery + foldername2 + '/' + fileNameOrigin + '.csv', sep=';', header=0, index_col=["SUBJECT"])

    noiseDataT1 = []
    noiseDataT2 = []

    # read in noisy data
    for fileAppendix in fileNameNoise:
        noiseDataT1.append(pd.read_csv(workdirectery + foldername1 + '/' + fileNameOrigin + fileAppendix + '.csv',
                                       sep=';', header=0, index_col=["SUBJECT"]))
        noiseDataT2.append(pd.read_csv(workdirectery + foldername2 + '/' + fileNameOrigin + fileAppendix + '.csv',
                                       sep=';', header=0, index_col=["SUBJECT"]))

    # Plot Parameter
    diceName = 'DICE'
    hausdorffName = 'HDRFDST'
    labelName = 'LABEL'


    # generate the subplot conditions
    fig, axes = plt.subplots(nrows=2, figsize=(10, 7), gridspec_kw={'height_ratios': [4, 1], 'hspace': 0.45})
    ax2 = axes[0].twinx()

    rotation = 0
    dotSize = 25

    # calculate the mean hausdorff distance for T1
    mean = []
    labels = dataT1[labelName].unique()
    for label in labels:
        dfLabel = dataT1[dataT1[labelName] == label]
        mean.append(dfLabel[hausdorffName].mean())

    # plot the mean of hausdorff in the boxplot graph
    dfT1 = pd.DataFrame([[0.8, mean[0]], [1.8, mean[1]], [2.8, mean[2]], [3.8, mean[3]], [4.8, mean[4]]],
                      columns=[labelName, 'hausdorff distance'])
    dfT1.plot.scatter(x=labelName, y='hausdorff distance', color=color31, ax=ax2, s=dotSize, edgecolors='#7F7F7F')

    # calculate the mean hausdorff distance for T2
    mean = []
    labels = dataT2[labelName].unique()
    for label in labels:
        dfLabel = dataT2[dataT2[labelName] == label]
        mean.append(dfLabel[hausdorffName].mean())

    # plot the mean of hausdorff in the boxplot graph
    dfT2 = pd.DataFrame([[1.2, mean[0]], [2.2, mean[1]], [3.2, mean[2]], [4.2, mean[3]], [5.2, mean[4]]],
                        columns=[labelName, hausdorffName])
    dfT2.plot.scatter(x=labelName, y=hausdorffName, color=color32, ax=ax2, s=dotSize, edgecolors='#7F7F7F')

    # define the positions for T1 and T2 boxplots that they are not overlaying
    positionsT1 = np.array(range(1, len(labels) + 1)) - 0.2
    positionsT2 = np.array(range(1, len(labels) + 1)) + 0.2

    widths = 0.3

    # plot the boxplots for T1 and T2
    dataT1.boxplot(ax=axes[0], by=labelName, column=[diceName], rot=rotation, grid=False, color=color1,
                   flierprops=flierprop1, positions=positionsT1, widths=widths, labels=None)
    dataT2.boxplot(ax=axes[0], by=labelName, column=[diceName], rot=rotation, grid=False, color=color2,
                    flierprops=flierprop2, positions=positionsT2, widths=widths, labels=None)

    # adjust different label and design parameters
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

    # create a legend
    patchT1 = mpatches.Patch(color=color1, label=label1)
    patchT2 = mpatches.Patch(color=color2, label=label2)
    patchDotsT1 = mpatches.Patch(color=color31, label=labelDot1)
    patchDotsT2 = mpatches.Patch(color=color32, label=labelDot2)
    axes[0].legend(handles=[patchT1, patchT2, patchDotsT1, patchDotsT2], loc='lower left')
    axes[0].set_title('Dice and Hausdorff - T1w vs T2w', fontsize=fontsizeLabels)

    if exeption == True:
        axes[0].set_title('Dice and Hausdorff', fontsize=fontsizeLabels)

    # initialise variables to calculate difference between noisy data and not noisy training data
    cols = dataT1.columns.difference(['LABEL'])
    totalNoiseDataT1 = noiseDataT1[0]
    totalNoiseDataT1[cols] = totalNoiseDataT1[cols].sub(dataT1[cols])
    totalNoiseDataT2 = noiseDataT2[0]
    totalNoiseDataT2[cols] = totalNoiseDataT2[cols].sub(dataT2[cols])

    # calculate difference between noisy data and not noisy training data
    for index in range(1, len(fileNameNoise)):
        NoiseBuffer = noiseDataT1[index]
        NoiseBuffer[cols] = NoiseBuffer[cols].sub(dataT1[cols])
        totalNoiseDataT1 = pd.concat([totalNoiseDataT1, NoiseBuffer])

    for index in range(1, len(fileNameNoise)):
        NoiseBuffer = noiseDataT2[index]
        NoiseBuffer[cols] = NoiseBuffer[cols].sub(dataT2[cols])
        totalNoiseDataT2 = pd.concat([totalNoiseDataT2, NoiseBuffer])

    # calculate the mean, min and max value out of the differences for T1
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

    # calculate the mean, min and max value out of the differences for T2
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

    # calculate the relative distance between mean and min / max for the error bar plot
    maxErrorT1 = np.subtract(maxNoiseT1, meanNoiseT1)
    minErrorT1 = np.subtract(minNoiseT1, meanNoiseT1)

    maxErrorT2 = np.subtract(maxNoiseT2, meanNoiseT2)
    minErrorT2 = np.subtract(minNoiseT2, meanNoiseT2)

    # define plot parameters
    elinewidth = 1
    capsize = 4
    capthick = 1
    markersize = 4

    # define the x axis coordinates to plot
    positionPlot1 = [-0.05, 0.95, 1.95, 2.95, 3.95]
    positionPlot2 = [0.05, 1.05, 2.05, 3.05, 4.05]

    # plot the mean with error bar in the bottom plot
    error = [abs(minErrorT1), abs(maxErrorT1)] # error has to be positive
    axes[1].plot(axisLabels, [-1, -1, -1, -1, -1])
    axes[1].errorbar(positionPlot1, meanNoiseT1, yerr=error, fmt='o', elinewidth=elinewidth, capsize=capsize,
                     capthick=capthick, ms=markersize, color=color1)

    error = [abs(minErrorT2), abs(maxErrorT2)]
    axes[1].errorbar(positionPlot2, meanNoiseT2, yerr=error, fmt='o', elinewidth=elinewidth, capsize=capsize,
                    capthick=capthick, ms=markersize, color=color2)

    # adjust labeling and design for the plot
    axes[1].set_ylim(-0.6, 0.1)
    axes[1].set_yticks(np.arange(-0.6, 0.1, 0.2))
    axes[1].set_xlim(-0.7, 4.7)
    axes[1].set_ylabel('Relative Dice', fontsize=fontsizeLabels)
    axes[1].set_xlabel('Label', fontsize=fontsizeLabels)
    axes[1].set_title('Relative Dice (Original vs Noisy Testing-Data) - T1w vs T2w', fontsize=fontsizeLabels)
    if exeption == True:
        axes[1].set_title('Relative Dice (Original vs Noisy Testing-Data)', fontsize=fontsizeLabels)
    axes[1].grid(axis='y', which='both', color=colorGray, linestyle='-', linewidth=1)

    # this is a workaround to remove the double x labels form the two boxplot and make new ones which are in the middle
    # of the two box plots
    axes[0].set_xlabel('Labels', fontsize=fontsizeLabels)
    axes[0].set_xticks(range(1, 6))
    axes[0].set_xticklabels(axisLabels)
    axes[0].plot(range(1, 6), [-1, -1, -1, -1, -1])

    fig.suptitle(graphTitle, fontsize=16)
    fig.savefig(os.path.join(workdirectery, foldername1 + '_and_' + foldername2 + '.png'), dpi=600)

    # zoom in for a second image
    if exeption == True:
        axes[0].set_ylim(0.3, 0.9)
        fig.savefig(os.path.join(workdirectery, foldername1 + '_and_' + foldername2 + '_zoomed.png'), dpi=600)


def plotWeightedDice():

    # define file names
    fileName = 'weightedDiceScore'
    fileNameResults = 'results'

    dfWDice = pd.DataFrame()
    dataDice = pd.DataFrame()

    # read in all the data
    for i in range(0, len(folderNames)):
        dfBufferDice = pd.read_csv(workdirectery + folderNames[i] + '/' + fileName + '.csv',
                                       sep=',', header=0, index_col=["SUBJECT"])
        # add a column with the folder to separate the date later
        dfBufferDice['FOLDER'] = [plotLabels[i]] * (dfBufferDice.shape[0])
        dfWDice = pd.concat([dfWDice, dfBufferDice])

        dfBuffer = pd.read_csv(workdirectery + folderNames[i] + '/' + fileNameResults + '.csv',
                                   sep=';', header=0)
        # add a column with the folder to separate the date later
        dfBuffer['FOLDER'] = [plotLabels[i]] * (dfBuffer.shape[0])

        # calculate the mean over the subjects to get a similar data set as the weighted dice
        subjects = dfBuffer['SUBJECT'].unique()
        for subject in subjects:
            dfsubject = dfBuffer[dfBuffer['SUBJECT'] == subject]
            meanDice = dfsubject['DICE'].mean()
            dfS = pd.DataFrame([[subject, meanDice, plotLabels[i]]],
                               columns=['SUBJECT', 'DICE', 'FOLDER'])
            dataDice = pd.concat([dataDice, dfS])

    rotation = 90

    widths = 0.5
    # plot the box plots out of the modified data sets according to the features (folders)
    ax = dfWDice.boxplot(by='FOLDER', column=['DICE'], rot=rotation, grid=False, color=color1, flierprops=flierprop1,
                         widths=widths)
    dataDice.boxplot(ax=ax, by='FOLDER', column=['DICE'], rot=rotation, grid=False, color=color2, flierprops=flierprop2,
                     widths=widths)
    # define label and design parameters
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

    # save the image normal and with zoomed y axis
    plt.savefig(os.path.join(workdirectery, 'diceStuff.png'), dpi=600)

    plt.ylim(0.4, 0.75)

    plt.savefig(os.path.join(workdirectery, 'diceStuffZoomed.png'), dpi=600)

def plotNoiseData():

    fileNameResults = 'results'

    folderNameNoiseTraining = ["10_T1W_C_I_GA_SP", "11_T1W_C_G_GA_SP", "12_T1W_C_NH_GA_SP"]

    dfNDice = pd.DataFrame()
    dfDice = pd.DataFrame()

    # read in all the data and add a columne to sort it later after the features
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

    # plot the box plots of the sorted dataset
    ax = dfDice.boxplot(by='FOLDER', column=['DICE'], rot=rotation, grid=False, color=color1, flierprops=flierprop1,
                        positions=positions, widths=widths)
    dfNDice.boxplot(by='FOLDER', column=['DICE'], rot=rotation, grid=False, color=color2, flierprops=flierprop2,
                    positions=positionsN, widths=widths, ax=ax)
    # adjust some label and design stuff
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

    # save the image
    plt.savefig(os.path.join(workdirectery, 'noisyStuff.png'), dpi=600)


def main():

    #foldername1 = "00_C"
    #foldername2 = "09_T1W_T2W_C_I_G_NH"

    #plotGraphs(foldername1, foldername2)

    ####################################
    # this section is made to evaluate the float value of mean / median of the different labels. With the debugger
    # the table can be read
    ####################################

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

    # put the different feature runs in the plot function to compare T1 and T2
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

    # this section is to visualize the noisy training data set

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

    # plot the two best against each other to get a final graph
    foldername1 = "04_T1W_C_I_G_NH"
    foldername2 = "09_T1W_T2W_C_I_G_NH"
    graphTitle = "Segmentation Quality - Top Two Results"

    plotGraphs(foldername1, foldername2, graphTitle, True)

    # compare dice with weighted dice
    plotWeightedDice()

    # compare the noisy training data with the normal one
    plotNoiseData()

if __name__ == '__main__':
    main()
