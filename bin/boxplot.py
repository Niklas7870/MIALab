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

    directory = "2023-10-24-12-17-44" # must be adjusted
    filename = "results.csv"
    filepath = os.path.join('mia-result', directory, filename)

    data = pd.read_csv(filepath, sep=';', header=0, index_col=["SUBJECT"])
    print(data)

    # dice and housdorffdistance per subject (over all labels)
    data.boxplot(by='SUBJECT', column=['DICE'], rot=45, grid=False)
    data.boxplot(by='SUBJECT', column=['HDRFDST'], rot=45, grid=False)

    # dice and housdorffdistance per label (over all subjects)
    data.boxplot(by='LABEL', column=['DICE'], grid=False)
    data.boxplot(by='LABEL', column=['HDRFDST'], grid=False)

    plt.show()

    # alternative: instead of manually loading/reading the csv file you could also use the pandas package
    # but you will need to install it first ('pip install pandas') and import it to this file ('import pandas as pd')
    # pass  # pass is just a placeholder if there is no other code

    #endtodo

if __name__ == '__main__':
    main()
