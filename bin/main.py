"""A medical image analysis pipeline.

The pipeline is used for brain tissue segmentation using a decision forest classifier.
"""
import argparse
import datetime
import os
import sys
import timeit
import warnings

import SimpleITK as sitk
import sklearn.ensemble as sk_ensemble
import numpy as np
import pymia.data.conversion as conversion
import pymia.evaluation.writer as writer

import csv

try:
    import mialab.data.structure as structure
    import mialab.utilities.file_access_utilities as futil
    import mialab.utilities.pipeline_utilities as putil
    import bin.test_set_creation as tset
except ImportError:
    # Append the MIALab root directory to Python path
    sys.path.insert(0, os.path.join(os.path.dirname(sys.argv[0]), '..'))
    import mialab.data.structure as structure
    import mialab.utilities.file_access_utilities as futil
    import mialab.utilities.pipeline_utilities as putil
    import bin.test_set_creation as tset

LOADING_KEYS = [structure.BrainImageTypes.T1w,
                structure.BrainImageTypes.T2w,
                structure.BrainImageTypes.GroundTruth,
                structure.BrainImageTypes.BrainMask,
                structure.BrainImageTypes.RegistrationTransform]  # the list of data we will load

label_mapping = {
            0: 'Background',
            1: 'WhiteMatter',
            2: 'GreyMatter',
            3: 'Hippocampus',
            4: 'Amygdala',
            5: 'Thalamus'
        }


def getDiceScores(results):
    uniqueLabes = set(results.label for results in results if results.metric == 'DICE')
    diceScore = {}
    for label in uniqueLabes:
        diceScore[label] = [result.value for result in results if result.metric == 'DICE' and result.label == label]
    return diceScore

def calculateWeightedDiceScore(buffer, diceScore):

    totalSegVolume = 0

    for index in range(1, 6):
        if np.sum(buffer==index) != 0:
            totalSegVolume += 1/float(np.sum(buffer==index))

    diceSumWithArea = 0

    for index in range(0, 5):
        if np.sum(buffer==index+1) != 0:
            diceSumWithArea += diceScore[index] / float(np.sum(buffer==index+1))

    weightedDice = diceSumWithArea / totalSegVolume

    return weightedDice


def main(result_dir: str, data_atlas_dir: str, data_train_dir: str, data_test_dir: str):
    """Brain tissue segmentation using decision forests.

    The main routine executes the medical image analysis pipeline:

        - Image loading
        - Registration
        - Pre-processing
        - Feature extraction
        - Decision forest classifier model building
        - Segmentation using the decision forest classifier model on unseen images
        - Post-processing of the segmentation
        - Evaluation of the segmentation
    """

    np.random.seed(42)

    # load atlas images
    putil.load_atlas_images(data_atlas_dir)

    print('-' * 5, 'Training...')

    # crawl the training image directories
    crawler = futil.FileSystemDataCrawler(data_train_dir,
                                          LOADING_KEYS,
                                          futil.BrainImageFilePathGenerator(),
                                          futil.DataDirectoryFilter())

    #starttodo
    pre_process_params = {'skullstrip_pre': True,
                          'normalization_pre': True,
                          'registration_pre': True,
                          'coordinates_feature': True,
                          'intensity_feature': True,
                          'gradient_intensity_feature': True,
                          'neighborhood_feature': False,
                          'T1W_Image': True,
                          'T2W_Image': False,
                          'gaussian': False,
                          'salt_pepper': False}

    multiprocess = False

    # load images for training and pre-process
    images = putil.pre_process_batch(crawler.data, pre_process_params, multi_process=multiprocess)

    # generate feature matrix and label vector
    data_train = np.concatenate([img.feature_matrix[0] for img in images])
    labels_train = np.concatenate([img.feature_matrix[1] for img in images]).squeeze()

    #starttodo
    # set proper random forest parameters
    # warnings.warn('Random forest parameters not properly set.')

    # n_estimator (number of trees) increased (1 --> 10)
    # default 10 (sklearn <0.22), default 100 (sklearn > 0.22)
    # best result for base pipeline --> n_estimators = 10, max_depth = 90
    # best result considering calculation time --> n_estimators = 10, max_depth =30

    # parameters to be adjusted
    n_estimators = 10
    max_depth = 30
    forest = sk_ensemble.RandomForestClassifier(max_features=images[0].feature_matrix[0].shape[1],
                                                n_estimators=n_estimators, max_depth=max_depth)
    #endtodo

    start_time = timeit.default_timer()
    forest.fit(data_train, labels_train)
    print(' Time elapsed:', timeit.default_timer() - start_time, 's')

    #starttodo
    # create a result directory with timestamp

    name = ''
    for key, value in pre_process_params.items():
        if key == 'T1W_Image':
            if value: name = key[0:3] + '_' + name
        if key == 'T2W_Image':
            if value: name = key[0:3] + '_' + name
        if key == 'coordinates_feature':
            if value: name = name + 'C_'
        if key == 'intensity_feature':
            if value: name = name + 'I_'
        if key == 'gradient_intensity_feature':
            if value: name = name + 'G_'
        if key == 'neighborhood_feature':
            if value: name = name + 'NH_'
        if key == 'gaussian':
            if value: name = name + 'GA_'
        if key == 'salt_pepper':
            if value: name = name + 'SP_'

    name = '_' + name[:-1]

    t = datetime.datetime.now().strftime('%Y-%m-%d-%H-%M-%S.%f')[:-3]  # milliseconds added (microseconds [:-3])
    t = t + name
    result_dir = os.path.join(result_dir, t)
    os.makedirs(result_dir, exist_ok=True)
    #stoptodo

    print('-' * 5, 'Testing...')

    test_loop_parameter = ["", "_gaussian_300", "_gaussian_1000", "_gaussian_2000", "_gaussian_5000",
                               "_salt_pepper_001", "_salt_pepper_002", "_salt_pepper_005"]

    tset.main()

    for test_str in test_loop_parameter:
        test_dir = data_test_dir + test_str

        # initialize evaluator
        evaluator = putil.init_evaluator()

        # crawl the training image directories
        crawler = futil.FileSystemDataCrawler(test_dir,
                                              LOADING_KEYS,
                                              futil.BrainImageFilePathGenerator(),
                                              futil.DataDirectoryFilter())

        # load images for testing and pre-process
        pre_process_params['training'] = False
        images_test = putil.pre_process_batch(crawler.data, pre_process_params, multi_process=multiprocess)

        images_prediction = []
        images_probabilities = []

        for img in images_test:
            print('-' * 10, 'Testing', img.id_, test_str)

            start_time = timeit.default_timer()
            predictions = forest.predict(img.feature_matrix[0])
            probabilities = forest.predict_proba(img.feature_matrix[0])
            print(' Time elapsed:', timeit.default_timer() - start_time, 's')

            # convert prediction and probabilities back to SimpleITK images
            image_prediction = conversion.NumpySimpleITKImageBridge.convert(predictions.astype(np.uint8),
                                                                            img.image_properties)
            image_probabilities = conversion.NumpySimpleITKImageBridge.convert(probabilities, img.image_properties)

            # evaluate segmentation without post-processing
            evaluator.evaluate(image_prediction, img.images[structure.BrainImageTypes.GroundTruth], img.id_)

            images_prediction.append(image_prediction)
            images_probabilities.append(image_probabilities)

        #stardtodo
        # --> postprocessing removed

        diceScores = getDiceScores(evaluator.results)

        weightedDiceScore = np.zeros((images_test.__len__(), 2))
        counter = 0

        for img in images_test:

            buffer = sitk.GetArrayFromImage(img.images[structure.BrainImageTypes.GroundTruth])
            diceList = []
            for index in range(1, 6):
                diceList.append(diceScores[label_mapping[index]][counter])

            weightedDiceScore[counter][0] = int(img.id_)
            weightedDiceScore[counter][1] = calculateWeightedDiceScore(buffer, diceList)

            counter += 1

        #stardtodo
        # --> postprocessing removed

        # post-process segmentation and evaluate with post-processing
        # post_process_params = {'simple_post': True}
        # images_post_processed = putil.post_process_batch(images_test, images_prediction, images_probabilities,
        #                                                  post_process_params, multi_process=multiprocess)
        #
        for i, img in enumerate(images_test):
        #     evaluator.evaluate(images_post_processed[i], img.images[structure.BrainImageTypes.GroundTruth],
        #                        img.id_ + '-PP')

            # save results
            sitk.WriteImage(images_prediction[i], os.path.join(result_dir, images_test[i].id_ + '_SEG'+test_str+'.nii.gz'), False)
            # sitk.WriteImage(images_post_processed[i], os.path.join(result_dir, images_test[i].id_ + '_SEG-PP.mha'), True)
            if test_str != "":
                break
        #endtodo

        # use two writers to report the results
        os.makedirs(result_dir, exist_ok=True)  # generate result directory, if it does not exists
        result_file = os.path.join(result_dir, 'results'+test_str+'.csv')
        writer.CSVWriter(result_file).write(evaluator.results)



        result_file = os.path.join(result_dir, 'weightedDiceScore'+test_str+'.csv')
        file = open(result_file, 'w', encoding='UTF8', newline='')
        writerCsv = csv.writer(file)
        writerCsv.writerow(['SUBJECT', 'DICE'])
        writerCsv.writerows(weightedDiceScore)

        print('\nSubject-wise results...')
        writer.ConsoleWriter().write(evaluator.results)

        # report also mean and standard deviation among all subjects
        result_summary_file = os.path.join(result_dir, 'results_summary'+test_str+'.csv')
        functions = {'MEAN': np.mean, 'STD': np.std}
        writer.CSVStatisticsWriter(result_summary_file, functions=functions).write(evaluator.results)
        print('\nAggregated statistic results...')
        writer.ConsoleStatisticsWriter(functions=functions).write(evaluator.results)

        # clear results such that the evaluator is ready for the next evaluation
        evaluator.clear()

        #starttodo
        # generate textfile with run-parameter
        filename = os.path.join(result_dir, 'run-parameter.txt')
        f = open(filename, mode='w')
        f.write('## Random forrest ##' + '\n')
        f.write('n_estimators: ' + str(n_estimators) + '\n')
        f.write('max_depth: ' + str(max_depth) + '\n')
        f.write('\n')

        f.write('## Notes ##' + '\n')
        for key, value in pre_process_params.items():
            f.write('%s: %s\n' % (key, value))

        f.close()
        #stoptodo





if __name__ == "__main__":
    """The program's entry point."""

    script_dir = os.path.dirname(sys.argv[0])

    parser = argparse.ArgumentParser(description='Medical image analysis pipeline for brain tissue segmentation')

    parser.add_argument(
        '--result_dir',
        type=str,
        default=os.path.normpath(os.path.join(script_dir, './mia-result')),
        help='Directory for results.'
    )

    parser.add_argument(
        '--data_atlas_dir',
        type=str,
        default=os.path.normpath(os.path.join(script_dir, '../data/atlas')),
        help='Directory with atlas data.'
    )

    parser.add_argument(
        '--data_train_dir',
        type=str,
        default=os.path.normpath(os.path.join(script_dir, '../data/train/')),
        help='Directory with training data.'
    )

    parser.add_argument(
        '--data_test_dir',
        type=str,
        default=os.path.normpath(os.path.join(script_dir, '../data/test/')),
        help='Directory with testing data.'
    )

    args = parser.parse_args()
    main(args.result_dir, args.data_atlas_dir, args.data_train_dir, args.data_test_dir)
