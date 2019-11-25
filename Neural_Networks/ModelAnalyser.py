import numpy as np
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf

FLAGS = tf.app.flags.FLAGS


class ModelAnalyser():

    def __init__(self, variable, accuracy = 0):
        self.weigh = variable['weight']
        self.bias = variable['bias']

        self.nbWeight = len(self.weigh)
        # [layer][sample][neuron]
        self.nbLayer = len(variable['layer'])
        self.nbSamples = len(variable['layer'][0])

        self.layer = [
            pd.DataFrame(data=variable['layer'][i], columns=[j for j in range(0, len(variable['layer'][i][0]))]) for i
            in range(self.nbLayer)]

        # [0][sample][neuron]
        self.labels = variable['labels'][0]

        #  [sample][cat]
        self.labels = pd.DataFrame([np.argmax(self.labels[i]) for i in range(len(self.labels))])

        # [0][sample][neuron]
        self.prediction = variable['prediction'][0]

        #  [sample][cat]
        self.prediction = pd.DataFrame([np.argmax(self.prediction[i]) for i in range(len(self.prediction))])


        self.accuracy = accuracy
    # implementation of z+ rule see http://heatmapping.org/tutorial/
    def lrp_zpr(self):

        newLayers = []

        # for i in range(self.nbSamples):
        for i in range(1):
            tmpLayer = []
            tmpLayer.append(self.layer[i][self.nbWeight - 1])
            for j in range(self.nbWeight - 1, -1, -1):
                inl = self.nbWeight - j
                print(" j", j)

                print("self.layer[j - 1]  ", self.layer[i][j - 1], "len(self.layer[i][j - 1] ) ",
                      len(self.layer[i][j - 1]))
                print("np.maximum(self.weigh[j], 0) ", np.maximum(self.weigh[j], 0))
                print("self.weigh[j] ", self.weigh[j])

                # vector
                z = self.layer[i][j - 1] * np.maximum(self.weigh[j], 0)

                print("z ", z)

                # vector
                s = np.divide(tmpLayer[inl], z)
                print("s ", s)
                print("newLayers[inl]  ", tmpLayer[inl])

                # vector
                c = np.maximum(self.weigh[j], 0) * s

                print("c ", c)
                print("np.maximum(self.weigh[j], 0) ", np.maximum(self.weigh[j], 0))

                # vector
                tmpLayer.append(self.layer[i][j - 2] * c)

                print("newLayers ", tmpLayer[1])
                print("self.layer[j - 2] ", self.layer[i][j - 2])

                exit(0)
        return

    def lda_prediction(self):
        '''

        Perform a LDA in layers and model prediction

        :return: save figs in /Results/Analyser
        '''

        for ii in range(0, self.nbLayer ):

            #  LDA
            lda = LinearDiscriminantAnalysis(n_components=None)
            X_lda = lda.fit(X=self.layer[ii], y=self.prediction).transform(self.layer[ii])

            # Percentage of variance explained for each components
            print('explained variance ratio (first two components): %s'
                  % str(lda.explained_variance_ratio_))

            # Plot the LDA
            plt.figure()
            colors = ['navy', 'turquoise', 'darkorange', "red"]

            for color, i, target_name in zip(colors, [0, 1, 2, 3], ["0", "1", "2", "3"]):

                #  np.where(self.labels[0] == i ) give a tuples and we want the first element
                subSetSample = np.ndarray.flatten(np.where(self.labels[0] == i)[0])
                # print(" subSetSample " , subSetSample)

                # In case we have just one component
                if (len(lda.explained_variance_ratio_) == 1):

                    plt.scatter(X_lda[subSetSample, 0], np.zeros(len(subSetSample)), alpha=.8, color=color,
                                label=target_name)
                    plt.xlabel('component 1 : %.2f' % lda.explained_variance_ratio_[0])
                else:
                    plt.scatter(X_lda[subSetSample, 0], X_lda[subSetSample, 1], alpha=.8,
                                color=color,
                                label=target_name)
                    plt.xlabel('component 1 : %.2f' % lda.explained_variance_ratio_[0])
                    plt.ylabel('component 2 : %.2f' % lda.explained_variance_ratio_[1])
            plt.legend(loc='best', shadow=False, scatterpoints=1)
            plt.title("LDA_of_" + FLAGS.modele_short_name + "  acc = %.2f" % self.accuracy, fontsize=7)
            plt.savefig(FLAGS.modele_name +  'ldaPrediction_'+ FLAGS.modele_short_name  + '_' + str(ii) + '.png')

    def lda_labels(self):
        '''

        Perform a LDA in layers and model prediction

        :return: save figs in /Results/Analyser
        '''

        for ii in range(0, self.nbLayer ):

            lda = LinearDiscriminantAnalysis(n_components=None)
            X_lda = lda.fit(X=self.layer[ii][0:self.nbSamples], y=self.labels[0:self.nbSamples]).transform(self.layer[ii][0:self.nbSamples])

            # Percentage of variance explained for each components
            print('explained variance ratio (first two components): %s'
                  % str(lda.explained_variance_ratio_))

            plt.figure()
            colors = ['navy', 'turquoise', 'darkorange', "red"]

            for color, i, target_name in zip(colors, [0, 1, 2, 3], ["0", "1", "2", "3"]):

                #  np.where(self.labels[0] == i ) give a tuples and we want the first element
                #subSetSample = np.ndarray.flatten(np.where(self.labels[0:500] == i)[0])
                subSetSample = np.ndarray.flatten(np.where(self.labels == i)[0])
                #print(" subSetSample " , subSetSample)

                # In case we have just one component
                if (len(lda.explained_variance_ratio_) == 1):

                    plt.scatter(X_lda[subSetSample, 0], np.zeros(len(subSetSample)), alpha=.8, color=color,
                                label=target_name)
                    plt.xlabel('component 1 : %.2f'  % lda.explained_variance_ratio_[0])

                else:
                    plt.scatter(X_lda[subSetSample, 0], X_lda[subSetSample, 1], alpha=.8,
                                color=color,
                                label=target_name)
                    plt.xlabel('component 1 : %.2f' % lda.explained_variance_ratio_[0])
                    plt.ylabel('component 2 : %.2f' % lda.explained_variance_ratio_[1])


            plt.legend(loc='best', shadow=False, scatterpoints=1)
            plt.title("LDA_of_" + FLAGS.modele_short_name  +  "  acc = %.2f" % self.accuracy  , fontsize=7)

            plt.savefig(FLAGS.modele_name +  'ldaLabels_' + FLAGS.modele_short_name  + '_' + str(ii) + '.png')

    def lda3(self):

        for ii in range(0, self.nbLayer):
            # mean of class  :

            lda = LinearDiscriminantAnalysis(n_components=3)

            # print("self.layer " , self.layer)
            # print("self.labels " , self.labels)

            # print("self.layer[0] " , self.layer[0])
            X_lda = lda.fit(X=self.layer[ii], y=self.labels).transform(self.layer[ii])

            print(X_lda)
            # Percentage of variance explained for each components
            print('explained variance ratio (first two components): %s'
                  % str(lda.explained_variance_ratio_))

            print("indice ", np.where(self.labels == 4))

            fig = plt.figure()
            ax = fig.add_subplot(111, projection='3d')
            colors = ['navy', 'turquoise', 'darkorange', "red"]

            for color, i, target_name in zip(colors, [1, 2, 3, 4], ["1", "2", "3", "4"]):
                ax.scatter(X_lda[np.where(self.labels == i), 0], X_lda[np.where(self.labels == i), 1],
                           X_lda[np.where(self.labels == i), 2], alpha=.8, color=color,
                           label=target_name)

            ax.set_xlabel('X Label')
            ax.set_ylabel('Y Label')
            ax.set_zlabel('Z Label')
            plt.legend(loc='best', shadow=False, scatterpoints=1)
            plt.title('LDA of ')

            for angle in range(0, 360, 90):
                ax.view_init(30, angle)
                plt.savefig('LDA3foo' + str(ii) + "_" + str(angle) + '.png')
                plt.pause(.001)

    # TODO
    def qda_prediction(self):
        '''
        Perform a LDA in layers and model prediction
        :return: save figs in /Results/Analyser
        '''

        for ii in range(0, self.nbLayer):

            qda = QuadraticDiscriminantAnalysis()




            # rotations_ : list of arrays
            # For each class k an array of shape [n_features, n_k], with n_k = min(n_features, number of elements in class k)
            # It is the rotation of the Gaussian distribution, i.e. its principal axis.
            X_qda = qda.fit(X=self.layer[ii], y=self.labels).rotations


            plt.figure()
            colors = ['navy', 'turquoise', 'darkorange', "red"]

            for color, i, target_name in zip(colors, [0, 1, 2, 3], ["0", "1", "2", "3"]):

                #  np.where(self.labels[0] == i ) give a tuples and we want the first element
                subSetSample = np.ndarray.flatten(np.where(self.labels[0] == i)[0])


                plt.scatter(X_qda[subSetSample, 0], X_qda[subSetSample, 1], alpha=.8,
                                color=color,
                                label=target_name)
            plt.legend(loc='best', shadow=False, scatterpoints=1)
            plt.title('LDA of ')

            plt.savefig(FLAGS.path_figs_modelAnalyser + '_lda_Layer_' + str(ii) + '.png')
