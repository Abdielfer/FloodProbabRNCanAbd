# Generate and plot a synthetic imbalanced classification dataset
import myServices as ms
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import ConfusionMatrixDisplay
import numpy as np
from numpy import where


class describeDataset():
    '''
    This class takes adataset as input to performe descriptive representation of features and they relationships. 
    '''
    def __init__(self, dataset, targetCol) -> None:
        self.X, self.Y = ms.importDataSet(dataset, targetCol)
        self.targetCol = targetCol
        self.classCount = np.unique(self.Y)
        self.dataset = dataset
        return

    def plotScatterOfClasses(self, colName: str):
        for label in self.classCount:
            row_ix = where(self.Y == label)[0]
            plt.scatter(self.X[row_ix,colName], label, label=str(label))
        plt.title("Class distribution with respect to ",colName)
        plt.legend()
        plt.show()

    def makeDatasetBoxPlot(self, a: str,b: str, hue: str):
        plt.figure, ax = plt.subplots(figsize=(10, 8))
        ax = sns.boxplot(x=a, y=a, hue=hue, data=self.dataset)
        ax.set_title(f"Visualize {a} wrt {b}")
        ax.legend()
        plt.show()
        return 
   
    def featuresPariPlot(self):
        sns.pairplot(self.X)
        plt.show()  

    def featuresCorrelationHeatMap(self):
        self.X.corr().style.format("{:.4}").background_gradient(cmap=plt.get_cmap('coolwarm'), axis=1)

def plotConfusionMatrixFromEstimator(estimator,X_test, y_test):
    ConfusionMatrixDisplay.from_estimator(estimator, X_test, y_test)
    plt.show()
        
def main():
    describer = describeDataset('basin1CleanToTrain.csv','percentage' )

if __name__ == "__main__":
    main()



