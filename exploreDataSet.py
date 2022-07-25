# Generate and plot a synthetic imbalanced classification dataset
from collections import Counter
from matplotlib import pyplot
from numpy import where
import myServices as ms
 

class describeDataset():
    def __init__(self, dataset, targetCol) -> None:
        self.X, self.Y = ms.importDataSet(dataset, targetCol)
        self.classCount = Counter(self.Y)
        describeDataset.plotScatterOfClasses(self)
        return

    def plotScatterOfClasses(self):
        for label, _ in self.classCountter.items():
            row_ix = where(self.y == label)[0]
            pyplot.scatter(self.X[row_ix, 0], self.X[row_ix, 1], label=str(label))
        pyplot.title(self.classCount)
        pyplot.legend()
        pyplot.show()

####  TODO ###

    def confussionMatrix():
        return 
    




