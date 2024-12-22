from .LSTCrPPG import LSTCrPPGDataGenerator, LSTCrPPGDataConfig

class PhythmFormerDataConfig(LSTCrPPGDataConfig):
    pass

class PhythmFormerDataGenerator(LSTCrPPGDataGenerator):
    def _normalization(self,X,y):
        # T,C,W,H
        X = X.transpose((0,3,1,2)) / 255
        y = (y-y.mean())/y.std()
        # y = (y - y.min())/(y.max() -y.min())
        return X,y