import numpy as np
import torch
import joblib

class DrowsyDetector(object):

    def __init__(self):
        self.model_type = None
        self.input_data = None

        self.model_rf_path = './models/RF_drowsy_alert.joblib'
        self.model_lstm_path = './models/clf_lstm_jit6.pth'

        self.model_rf = joblib.load(self.model_rf_path)
        self.model_lstm = torch.jit.load(self.model_lstm_path)
        self.model_lstm.eval()

    def get_classification(self, input_data, model_type):

        # Convert self.input_data to numpy array
        # Convert self.model_type to int
        print(input_data)
        self.model_type = int(model_type)
        self.input_data = np.array(input_data)
        self.input_data = self.input_data.astype(np.float)

        # Manual
        if self.model_type == 0:
            model_input = np.array(self.input_data)[:, 3]
            preds = np.where(np.logical_or(
                model_input > 4, model_input == -1000), 1, 0)
            return int(preds.sum() >= 8)

       # Python RF
        elif self.model_type == 1:
            model_input = np.array(self.input_data)
            preds = self.model_rf.predict(model_input)
            return int(preds.sum() >= 13)
            #return int("1")

        # PyTorch_LSTM
        elif self.model_type == 2:
            model_input = []
            model_input.append(self.input_data[:5])
            model_input.append(self.input_data[3:8])
            model_input.append(self.input_data[6:11])
            model_input.append(self.input_data[9:14])
            model_input.append(self.input_data[12:17])
            model_input.append(self.input_data[15:])
            model_input = torch.FloatTensor(np.array(model_input))
            preds = torch.sigmoid(self.model_lstm(model_input)
                                  ).gt(0.5).int().data.numpy()
            return int(preds.sum() >= 5)

        else:
            raise ValueError('Invalid model type')
