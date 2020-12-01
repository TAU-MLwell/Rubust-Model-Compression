import torch
import torch.nn as nn
from nn_models import UCIModel
import numpy as np


class Oracle(nn.Module):
    def __init__(self, output_number, model_path=None, model_name='uci', device='cpu', in_features=None):
        super(Oracle, self).__init__()
        self.device = device
        if model_name == 'uci':
            self.model = UCIModel(in_features, output_number)
        else:
            raise NotImplementedError('No such model')

        if model_path:
            self.load_model(model_path)

        self.model.eval()

    def set_train(self):
        self.model.train()

    def set_eval(self):
        self.model.eval()

    def load_model(self, model_path):
        self.model.load_state_dict(torch.load(model_path, map_location=self.device))

    def forward(self, x):
        x = self.model(x)
        return x

    def predict(self, data_loader):
        self.set_eval()
        predictions = None
        for i, (samples, labels) in enumerate(data_loader):
            samples = samples.to(self.device, dtype=torch.float)
            with torch.no_grad():
                preds = self.model.predict(samples)
                preds = preds.detach().cpu().numpy()
                if predictions is None:
                    predictions = preds
                else:
                    predictions = np.concatenate([predictions, preds], axis=0)

        return predictions
