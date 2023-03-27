import torch

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# DEVICE = 'cpu'
TIME_RANGE = 200
TIME_NUM = 5000
FUNCTION_NUM = 100

LSTM_WINDOW_SIZE = 8
LSTM_PREDICT_SIZE = 1
LSTM_INPUT_SIZE = LSTM_WINDOW_SIZE
LSTM_HIDDEN_SIZES = 128
LSTM_NUMS_LAYERS = 3
LSTM_NUM_CLASSES = LSTM_PREDICT_SIZE
LSTM_NUM_EPOCH = 500
LSTM_LEARNING_RATE = 1e-2

HAS_LOAD = False
