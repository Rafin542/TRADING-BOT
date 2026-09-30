import torch
import numpy as np
from config import CONFIG
from models.transformer_model import TransformerPredictor

print('Allocating mock data...')
X_train = np.random.randn(1260, 60, 148).astype(np.float32)
y_train = np.random.randint(0, 2, size=(1260,)).astype(np.int64)

print('Initializing model...')
model = TransformerPredictor(CONFIG)
print('Starting fit...')
model.fit(X_train, y_train)
print('Done!')
