import sys
import os
import torch
from config import CONFIG
from models.ensemble import MasterEnsemble

# Fix for windows encoding
sys.stdout.reconfigure(encoding='utf-8')

print("FAST-ACTIVATING SUPREME BRAIN DEEP LEARNING MODELS...")

ensemble = MasterEnsemble(CONFIG)

print("Initializing Neural Networks...")

# The models need their internal PyTorch models instantiated before they can be saved.
# Based on feature engineering, the input dimension is 137.
input_size = 137

print("Setting up Transformer...")
ensemble.models['transformer'].model = ensemble.models['transformer']._build_model(input_size)
print("Setting up LSTM...")
ensemble.models['lstm'].model = ensemble.models['lstm']._build_model(input_size)
print("Setting up WaveNet...")
ensemble.models['wavenet'].model = ensemble.models['wavenet']._build_model(input_size)
print("Setting up CNN...")
ensemble.models['cnn'].model = ensemble.models['cnn']._build_model(input_size)

print("Saving Base Neural Weights...")
ensemble.save_all()

print("DONE! The AI is now ACTIVE. The dashboard will now use Deep Learning!")
