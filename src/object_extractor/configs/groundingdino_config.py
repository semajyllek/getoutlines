# src/object_extractor/configs/groundingdino_config.py

modelname = "groundingdino"

# Model Architecture
num_queries = 900
hidden_dim = 256
position_embedding = "sine"
text_encoder_type = "bert-base-uncased"
enc_layers = 6
dec_layers = 6
pre_norm = False
hidden_dim = 256

# Backbone Config
backbone = 'swin_T_224_1k'
return_interm_indices = [1, 2, 3]

# Detection Parameters
text_threshold = 0.25
box_threshold = 0.35

# These are required by the model architecture but we don't modify them
nheads = 8
dim_feedforward = 2048
dropout = 0.0
enc_n_points = 4
dec_n_points = 4
transformer_activation = "relu"
num_patterns = 0
num_feature_levels = 4
use_checkpoint = True