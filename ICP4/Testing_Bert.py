import keras
from keras_bert import get_base_dict, get_model, compile_model, gen_batch_inputs
# Build & train the model
model = get_model(
    token_num=3000,
    head_num=1,
    transformer_num=12,
    embed_dim=768,
    feed_forward_dim=3072,
    seq_len=512,
    pos_num=512,
    dropout_rate=0.05,
)
compile_model(model)
model.summary()
