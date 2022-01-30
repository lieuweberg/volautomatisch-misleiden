# LSTM model text prediction

## ⚠️ This code is part of a paper, not intended for public use. ⚠️

Text prediction using an LSTM. This code and the pretrained model are part of the "Volautomatisch Misleiden" paper. If you're here for an example of a model, I'm fairly sure you've come to the worst place possible. The model is super slow and takes ages to train (despite using cuDNN) and isn't even accurate. You'd probably be better off looking somewhere else!

The `saved_model.zip` is the 42nd epoch. Unzip it, and run:
```
python main.py ckpt_e42_l0.058373790234327316
```