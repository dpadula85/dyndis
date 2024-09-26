Files provided contain:

- `dataset.py`: script to generate an `.npz` object to be
used in conjunction with [PyTorch Dataset](https://pytorch.org/tutorials/beginner/basics/data_tutorial.html)
to be supplied to the model;

- `splits_sp2.pt`: indices of molecules used in train, validation, and test sets;

- `trainsp2_chk.pt`: weights of the SphereNet model used for the paper;

- `datasets.py`: script to create a PyTorch Dataset to feed to the model;

- `train_sp2.py`: script to train the network;

- `test_sp2.py`: script to obtain predictions;

- `grads_sp2.py`: script to obtain model gradients.
