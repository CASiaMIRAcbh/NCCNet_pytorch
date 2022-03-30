# NCCNet_pytorch
This is a project which uses pytorch in windows to re-implement NCCNet.
Previous NCCNet is developed with tensorflow in https://github.com/seung-lab/NCCNet.
We try to re-implement the NCCNet based on pytorch, thus we create this project.


# How to use?
This project is developed in jupyter+pytorch in windows 10. We suppose you to install pytorch and jupyter with anaconda to test it.
The pytorch virsion used to develop this project is 1.3.0.  But I think more recent version of pytorch also may work.

# How to generate training data?
The training data generating script is generateTrainingData.ipynb. You can use jupyter to open and run it. You just need to change the
original data path and image size(Our EM images is 4600x4600, you may have images of other size).

# What are the other python file?
The files are other tools of nccnet. We think the most important file is lossfunc.py. It re-implement the loss function. Maybe you can
read it and make it better. The loss function, *ssim in indice.py cannot be fixed. Please keep using NCC as your loss function.

# I have tried several times to run your code, but it still not works. What should I do?
Maybe you can connect to us. Write an e-mail to us and we may try to solve your problems.

# Will the project be fixed/upgraded in the future?
As a new developer who try to use github, I think I will push&pull several times to practice my git skill. So maybe the project will
have too many branches. Please use main branch. BTW, I will graduate from my master student, and try to work for some new challenges.
