# KMAP-Net
1. This is a multi-task framework for survival prediction of lung and gastric cancers.
2. The multi-task include tumor stage, node stage, and survival prediction tasks.
3. Datasets used in this study includes one public lung cancer study[1]
4. The main survial net is showed in "trainattention120.py'.
5. Training and preprocessing is performed using "trainattention1120.py".
6. Training for single task of overall survival is performed using "trainattention112oTaskonlyOS.py


Acknowledgements
We thank a lot for the authors in Ref.[1]. The python library imgaug is used in our code (https://github.com/aleju/imgaug). Furthermore, We also would like to thank the authors of [CheXNet](https://arxiv.org/pdf/1711.05225.pdf) for sharing codes.


Reference
1. Mukherjee P, Zhou M, Lee E, et al. A shallow convolutional neural network predicts prognosis of lung cancer patients in multi-institutional computed tomography image datasets. Nature Machine Intelligence. 2020;2(5):274-282.
