# Container Infrastructure

Generally, the entire project can be devided into two segments, the `data_generation` and the `deep_learning` part. In order to convey this idea, the infrastructure part is also devided accordingly. The table below provides a brief overview how the infrastructure part is separated. Furthermore, an elaborate outline can be found in every of the two container folders:

|Directory|Purpose|
|----|-------|
|[`data_generation`](data_generation)|This directory contains everything which is needed in order to set up the Docker/Enroot container for the data generation part.|
|[`deep_learning`](deep_learning)|This directory contains everything which is needed in order to set up the Docker/Enroot container for the deep learning part.|
