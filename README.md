This repository contains the poster and the source code for the project "**Early Bushfire Detection Using Environmental Monitoring Sensing and Deep Learning Approach**".

The datasets generated during the ignition experiments are available upon request.

# Offline models

For the offline models, simply run `Kmeans_and_Isolation_Forest.ipynb` and `autoencoder_offline.ipynb` to generate and visualise the results.

# Online models

The online model is fine-tuned from the pre-trained model of  COUTA (https://github.com/xuhongzuo/couta). To train the real-time **COUTA** model based on customised dataset, place the scripts into the root of the **COUTA** repository, `cd couta` and simply run `bushfire_showcase.py`. The model training parameters can be edited in this file and the model will be saved to `/saved_models`.

To test the model and visualise the score, run `bushfire_experiment_generalization_ability.ipynb`. 

For results and visualization, please refer to the poster.











References:

```
@article{xu2022deep,
  title={Calibrated One-class Classification for Unsupervised Time Series Anomaly
Detection},
  author={Xu, Hongzuo and Wang, Yijie and Jian, Songlei and Liao, Qing and Wang, Yongjun and Pang, Guansong},
  journal={arXiv preprint arXiv:2207.12201},
  year={2022}
}
```
