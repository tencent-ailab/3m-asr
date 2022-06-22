## 3M-ASR for End-to-End Speech Recognition

This project is used to build an End-to-End Speech Recognition system based on Mixture-of-Experts(MoE) model.  MoE is an efficient way to train a large scale model and we have proved its efficiency on public dataset. More details about the algorithm can be found in "[3M: Multi-loss, Multi-path and Multi-level Neural Networks for speech recognition](https://arxiv.org/abs/2204.03178)".



## Installation

- Clone this repo

```shell
git clone https://github.com/tencent-ailab/3m-asr.git
```

- Install Conda: please see [https://docs.conda.io/en/latest/miniconda.html](https://docs.conda.io/en/latest/miniconda.html)
- Create Conda env:

```shell
conda create -n moe python=3.8
conda activate moe
pip install -r requirements.txt
conda install pytorch==1.9.0 torchvision==0.10.0 torchaudio==0.9.0 cudatoolkit=11.1 -c pytorch -c conda-forge
```

- Follow the instruction under directory `fastmoe` to install `fastmoe` 



## Performance Benchmark

We evaluate our system on the public [WenetSpeech](https://github.com/wenet-e2e/WenetSpeech) dataset and the recipe of `Conformer-MoE` is provided(trained on 24 V100).  CER results are listed below and the first three lines are provided by [WenetSpeech](https://github.com/wenet-e2e/WenetSpeech)

|      Toolkit       |   Dev    | Test_net | Test_Meeting | AIShell-1 |
| :----------------: | :------: | :------: | :----------: | :-------: |
|       Kaldi        |   9.07   |  12.83   |    24.72     |   5.41    |
|       Espnet       |   9.70   |   8.90   |    15.90     | **3.90**  |
|       WeNet        |   8.88   |   9.70   |    15.59     |   4.61    |
| Conformer-MoE(32e) | **7.49** | **7.99** |  **13.69**   |   4.03    |



## Acknowledge

- We used [FastMoE](https://github.com/laekov/fastmoe) to support Mixture-of-Experts model training in Pytorch
- We borrowed  a lot of code from [WeNet](https://github.com/wenet-e2e/wenet) for the implementation of Conformer and data processing



## Reference

[1] [SpeechMoE: Scaling to Large Acoustic Models with Dynamic Routing Mixture of Experts](https://arxiv.org/abs/2105.03036)(InterSpeech 2021)

[2] [3M: Multi-loss, Multi-path and Multi-level Neural Networks for speech recognition](https://arxiv.org/abs/2204.03178)(Submitted to InterSpeech 2022)



## Citation

```tex
@inproceedings{you21_interspeech,
  author={Zhao You and Shulin Feng and Dan Su and Dong Yu},
  title={{SpeechMoE: Scaling to Large Acoustic Models with Dynamic Routing Mixture of Experts}},
  year=2021,
  booktitle={Proc. Interspeech 2021},
  pages={2077--2081},
  doi={10.21437/Interspeech.2021-478}
}

@article{you20223m,
  title={3M: Multi-loss, Multi-path and Multi-level Neural Networks for speech recognition},
  author={You, Zhao and Feng, Shulin and Su, Dan and Yu, Dong},
  journal={arXiv preprint arXiv:2204.03178},
  year={2022}
}
```

## Contact
If you have any questions about this project, please feel free to contact shulinfeng@tencent.com or dennisyou@tencent.com

## Disclaimer

This is not an officially supported Tencent product
