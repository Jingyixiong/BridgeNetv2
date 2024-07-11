# A lightweight Transformer-based neural network for large-scale masonry arch bridge point cloud segmentation

This is a official implementation of **BridgeNetv2**. BridgeNetv2 is designed to perform segmentation on large-scale point clouds of infrastructures. To alleviate the memory consumption of Transformer, the local attention mechanism is adopted which enable efficient computations. The source code of **BridgeNet** is also integrated for the comparison.

## Authors

- [Yixiong Jing](https://www.researchgate.net/profile/Yixiong_Jing2), [Brian Sheil](https://www.construction.cam.ac.uk/staff/dr-brian-sheil), [Sinan Acikgoz](https://eng.ox.ac.uk/people/sinan-acikgoz/)
[[BridgeNetv2](https://onlinelibrary.wiley.com/doi/full/10.1111/mice.13201)], [[BridgeNet](https://www.sciencedirect.com/science/article/pii/S0926580522003326)]


## Setup
This code has been tested with Python 3.10, CUDA 11.8, and Pytorch 2.0.1 on Ubuntu 18.04 or higher. BridgeNet and BridgeNetv2 are all trained on a single RTX3080.
- Python environment

```bash
  conda create -n seg_bridge python=3.10
  conda activate seg_bridge
  pip install -r requirements.txt
```
    
## Dataset 
Synthetic dataset is available in [here](https://huggingface.co/datasets/jing222/syn_masonry_bridge/tree/main). Please download the syn_data.zip file and unzip it in the `\data` path for training.


## Usage

BridgeNet and BridgeNetv2 can be used for training and inferencing on the large-scale masonry point clouds.

Train on synthetic point cloud:

- BridgeNet
```python
  python main.py ++general.backbone='bridgenetv1'
```
- BridgeNetv2 
```python
  python main.py ++general.backbone='bridgenetv2'
```

## Results

Comparison of different models in the real masonry bridge point cloud:

![Results](/img/perform_comparison.png)

Visualization of segmentation results:

![Results](/img/results.png)

## Citations

If you find the code is beneficial to your research, please consider citing:

```cite
@article{jing2024lightweight,
  title={A lightweight Transformer-based neural network for large-scale masonry arch bridge point cloud segmentation},
  author={Jing, Yixiong and Sheil, Brian and Acikgoz, Sinan},
  journal={Computer-Aided Civil and Infrastructure Engineering},
  year={2024},
  publisher={Wiley Online Library}
}

@article{jing2022segmentation,
  title={Segmentation of large-scale masonry arch bridge point clouds with a synthetic simulator and the BridgeNet neural network},
  author={Jing, Yixiong and Sheil, Brian and Acikgoz, Sinan},
  journal={Automation in Construction},
  volume={142},
  pages={104459},
  year={2022},
  publisher={Elsevier}
}
```

## License

Our work is subjected to MIT License.

