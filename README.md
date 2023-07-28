# Contrastive-Guided-Diffusion-Process

This repo contains the pytorch code for experiments in the paper [Improving Adversarial Robustness Through the Contrastive-Guided Diffusion Process](https://arxiv.org/abs/2210.09643)

by Yidong Ouyang, Liyan Xie, Guang Cheng.

We analyze the optimality condition of synthetic distribution for achieving improved robust accuracy and then propose to use the contrastive guidance to enhance the discriminative of the synthetic data. Contrastive-Guided Diffusion Process enhances the sample efficiency of synthetic data.

### Usage

Training Contrastive-DP for CIFAR through ./ddim-main-contrast
```
python main.py --config {DATASET}.yml --exp {PROJECT_PATH} --doc {MODEL_NAME} --ni
```
Synthesize data
```
python main.py --config {DATASET}.yml --exp {PROJECT_PATH} --doc {MODEL_NAME} --sample --fid --timesteps {STEPS} --eta {ETA} --ni
```
Training Contrastive-DP for MNIST/GTSRB and sampling data through ./minimal-diffusion-main

```
python main_nonddp.py \
--arch UNet --dataset cars --class-cond --sampling-only --sampling-steps 250 \
--num-sampled-images 50000 --pretrained-ckpt 
```

```
python main_nonddp.py \
--arch UNet --dataset cars --class-cond --sampling-only --sampling-steps 250 \
--num-sampled-images 50000 --pretrained-ckpt /mntnfs/apmath_data1/UNet_cars-epoch_500-timesteps_1000-class_condn_True_ema_0.9995.pt
```

Adversarial training through ./adversarial_robustness_pytorch-main

```
python train-wa.py --data-dir <data_dir> \
    --log-dir <log_dir> \
    --desc <name_of_the_experiment> \
    --data cifar10s \
    --batch-size 1024 \
    --model wrn-28-10-swish \
    --num-adv-epochs 400 \
    --lr 0.4 \
    --beta 6.0 \
    --unsup-fraction 0.7 \
    --aux-data-filename <path_to_additional_data>
```

### References

If you find the code useful for your research, please consider citing

```bib
@inproceedings{
  Ouyang2023cont,
  title={Improving Adversarial Robustness Through the Contrastive-Guided Diffusion Process},
  author={Yidong Ouyang, Liyan Xie, Guang Cheng},
  booktitle={International Conference on Machine Learning},
  year={2023},
}
```

This implementation is heavily based on 
* [Diffusion model for CIFAR](https://github.com/ermongroup/ddim/blob/main/main.py) (provide the DDIM for CIFAR)
* [Diffusion model for MNIST/GTSRB](https://github.com/VSehwag/minimal-diffusion) (provide the diffusion model on MNIST/GTSRB)
* [Adversarial training](https://github.com/imrahulr/adversarial_robustness_pytorch)(provide code for adversarial training)
