# ControlStyle

repo for the paper **ControlStyle: Text-Driven Stylized Image Generation Using Diffusion Priors (MM'23)**

:black_square_button: training code implemented in diffusers (WIP)

:black_square_button: convert pre-trained model into diffusers style (WIP)

:white_check_mark: inference code implemented in diffusers

## run inference
First of all, setting the pre-trained text-to-image diffusion model (sd-v15) and the pre-trained controlstyle model in the bash. Also, different prompt, controlnet_scale and random seed can be set in the bash. Then, 
```
bash test_model.sh
```

## citation
If you find this paper useful, please consider staring 🌟 this repo and citing 📑 our paper:
```
@inproceedings{chen2023controlstyle,
  title={ControlStyle: Text-Driven Stylized Image Generation Using Diffusion Priors},
  author={Chen, Jingwen and Pan, Yingwei and Yao, Ting and Mei, Tao},
  booktitle={Proceedings of the 31st ACM International Conference on Multimedia},
  pages={7540--7548},
  year={2023}
}
```
