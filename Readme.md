## Dataset
- Unzip data.zip to `./{current_working_directory}`
    ```sh
    unzip data.zip -d ./{current_working_directory}
    ```
- Manually create a sub folder named "data", and put all images that were unzipped into the sub folder.

- Folder structure
    ```
    .
    ├── generate
    ├── mnist_data
    │   └── data/  
    ├── model_weight
    ├── data.csv
    ├── Generate_digits.py
    ├── main.py
    ├── mnist.npz
    └── prepare_data_path.py
    ```

## Environment
- Python 3.9 or later version
    ```sh
    conda create --name <env> --file requirements.txt
    ```

## Preprocess unlabeled Pre-training data
- Saving the jpgs paths.
- Create a csv file named as `data.csv`
```sh
python prepare_data_path.py
```

## Train Denoising Diffusion Probabilistic Model
- Customed Unet
- With RTX 2080ti and 128GB RAM, it may cost 1 hour to train.
- The model weight would be saved in folder "model_weight", its weight name **may be** `ddpm_mnist.pt`.
```sh
python main.py
```

## Generating digits, dissusion process visualization, calculate FID
- 10000 samples would be generated in the directory "generate".
- 8 digits of diffusion process example plot would be saved as `diffusion_process.png` 
- After the 10000 sample digits are generated. The Frechet Inception Distance FID would be calcuted and displayed (the last number is FID).
```sh
python Generate_digits.py
```
