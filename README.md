# How To Validate Pretrained Models

## Prerequisites
### Package Installation
1. Run the following command to install the required packages:
```
sudo rm /etc/apt/sources.list.d/cuda*
sudo apt remove --autoremove nvidia-cuda-toolkit
sudo apt remove --autoremove nvidia-*
sudo apt update
sudo add-apt-repository ppa:graphics-drivers/ppa
sudo apt-key adv --fetch-keys  http://developer.download.nvidia.com/compute/cuda/repos/ubuntu1604/x86_64/7fa2af80.pub
sudo bash -c 'echo "deb http://developer.download.nvidia.com/compute/cuda/repos/ubuntu1604/x86_64 /" > /etc/apt/sources.list.d/cuda.list'
sudo apt-get update
sudo apt-get install cuda-drivers-465
   (not mandatory) sudo apt-get install cuda-11-1 
   (not mandatory) sudo apt-get install libcudnn8
sudo reboot
```
2. Install Anaconda following the installation instructions [here](https://www.anaconda.com/download/)
3. After Installing Anaconda, run the following command to create a virtual environment that contains the required packages:
   (File idx_environment.yml can be found [here](https://drive.google.com/file/d/136fF3l7v_9OCRz9p754iTUsLQ7SpVgdW/view?usp=sharing))
```
conda env create -n IDX2021 --file idx_environment.yml
conda activate IDX2021
``` 
4. Activate the virtual environment to run the testing scripts.

### Data Preparation
1. Download the dataset from [here](https://drive.google.com/file/d/1zqoXBhf-3mpChcXa_t22Hvs-XBMiUz8F/view?usp=sharing) (Data will be available for download until 12/31).
2. Move the zipfile to the data folder. Now we will refer this data path as $DATA_PATH.
3. Unzip the dataset. 
4. Data hierarchy is as follows:
```
$DATA_PATH/
    official/
        train/
            103_2b2_Ar_mc_LittC2SE_0_01.wav
            105_1b1_Tc_sc_Meditron_0_00.wav
            ...
        val/
            101_1b1_Al_sc_Meditron_0_00.wav
            104_1b1_Ar_sc_Litt3200_12_00.wav
            ...
        image_16k/
        spec_16k/
        wav_16k/
        ICBHI_Challenge_diagnosis.txt
        ICBHI_Challenge_train_test.txt
    spec_16k/
    wav_16k/
    spec_8k/
    wav_8k/
    wav_original/
    txt/
    data_prep.py
    spec_to_image.py
    scripts.sh
```

## Testing
### Pre-Processing
~~1. Run the data_prep.py script to generate the spectrograms from wav files.~~

~~2. Run the spec_to_image.py script to generate the spectrogram images from spec files.~~\
Pre-Processing is no longer required since it is done while loading the data with the dataloader. But you can still check the expected results of the preprocessing in the $DATA_PATH/official/image_16k.

### Run Validation
1. Move to the code folder. Now the code is in the folder $CODE_PATH(unzip the [checkpoints.zip file](https://drive.google.com/file/d/1rLY0lN1aQMnmC-BdzMt1MTKo1nRQGeqq/view?usp=sharing)).
$CODE_PATH hierarchy is as follows:
```
$CODE_PATH/
    checkpoints/
    models/
        __init__.py
        ast_models.py
        cbam.py
        lstm.py
        resnet.py
    cuda0_diagnosis.sh
    cuda1_diagnosis.sh
    cuda2_diagnosis.sh
    dataloader.py
    dataset.py
    stats.py
    validate_with_diagnosis.py
```
2. Run the following command to run the validation script:
```
python validate_with_diagnosis.py --prefix $PREFIX --fixed_size --data_size 256 --batch_size $BATCH_SIZE\
  --model $MODEL_TYPE(one of [ast, lstm, cnnlstm, resnet18]) --data_path $DATA_PATH

example :
python validate_with_diagnosis.py --prefix AST_T256_BS24_ADAM1e-5_MIXUP_SEED1234 --fixed_size --data_size 256 --batch_size 24\
  --model ast --data_path $DATA_PATH/official/
```
3. The output will be saved in the folder $CODE_PATH/results/{$PREFIX}_official/.
