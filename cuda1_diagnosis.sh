CUDA_VISIBLE_DEVICES=1 python train_with_diagnosis.py --prefix LSTM_T256_BS24_SGD0.01_MIXUP_SEED1234 --model lstm --lr 0.01 --optimizer sgd --seed 1234 --mixup --fixed_size  --data_size 256 --batch_size 24 --data_path dataset/ICBHI_final_database/official
CUDA_VISIBLE_DEVICES=1 python train_with_diagnosis.py --prefix LSTM_T256_BS24_ADAM3e-4_MIXUP_SEED1234 --model lstm --lr 3e-4 --optimizer adam --seed 1234 --mixup --fixed_size  --data_size 256 --batch_size 24 --data_path dataset/ICBHI_final_database/official

CUDA_VISIBLE_DEVICES=1 python train_with_diagnosis.py --prefix CNNLSTM_T256_BS24_SGD0.01_MIXUP_SEED1234 --model cnnlstm --lr 0.01 --optimizer sgd --seed 1234 --mixup --fixed_size  --data_size 256 --batch_size 24 --data_path dataset/ICBHI_final_database/official
CUDA_VISIBLE_DEVICES=1 python train_with_diagnosis.py --prefix CNNLSTM_T256_BS24_ADAM3e-4_MIXUP_SEED1234 --model cnnlstm --lr 3e-4 --optimizer adam --seed 1234 --mixup --fixed_size  --data_size 256 --batch_size 24 --data_path dataset/ICBHI_final_database/official

CUDA_VISIBLE_DEVICES=1 python train_with_diagnosis.py --prefix RESNET_T256_BS24_SGD0.01_MIXUP_SEED1234 --model resnet18 --lr 0.01 --optimizer sgd --seed 1234 --mixup --fixed_size  --data_size 256 --batch_size 24 --data_path dataset/ICBHI_final_database/official
CUDA_VISIBLE_DEVICES=1 python train_with_diagnosis.py --prefix RESNET_T256_BS24_ADAM3e-4_MIXUP_SEED1234 --model resnet18 --lr 3e-4 --optimizer adam --seed 1234 --mixup --fixed_size  --data_size 256 --batch_size 24 --data_path dataset/ICBHI_final_database/official

CUDA_VISIBLE_DEVICES=1 python train_with_diagnosis.py --prefix AST_T256_BS24_SGD0.01_MIXUP_SEED1234 --model ast --lr 0.01 --optimizer sgd --seed 1234 --mixup  --fixed_size  --data_size 256 --batch_size 24 --data_path dataset/ICBHI_final_database/official
CUDA_VISIBLE_DEVICES=1 python train_with_diagnosis.py --prefix AST_T256_BS24_ADAM1e-5_MIXUP_SEED1234 --model ast --lr 1e-5 --optimizer adam --seed 1234 --mixup  --fixed_size  --data_size 256 --batch_size 24 --data_path dataset/ICBHI_final_database/official

# CUDA_VISIBLE_DEVICES=1 python train.py --prefix AST_T256_BS24_SGD0.01_MIXUP_MULT_SEED5959 --model ast --lr 0.01 --optimizer sgd --seed 5959 --mixup --multi_label --fixed_size  --data_size 256 --batch_size 24 --data_path dataset/ICBHI_final_database/wav_16k/
# CUDA_VISIBLE_DEVICES=1 python train.py --prefix AST_T256_BS24_ADAM1e-5_MIXUP_MULT_SEED5959 --model ast --lr 1e-5 --optimizer adam --seed 5959 --mixup --multi_label --fixed_size  --data_size 256 --batch_size 24 --data_path dataset/ICBHI_final_database/wav_16k/

# CUDA_VISIBLE_DEVICES=1 python train.py --prefix LSTM_T256_BS24_SGD0.01_MIXUP_MULT_SEED5959 --model lstm --lr 0.01 --optimizer sgd --seed 5959 --mixup --multi_label --fixed_size  --data_size 256 --batch_size 24 --data_path dataset/ICBHI_final_database/wav_16k/
# CUDA_VISIBLE_DEVICES=1 python train.py --prefix LSTM_T256_BS24_ADAM1e-5_MIXUP_MULT_SEED5959 --model lstm --lr 1e-5 --optimizer adam --seed 5959 --mixup --multi_label --fixed_size  --data_size 256 --batch_size 24 --data_path dataset/ICBHI_final_database/wav_16k/

# CUDA_VISIBLE_DEVICES=1 python train.py --prefix CNNLSTM_T256_BS24_SGD0.01_MIXUP_MULT_SEED5959 --model cnnlstm --lr 0.01 --optimizer sgd --seed 5959 --mixup --multi_label --fixed_size  --data_size 256 --batch_size 24 --data_path dataset/ICBHI_final_database/wav_16k/
# CUDA_VISIBLE_DEVICES=1 python train.py --prefix CNNLSTM_T256_BS24_ADAM1e-5_MIXUP_MULT_SEED5959 --model cnnlstm --lr 1e-5 --optimizer adam --seed 5959 --mixup --multi_label --fixed_size  --data_size 256 --batch_size 24 --data_path dataset/ICBHI_final_database/wav_16k/

# CUDA_VISIBLE_DEVICES=1 python train.py --prefix RESNET_T256_BS24_SGD0.01_MIXUP_MULT_SEED5959 --model resnet18 --lr 0.01 --optimizer sgd --seed 5959 --mixup --multi_label --fixed_size  --data_size 256 --batch_size 24 --data_path dataset/ICBHI_final_database/wav_16k/
# CUDA_VISIBLE_DEVICES=1 python train.py --prefix RESNET_T256_BS24_ADAM1e-5_MIXUP_MULT_SEED5959 --model resnet18 --lr 1e-5 --optimizer adam --seed 5959 --mixup --multi_label --fixed_size  --data_size 256 --batch_size 24 --data_path dataset/ICBHI_final_database/wav_16k/

# CUDA_VISIBLE_DEVICES=1 python train.py --prefix AST_T256_BS24_SGD0.01_MIXUP_MULT_SEED9595 --model ast --lr 0.01 --optimizer sgd --seed 9595 --mixup --multi_label --fixed_size  --data_size 256 --batch_size 24 --data_path dataset/ICBHI_final_database/wav_16k/
# CUDA_VISIBLE_DEVICES=1 python train.py --prefix AST_T256_BS24_ADAM1e-5_MIXUP_MULT_SEED9595 --model ast --lr 1e-5 --optimizer adam --seed 9595 --mixup --multi_label --fixed_size  --data_size 256 --batch_size 24 --data_path dataset/ICBHI_final_database/wav_16k/

# CUDA_VISIBLE_DEVICES=1 python train.py --prefix LSTM_T256_BS24_SGD0.01_MIXUP_MULT_SEED9595 --model lstm --lr 0.01 --optimizer sgd --seed 9595 --mixup --multi_label --fixed_size  --data_size 256 --batch_size 24 --data_path dataset/ICBHI_final_database/wav_16k/
# CUDA_VISIBLE_DEVICES=1 python train.py --prefix LSTM_T256_BS24_ADAM1e-5_MIXUP_MULT_SEED9595 --model lstm --lr 1e-5 --optimizer adam --seed 9595 --mixup --multi_label --fixed_size  --data_size 256 --batch_size 24 --data_path dataset/ICBHI_final_database/wav_16k/

# CUDA_VISIBLE_DEVICES=1 python train.py --prefix CNNLSTM_T256_BS24_SGD0.01_MIXUP_MULT_SEED9595 --model cnnlstm --lr 0.01 --optimizer sgd --seed 9595 --mixup --multi_label --fixed_size  --data_size 256 --batch_size 24 --data_path dataset/ICBHI_final_database/wav_16k/
# CUDA_VISIBLE_DEVICES=1 python train.py --prefix CNNLSTM_T256_BS24_ADAM1e-5_MIXUP_MULT_SEED9595 --model cnnlstm --lr 1e-5 --optimizer adam --seed 9595 --mixup --multi_label --fixed_size  --data_size 256 --batch_size 24 --data_path dataset/ICBHI_final_database/wav_16k/

# CUDA_VISIBLE_DEVICES=1 python train.py --prefix RESNET_T256_BS24_SGD0.01_MIXUP_MULT_SEED9595 --model resnet18 --lr 0.01 --optimizer sgd --seed 9595 --mixup --multi_label --fixed_size  --data_size 256 --batch_size 24 --data_path dataset/ICBHI_final_database/wav_16k/
# CUDA_VISIBLE_DEVICES=1 python train.py --prefix RESNET_T256_BS24_ADAM1e-5_MIXUP_MULT_SEED9595 --model resnet18 --lr 1e-5 --optimizer adam --seed 9595 --mixup --multi_label --fixed_size  --data_size 256 --batch_size 24 --data_path dataset/ICBHI_final_database/wav_16k/