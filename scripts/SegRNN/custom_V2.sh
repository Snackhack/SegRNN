if [ ! -d "./logs" ]; then
    mkdir ./logs
fi

if [ ! -d "./logs/LongForecasting" ]; then
    mkdir ./logs/LongForecasting
fi
model_name=SegRNN

root_path_name=./dataset/
data_path_name=data.csv
model_id_name=customv2NOSCALE
data_name=Dataset_CustomV2
seq_len=720

for pred_len in 96 192 336 720
do
    python -u run_longExp_new.py \
      --is_training 1 \
      --root_path $root_path_name \
      --data_path $data_path_name \
      --model_id $model_id_name'_'$seq_len'_'$pred_len \
      --model $model_name \
      --data $data_name \
      --features 4 \
      --seq_len $seq_len \
      --pred_len $pred_len \
      --seg_len 48 \
      --enc_in 9 \
      --d_model 512 \
      --dropout 0.5 \
      --train_epochs 30 \
      --patience 10 \
      --rnn_type gru \
      --dec_way pmf \
      --channel_id 1 \
      --use_weather=False \
      --itr 1 --batch_size 256 --learning_rate 0.001 > logs/LongForecasting/$model_name'_'$model_id_name'_'$seq_len'_'$pred_len.log
done

