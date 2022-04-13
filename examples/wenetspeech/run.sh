#!/bin/bash

# We borrowed the data preparation part of wenet's recipe
# The major difference is the train stage.

. ./path.sh || exit 1;

# specify how many gpus each node, this is a 1-node example
export CUDA_VISIBLE_DEVICES="0,1,2,3,4,5,6,7"
world_size=`echo $CUDA_VISIBLE_DEVICES | sed -e 's/,/\n/g' | wc -l`
stage=0
stop_stage=5

# num of nodes
num_nodes=1
# rank of the node
node_rank=0
# master addr, specify ip of node with rank=0 if num_nodes > 1
MASTER_ADDR=
# master port, specify a free port of node with rank=0
MASTER_PORT=32666

# Use your own data path. You need to download the WenetSpeech dataset by yourself.
wenetspeech_data_dir=/data/asr_data/open_dataset/wenetspeech
# Make sure you have 1.2T for ${shards_dir}
shards_dir=/data/asr_data/open_dataset/wenetspeech_shards

# WenetSpeech training set
set=L
train_set=train_`echo $set | tr 'A-Z' 'a-z'`
dev_set=dev
test_sets="test_net test_meeting"
cmvn=true
decoding_mode="attention_rescoring"
# job dir
dir=exp/conformer_moe
config=conf/moe_config.yaml

. tools/parse_options.sh || exit 1;

set -u
set -o pipefail

# Data download
if [ ${stage} -le -1 ] && [ ${stop_stage} -ge -1 ]; then
    echo "Please follow https://github.com/wenet-e2e/WenetSpeech to download the data."
    exit 0;
fi

if [ ${stage} -le 0 ] && [ ${stop_stage} -ge 0 ]; then
  echo "Data preparation"
  local/wenetspeech_data_prep.sh \
    --train-subset $set \
    $wenetspeech_data_dir \
    data || exit 1;
fi

dict=data/dict/lang_char.txt
if [ ${stage} -le 1 ] && [ ${stop_stage} -ge 1 ]; then
    echo "Make a dictionary"
    echo "dictionary: ${dict}"
    mkdir -p $(dirname $dict)
    echo "<blank> 0" > ${dict} # 0 will be used for "blank" in CTC
    echo "<unk> 1" >> ${dict} # <unk> must be 1
    echo "▁ 2" >> ${dict} # ▁ is for space
    tools/text2token.py -s 1 -n 1 --space "▁" data/${train_set}/text \
        | cut -f 2- -d" " | tr " " "\n" \
        | sort | uniq | grep -a -v -e '^\s*$' \
        | grep -v "▁" \
        | awk '{print $0 " " NR+2}' >> ${dict} \
        || exit 1;
    num_token=$(cat $dict | wc -l)
    echo "<sos/eos> $num_token" >> $dict
fi

if [ ${stage} -le 2 ] && [ ${stop_stage} -ge 2 ]; then
  echo "Compute cmvn"
  # Here we use all the training data, you can sample some some data to save time
  # BUG!!! We should use the segmented data for CMVN
  if $cmvn; then
      python3 tools/compute_cmvn_stats.py \
      --num_workers 16 \
      --train_config $train_config \
      --in_scp data/$train_set/wav.scp \
      --out_cmvn data/$train_set/global_cmvn \
      || exit 1;
  fi
fi

if [ ${stage} -le 3 ] && [ ${stop_stage} -ge 3 ]; then
  echo "Making shards, please wait..."
  RED='\033[0;31m'
  NOCOLOR='\033[0m'
  echo -e "It requires ${RED}1.2T ${NOCOLOR}space for $shards_dir, please make sure you have enough space"
  echo -e "It takes about ${RED}12 ${NOCOLOR}hours with 32 threads"
  for x in $dev_set $test_sets ${train_set}; do
    dst=$shards_dir/$x
    mkdir -p $dst
    tools/make_shard_list.py --resample 16000 --num_utts_per_shard 1000 \
      --num_threads 32 --segments data/$x/segments \
      data/$x/wav.scp data/$x/text \
      $(realpath $dst) data/$x/data.list
  done
fi

if [ ${stage} -le 4 ] && [ ${stop_stage} -ge 4 ]; then
    echo "Start training"
    # data info, you can also specify your prepared data path here
    # and run this stage directly.
    train_data=data/$train_set/data.list
    cv_data=data/$dev_set/data.list
    data_type="shard"
    symbol_table=$dict
    $cmvn && cmvn_file=data/$train_set/global_cmvn
    # train embedding network first
    echo "Start training embedding network"
    embed_dir=exp/conformer_embedding
    mkdir -p $embed_dir/logs
    log_file=$embed_dir/logs/train.WORKER-ID.log
    # config file for embedding network
    embed_config=conf/embedding_config.yaml
    cp $embed_config $embed_dir
    python -m torch.distributed.launch \
        --nnodes=$num_nodes \
        --node_rank=$node_rank \
        ${MASTER_ADDR:+--master_addr "$MASTER_ADDR"} \
        --master_port=$MASTER_PORT \
        --nproc_per_node $world_size \
        trainer/train.py \
        --output_dir $embed_dir \
        --config $embed_config \
        --train_data $train_data \
        --cv_data $cv_data \
        --log_file $log_file \
        --data_type $data_type \
        --symbol_table $symbol_table \
        ${cmvn_file:+--cmvn_file $cmvn_file} \
        --num_workers 8 \
        --pin_memory
    wait
    # strip the encoder for embedding network
    if [ $node_rank == 0 ]; then
        python tools/strip_encoder.py \
            -i $embed_dir/final.nnet \
            -o $embed_dir/final.encoder
    else
        while [ ! -f $embed_dir/final.encoder ]; do
            sleep 2
        done
    fi
    # train base model
    echo "Start training base conformer model"
    base_dir=exp/conformer_base
    mkdir -p $base_dir/logs
    log_file=$base_dir/logs/train.WORKER-ID.log
    base_config=conf/base_config.yaml
    cp $base_config $base_dir
    python -m torch.distributed.launch \
        --nnodes=$num_nodes \
        --node_rank=$node_rank \
        ${MASTER_ADDR:+--master_addr "$MASTER_ADDR"} \
        --master_port=$MASTER_PORT \
        --nproc_per_node $world_size \
        trainer/train.py \
        --output_dir $base_dir \
        --config $base_config \
        --train_data $train_data \
        --cv_data $cv_data \
        --log_file $log_file \
        --data_type $data_type \
        --symbol_table $symbol_table \
        ${cmvn_file:+--cmvn_file $cmvn_file} \
        --num_workers 8 \
        --pin_memory
    wait
    # train conformer-moe
    echo "Start training conformer-moe initialized with embedding and base model"
    moe_dir=$dir
    mkdir -p $moe_dir/logs
    log_file=$moe_dir/logs/train.WORKER-ID.log
    moe_config=$config
    cp $moe_config $moe_dir
    # init part
    init_embed_model=$embed_dir/final.encoder
    init_experts_from_base=$base_dir/final.nnet
    # the number of experts in MoE model equals the number of
    # `expert_world_size` multiplied by `num_experts` in moe_config
    expert_world_size=$world_size \
        python -m torch.distributed.launch \
        --nnodes=$num_nodes \
        --node_rank=$node_rank \
        ${MASTER_ADDR:+--master_addr "$MASTER_ADDR"} \
        --master_port=$MASTER_PORT \
        --nproc_per_node $world_size \
        trainer/train.py \
        --output_dir $moe_dir \
        --config $moe_config \
        --train_data $train_data \
        --cv_data $cv_data \
        --init_embed_model $init_embed_model \
        --init_experts_from_base $init_experts_from_base \
        --log_file $log_file \
        --data_type $data_type \
        --symbol_table $symbol_table \
        ${cmvn_file:+--cmvn_file $cmvn_file} \
        --num_workers 8 \
        --pin_memory
    wait
fi

# test model with node 0
if [ ${stage} -le 5 ] && [ ${stop_stage} -ge 5 ] && [ $node_rank -eq 0 ]; then
    echo "Test model"
    # default config is for a full-chunk model
    ctc_weight=0.5
    reverse_weight=0.0
    beam_size=10
    # test data directory
    test_dir=data
    data_type="shard"
    decode_model=$dir/final.nnet
    model_name=$(basename ${decode_model})
    model_dir=$(dirname ${decode_model})
    result_dir=${model_dir}/results_${model_name}_${decoding_mode}
    mkdir -p $result_dir
    $cmvn && cmvn_file=data/$train_set/global_cmvn
    symbol_table=$dict
    for testset in ${test_sets} ${dev_set}; do
        test_data=${test_dir}/${testset}/data.list
        decode_out=${result_dir}/${testset}.out
        python -m torch.distributed.launch \
            --master_port $MASTER_PORT \
            --nproc_per_node $world_size \
            bin/recognize.py \
            --cuda \
            --config $config \
            --output_file $decode_out \
            --load_path $decode_model \
            --test_data $test_data \
            --data_type $data_type \
            ${cmvn_file:+--cmvn_file $cmvn_file} \
            --symbol_table $symbol_table \
            --beam_size $beam_size \
            --ctc_weight $ctc_weight \
            --reverse_weight $reverse_weight \
            --mode $decoding_mode
        python tools/compute-wer.py --char=1 --v=1 \
            ${test_dir}/${testset}/text ${decode_out} > ${result_dir}/${testset}.wer
    done
    wait
fi
