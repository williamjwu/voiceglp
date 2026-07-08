#!/bin/zsh
# Staged LS-100h model scoring: one shard on disk at a time (purge rule).
cd /Users/williamjwu/workspace/voiceglp
export HF_HOME=$PWD/data/hf_cache
for i in $(seq 0 13); do
  n=$(printf "%04d" $i)
  out=results/model_scores_ls100_shard$i.parquet
  [ -f $out ] && { echo "shard $i already done"; continue; }
  echo "[$(date +%H:%M:%S)] shard $i: downloading"
  curl -sL --retry 3 -o data/ls_shard_tmp.parquet "https://huggingface.co/datasets/openslr/librispeech_asr/resolve/refs%2Fconvert%2Fparquet/clean/train.100/$n.parquet" || { echo "shard $i DOWNLOAD FAILED"; exit 1; }
  echo "[$(date +%H:%M:%S)] shard $i: scoring"
  .venv/bin/python scripts/01_score.py --from-parquet data/ls_shard_tmp.parquet \
    --text-col text --model-scores --out $out || { echo "shard $i SCORING FAILED"; exit 1; }
  rm -f data/ls_shard_tmp.parquet
done
echo "PIPELINE DONE: all 14 shards model-scored"
