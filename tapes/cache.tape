task cache : hiertype
  < data=@
  > dump
  :: contextualizer=@ :: unit=@ :: layers=@
  :: grid=@ :: .submitter=$grid .action_flags=@ .resource_flags=@
  :: gpuId=@
{
  mkdir -p $dump
  for partition in train dev test; do
    PYTHONPATH=$hiertype CUDA_VISIBLE_DEVICES=$gpuId \
    python $hiertype/hiertype/commands/cache_repr.py \
     --input $data/${partition}.tsv --output $dump/${partition}.db \
     --model $contextualizer \
     --unit $unit \
     --layers $layers
  done
}
