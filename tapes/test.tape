
task test : hiertype
  < data=@
  < dump=$dump@cache
  < ontology=@
  < model=$out@train
  > out
  > metrics="metrics.txt"
  :: maxBranchingFactors=@ :: delta=@ :: strategies=@ :: otherDelta=@
  :: grid=@
  :: .submitter=$grid .action_flags=@ .resource_flags=@
  :: gpuId=@
{
  mkdir -p $out
  PYTHONPATH=$hiertype CUDA_VISIBLE_DEVICES=$gpuId \
  python $hiertype/hiertype/commands/run.py \
    --test $dump/test.db:$data/test.tsv \
    --model_dir $model \
    --out $out \
    --gpuid 0 \
    --strategies $strategies \
    --max_branching_factors $maxBranchingFactors \
    --delta $delta \
    --other_delta $otherDelta \
    > $metrics
}
