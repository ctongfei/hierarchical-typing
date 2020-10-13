task testHierarchical : hiertype
  < data=@
  < dump=$dump@cache
  < ontology=@
  < model=$out@train
  > out
  > metrics="metrics.txt"
  :: maxBranchingFactors=@ :: strategies=@ :: delta=@ :: otherDelta=@
  :: grid=@
  :: .submitter=$grid .action_flags=@ .resource_flags=@
  :: gpuId=@
{
  mkdir -p $out
  PYTHONPATH=$hiertype python $hiertype/hiertype/commands/run.py \
    --test $dump/test.db:$data/test.tsv \
    --model_dir $model \
    --out $out \
    --gpuid $gpuId \
    --strategies $strategies \
    --max_branching_factors $testBranch0 $testBranch1 1 \
    --delta $testDelta0 $testDelta1 0 \
    --other_delta $testTopOtherDelta \
    > $metrics
}
