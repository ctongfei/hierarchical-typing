task train : hiertype
  < data=@
  < dump=$dump@cache
  < ontology=@
  > out
  :: contextualizer=@ :: inputDim=@ :: typeDim=@ :: bottleneckDim=@ :: embDropoutRate=@ :: dropoutRate=@
  :: margins=@ :: thresholdRatio=@ :: relationConstraintCoef=@ :: numNegativeSamples=@
  :: withOther=@ :: liftOther=@
  :: maxBranchingFactors=@ :: strategies=@ :: delta=@ :: otherDelta=@
  :: regularizer=@ :: batchSize=@ :: devBatchSize=@ :: numEpochs=@
  :: grid=@
  :: .submitter=$grid .action_flags=@ .resource_flags=@
  :: gpuId=@
{
  mkdir -p $out
  PYTHONPATH=$hiertype CUDA_VISIBLE_DEVICES=$gpuId \
  python $hiertype/hiertype/commands/train.py \
    --train $dump/train.db:$data/train.tsv \
    --dev $dump/dev.db:$data/dev.tsv \
    --ontology $ontology \
    --out $out \
    --contextualizer $contextualizer --input_dim $inputDim --type_dim $typeDim --bottleneck_dim $bottleneckDim \
    --emb_dropout_rate $embDropoutRate --dropout_rate $dropoutRate --margins $margins --threshold_ratio $thresholdRatio \
    --with_other $withOther --lift_other $liftOther \
    --relation_constraint_coef $relationConstraintCoef \
    --num_negative_samples $numNegativeSamples \
    --max_branching_factors $maxBranchingFactors --delta $delta --other_delta $otherDelta --strategies $strategies \
    --regularizer $regularizer \
    --dev_metric +O_MiF \
    --batch_size $batchSize --dev_batch_size $devBatchSize --num_epochs $numEpochs
}

plan TrainBBN {
  reach train via (Dataset: BBN) * (Contextualizer: elmo-original) * (EmbDropout: 0.6) * (ThresholdRatio: 0.2)
    * (RelConsCoef: 0.1) * (Regularizer: 0.003) * (MarginMode: graded) * (WithOther: True) * (LiftOther: True)
}

plan TrainAida {
  reach train via (Dataset: Aida) * (Contextualizer: elmo-original) * (EmbDropout: 0.6) * (ThresholdRatio: 0.1)
    * (RelConsCoef: 0.3) * (Regularizer: 0.1) * (MarginMode: graded) * (WithOther: True) * (LiftOther: False)
}

plan TrainFiger {
  reach train via (Dataset: Figer) * (Contextualizer: elmo-original) * (EmbDropout: 0.5) * (Dropout: 0.1 0.2 0.3) *
  (ThresholdRatio: 0.2) * (RelConsCoef: 0.1) * (Regularizer: 0.0001) * (WithOther: True) * (LiftOther: False)
}

plan TrainOntoNotes {
  reach train via (Dataset: OntoNotes) * (Contextualizer: elmo-original) * (EmbDropout: 0.6) * (Dropout: 0.1) * (ThresholdRatio: 0.15)
    * (RelConsCoef: 0.1) * (Regularizer: 0.001) * (MarginMode: graded) * (WithOther: False)
}
