global {

  dataset=(Dataset: Aida Figer OntoNotes BBN)
  partition=(Partition: train dev test)

  unit=(Unit: subword word span sentence)

  data=(Dataset:
    Aida=$out@aidaData
    Figer=$out@shimaokaData[ShimaokaDataset:Figer]
    OntoNotes=$out@shimaokaData[ShimaokaDataset:OntoNotes]
    BBN=$out@bbnData
  )

  ontology=(Dataset:
    Aida=$ontology@aidaOntology
    Figer=$ontology@shimaokaOntology[ShimaokaDataset:Figer]
    OntoNotes=$ontology@shimaokaOntology[ShimaokaDataset:OntoNotes]
    BBN=$ontology@bbnOntology
  )

  numLevels=(Dataset: Aida=3 Figer=2 OntoNotes=3 BBN=2)
  multiLabel=(Dataset: Aida=False Figer=True OntoNotes=True BBN=True)
  forceOther=(Dataset: Aida=False Figer=False OntoNotes=True BBN=False)

  thresholdRatio=(ThresholdRatio: 0.05 0.1 0.15 0.2 0.25)
  metric="+O_MiF"

  contextualizer=(Contextualizer: elmo-original bert-base-cased xlm-roberta-base)

  layers=(Contextualizer:
    elmo-original="[0,1,2]"
    bert-base-cased="[0,10,11,12]"
    xlm-roberta-base="[0,10,11,12]"
  )

  inputDim=3072
  typeDim=1024

  embDropoutRate=(EmbDropout: 0.0 0.1 0.2 0.3 0.4 0.5 0.6)
  dropoutRate=(Dropout: 0.0 0.1 0.2 0.3)

  margins=(MarginMode:
    graded=(Dataset:
      Aida="[3,2,1]" Figer="[2,1]" OntoNotes="[3,2,1]" BBN="[2,1]"
    )
    flat=(Dataset:
      Aida="[1,1,1]" Figer="[1,1]" OntoNotes="[1,1,1]" BBN="[1,1]"
    )
  )

  relationConstraintCoef=(RelConsCoef: 0.0 0.01 0.05 0.1 0.2 0.3)

  bottleneckDim=(BottleneckDim: 0 64 128 256)

  maxBranchingFactors=(Dataset:
    Aida="[1,1,1]"
    Figer="[2,1]"
    OntoNotes="[1,1,1]"
    BBN="[1,1]"
  )

  batchSize=256
  devBatchSize=256

  numNegativeSamples=0

  numEpochs=(Dataset: Aida=20 Figer=5 OntoNotes=5 BBN=5)

  strategies=(Dataset:
    Aida="[top,none,none]"
    Figer="[top,none]"
    OntoNotes="[other,none,none]"
    BBN="[other,none]"
  )

  regularizer=(Regularizer: 0.0001 0.0003 0.001 0.003 0.01 0.03 0.1 0.3)

  delta=(Dataset:
    Aida="[0,0,0]" Figer="[3,1]" OntoNotes="[0,1,2.5]" BBN="[0,0]"
  )
  otherDelta=(Dataset:
    Aida=0 Figer=0 OntoNotes=2 BBN=0
  )

  withOther=(WithOther: False True)
  liftOther=(LiftOther: False True)

}
