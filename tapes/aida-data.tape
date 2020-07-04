global {
  aidaPartition=(AidaPartition: practice eval)
  aidaLang=(AidaLang: eng rus ukr all)
  aidaTask=(AidaTask: entity event relation)
  aidaSheetName=(AidaTask: entity=entities event=events relation=relations)
  aidaLdcDataIdentifier=(AidaTask: entity=arg event=evt relation=rel)
}

task aidaLtfToConcrete : concreteAida
  < ltf=(AidaPartition: practice=$aidaPracticeSourcePath eval=$aidaEvalSourcePath)
  > concrete
{
  for z in $(ls $ltf); do
    unzip $ltf/$z -d .
  done
  ls -1 ltf/* > ltf-filelist
  mkdir -p $concrete
  PYTHONPATH=$concreteAida python $concreteAida/aida2concrete.py \
    --filelist ltf-filelist --output-dir $concrete
}

plan AidaLtfToConcrete {
  reach aidaLtfToConcrete via (AidaPartition: practice eval)
}


task aidaOntology : hiertype
  < data=$aidaPracticeAnnotationsPath
  > ontology="ontology.txt"
  :: aidaSheetName=@
{
  python $hiertype/aux/scripts/aida/get-aida-ontology.py \
    --path $data --sheet $aidaSheetName > $ontology
}

plan AidaOntology {
  reach aidaOntology via (AidaTask: entity)
}


task aidaMergeLdcTsvs : hiertype
  < aidaPracticeAnnotationsPath=@
  < aidaEvalAnnotationsPath=@
  > out
  :: aidaLdcDataIdentifier=@
{
  mkdir -p $out
  python $hiertype/aux/scripts/aida/merge-tsvs.py \
    $aidaPracticeAnnotationsPath/data/R103/R103_${aidaLdcDataIdentifier}_mentions.tab \
    $aidaPracticeAnnotationsPath/data/R107/R107_${aidaLdcDataIdentifier}_mentions.tab > $out/train.tsv

  python $hiertype/aux/scripts/aida/merge-tsvs.py \
    $aidaPracticeAnnotationsPath/data/R105/R105_${aidaLdcDataIdentifier}_mentions.tab > $out/dev.tsv

  python $hiertype/aux/scripts/aida/merge-tsvs.py \
    $aidaEvalAnnotationsPath/data/E101/E101_${aidaLdcDataIdentifier}_mentions.tab \
    $aidaEvalAnnotationsPath/data/E102/E102_${aidaLdcDataIdentifier}_mentions.tab \
    $aidaEvalAnnotationsPath/data/E103/E103_${aidaLdcDataIdentifier}_mentions.tab > $out/test.tsv

}

plan AidaMergeLdcTsvs {
  reach aidaMergeLdcTsvs via (AidaLang: eng) * (AidaTask: entity)
}


task aidaData : hiertype
  < tsv=$out@aidaMergeLdcTsvs
  < practiceConcrete=$concrete@aidaLtfToConcrete[AidaPartition:practice]
  < evalConcrete=$concrete@aidaLtfToConcrete[AidaPartition:eval]
  > out
  :: aidaLang=@
{
  mkdir -p $out
#  python $hiertype/aux/scripts/aida/resolve-token-index.py \
#    --tsv $tsv/train.tsv --lang $aidaLang --concrete_dir $practiceConcrete | grep -Pv "[ЁёА-я]" > $out/train.tsv
#  python $hiertype/aux/scripts/aida/resolve-token-index.py \
#    --tsv $tsv/dev.tsv --lang $aidaLang --concrete_dir $practiceConcrete | grep -Pv "[ЁёА-я]" > $out/dev.tsv
#  python $hiertype/aux/scripts/aida/resolve-token-index.py \
#    --tsv $tsv/test.tsv --lang $aidaLang --concrete_dir $evalConcrete | grep -Pv "[ЁёА-я]" > $out/test.tsv
  python $hiertype/aux/scripts/aida/resolve-token-index.py \
    --tsv $tsv/train.tsv --lang $aidaLang --concrete_dir $practiceConcrete > $out/train.tsv
  python $hiertype/aux/scripts/aida/resolve-token-index.py \
    --tsv $tsv/dev.tsv --lang $aidaLang --concrete_dir $practiceConcrete > $out/dev.tsv
  python $hiertype/aux/scripts/aida/resolve-token-index.py \
    --tsv $tsv/test.tsv --lang $aidaLang --concrete_dir $evalConcrete > $out/test.tsv
}

plan PreprocessAidaData {
  reach aidaData via (AidaLang: eng all) * (AidaTask: entity)
  reach aidaOntology via (AidaTask: entity)
}
