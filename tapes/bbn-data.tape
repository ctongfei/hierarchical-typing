global {
  bbnDataUrl="https://drive.google.com/file/d/0B2ke42d0kYFfdVk2ZkJ6TGRzR2M/view"
}

task bbnData : hiertype
  < bbnPath=@
  > out
{
  mkdir -p $out
  cat $bbnPath/train.json | python $hiertype/aux/scripts/bbn/bbn-jsonlines-to-tsv.py > $out/train-dev.tsv
  cat $bbnPath/test.json | python $hiertype/aux/scripts/bbn/bbn-jsonlines-to-tsv.py > $out/test.tsv
  python $hiertype/aux/scripts/bbn/bbn-gen-dev.py $out/train-dev.tsv $out/train.tsv $out/dev.tsv
}

plan BBNData {
  reach bbnData
}

task bbnOntology : hiertype
  < data=$out@bbnData
  > ontology="ontology.txt"
{
  cat $data/train.tsv $data/dev.tsv $data/test.tsv | python $hiertype/aux/scripts/bbn/bbn-ontology.py > $ontology
}

plan BBNOntology {
  reach bbnOntology
}

