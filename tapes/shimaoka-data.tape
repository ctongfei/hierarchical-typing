global {
  shimaokaDataUrl="http://www.cl.ecei.tohoku.ac.jp/~shimaoka/corpus.zip"
  shimaokaDataset=(ShimaokaDataset: Figer OntoNotes)
}

task shimaokaDataDownload : hiertype
  :: shimaokaDataUrl=@
  > corpus
{
  wget $shimaokaDataUrl
  unzip corpus.zip
  rm corpus.zip
}

plan ShimaokaDataDownload {
  reach shimaokaDataDownload
}

task shimaokaData : hiertype
  < data=$corpus@shimaokaDataDownload
  > out
  :: dir=(ShimaokaDataset: Figer=Wiki OntoNotes=OntoNotes)
{
  mkdir -p $out
  java -cp $hiertype/aux/target/*/*.jar hiertype.PreprocessShimaokaData $data/$dir/train.txt \
    | sed 's/geograpy/geography/g' | sed 's|livingthing|living_thing|g' > $out/train.tsv
  java -cp $hiertype/aux/target/*/*.jar hiertype.PreprocessShimaokaData $data/$dir/dev.txt \
    | sed 's/geograpy/geography/g' | sed 's|livingthing|living_thing|g' > $out/dev.tsv
  java -cp $hiertype/aux/target/*/*.jar hiertype.PreprocessShimaokaData $data/$dir/test.txt \
    | sed 's/geograpy/geography/g' | sed 's|livingthing|living_thing|g' > $out/test.tsv
}

task shimaokaOntology : hiertype
  < data=$out@shimaokaData
  > ontology="ontology.txt"
{
  java -cp $hiertype/aux/target/*/*.jar hiertype.GetHierarchy $data/train.tsv > $ontology
}


plan PreprocessShimaokaData {
  reach shimaokaData via (ShimaokaDataset: *)
  reach shimaokaOntology via (ShimaokaDataset: *)
}


task ontoNotesOtherRemovedData : hiertype
  < data=$out@shimaokaData[ShimaokaDataset:OntoNotes]
  > out
{
  mkdir -p $out
  cat $data/train.tsv | python $hiertype/aux/scripts/ontonotes/remove_other.py > $out/train.tsv
  cat $data/dev.tsv | python $hiertype/aux/scripts/ontonotes/remove_other.py > $out/dev.tsv
  cat $data/test.tsv | python $hiertype/aux/scripts/ontonotes/remove_other.py > $out/test.tsv
}

task ontoNotesOtherRemovedOntology
  < original=$ontology@shimaokaOntology[ShimaokaDataset:OntoNotes]
  > ontology="ontology.txt"
{
  cat $original | sed -e 's|/other||g' > $ontology
}
