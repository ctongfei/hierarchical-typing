package hiertype :: .versioner=disk .path="/home/tongfei/proj/hierarchical-typing-release/" .rev=NULL {
    cd aux
    sbt clean compile assembly
}

package concreteAida :: .versioner=disk .path="/home/tongfei/proj/concrete-aida" .rev=NULL {}

global {

  grid="shell"
  gpuId="0"

  .action_flags=(Device:
    GPU=""
    CPU=""
  )

  .resource_flags=(Device:
    GPU=""
    CPU=""
  )

}


global {

  aidaLtfPath="/media/tongfei/dump/data/aida2019/Phase_1_Eval_Practice"
  aidaLdcData="/media/tongfei/dump/data/aida2019/LDC2019E07_AIDA_Phase_1_Evaluation_Practice_Topic_Annotations_V8.0"

  aidaPracticeSourcePath="/media/tongfei/dump/data/aida/aida-practice-source"
  aidaPracticeAnnotationsPath="/media/tongfei/dump/data/aida/aida-practice-annotations"
  aidaEvalSourcePath="/media/tongfei/dump/data/aida/aida-eval-source"
  aidaEvalAnnotationsPath="/media/tongfei/dump/data/aida/aida-eval-annotations"

  ultrafineCleanedPath="/media/tongfei/dump/data/cleaned-ultrafine/cleaned_data"

  bbnPath="/media/tongfei/dump/data/bbn"

}