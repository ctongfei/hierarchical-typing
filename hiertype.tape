global {
  ducttape_output="out"
  ducttape_experimental_packages=true
  ducttape_experimental_submitters=true
  ducttape_experimental_imports=true
  ducttape_experimental_multiproc=true
}

import "tapes/sge.tape"
import "tapes/env.tape"

import "tapes/params.tape"

import "tapes/shimaoka-data.tape"
import "tapes/aida-data.tape"
import "tapes/bbn-data.tape"

import "tapes/cache.tape"
import "tapes/train.tape"
import "tapes/test-cached.tape"
