This repository contains code for the following paper:
 - Tongfei Chen, Yunmo Chen, Benjamin Van Durme (2020): [Hierarchical Entity Typing via Multi-level Learning to Rank](https://www.aclweb.org/anthology/2020.acl-main.749/). In _Proceedings of ACL_.

 ```bibtex
@inproceedings{ChenCD20,
  author    = {Tongfei Chen and
               Yunmo Chen and
               Benjamin {Van Durme}},
  title     = {Hierarchical Entity Typing via Multi-level Learning to Rank},
  booktitle = {Proceedings of the 58th Annual Meeting of the Association for Computational
               Linguistics, {ACL} 2020, Online, July 5-10, 2020},
  pages     = {8465--8475},
  year      = {2020},
  url       = {https://www.aclweb.org/anthology/2020.acl-main.749/}
}
 ```

 ### Setup

This repository uses [Ducttape](https://github.com/jhclark/ducttape) to manage intermediate results 
of the experiment pipeline.

To run a portion of the pipeline, first clone this repository to your location, then in `tapes/env.tape`,
modify various paths to point to various datasets or packages.

Then use the following command:

  ```bash
    ducttape hiertype.tape -p <TASK>
  ```
  where `<TASK>` is any of the plans defined in the Ducttape scripts.

One can easily execute different tasks by modifying the plans in the tape files.
