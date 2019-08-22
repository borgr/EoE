# EoE
The project was written in python3.6

Create_db.py 
contains a code to create MAEGE lattices

compare_measures.py
runs comparisons and analysis, please use -h flag to find usage
Examples of current measures and ways to pass them to the evaluation scripts can be found in the code 
(e.g. lt which is run by calling an outside script)
utils.py contains among other things the way to compute partial ordering kendall tau.

If you use this repository in a research, please cite

@inproceedings{choshen2018automatic,
  title={Automatic Metric Validation for Grammatical Error Correction},
  author={Leshem Choshen and Omri Abend},
  booktitle={ACL},
  year={2018}
}
