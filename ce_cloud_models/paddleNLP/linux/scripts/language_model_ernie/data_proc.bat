@echo off
cd ../..
cd models_repo\examples\language_model\ernie\


python -m wget https://paddlenlp.bj.bcebos.com/data/ernie_hybrid_parallelism_data.tar
tar -xvf ernie_hybrid_parallelism_data.tar

python -m wget https://paddlenlp.bj.bcebos.com/data/ernie_hybrid_parallelism-30k-clean.vocab.txt 


move /Y .\ernie_hybrid_parallelism-30k-clean.vocab.txt   .\config\vocab.txt

