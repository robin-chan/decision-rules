#!/bin/bash

#--------------------------------------------------------------------------------
# PRIORS
#--------------------------------------------------------------------------------

mkdir -p out/priors-array
cp {globals.py,labels.py} scripts-priors/

python3 scripts-priors/priors.py
python3 scripts-priors/gauss-smooth.py

#--------------------------------------------------------------------------------
# PREDICT
#--------------------------------------------------------------------------------

mkdir -p out/predictions
cp {globals.py,labels.py} scripts-predict/

python3 scripts-predict/predict.py -i data/INPUT -d B -g 10
python3 scripts-predict/predict.py -i data/INPUT -d ML -g 10

#--------------------------------------------------------------------------------
# GRAPHICS
#--------------------------------------------------------------------------------

mkdir -p out/graphics
cp {globals.py,labels.py} scripts-graphics/

python3 scripts-graphics/priors-heat.py
python3 scripts-graphics/merge.py
python3 scripts-graphics/tables.py
python3 scripts-graphics/cdf.py
python3 scripts-graphics/heat.py
python3 scripts-graphics/bar.py

#--------------------------------------------------------------------------------
# CLEAN UP
#--------------------------------------------------------------------------------

rm scripts-priors/{globals.py,labels.py}
rm scripts-predict/{globals.py,labels.py}
rm scripts-graphics/{globals.py,labels.py}
find . | grep -E "(__pycache__|\.pyc|\.pyo$)" | xargs rm -rf

printf '\e[1;34m%-6s\e[m' "Scripts succesfully executed!!!"; printf "\n"
