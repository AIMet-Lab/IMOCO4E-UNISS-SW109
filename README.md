# IMOCO4E-UNISS-SW109

SW-109 is a verification tool designed to enhance the reliability of Neural Networks (NNs) developed using SW-107.

## Requirements
SW-109 requires [pyNeVer](https://github.com/NeVerTools/pyNeVer) and all its dependencies. We refer to its 
[Github Page](https://github.com/NeVerTools/pyNeVer) for more information regarding how to install them.

## How to use
The verification script can be executed launching [verification_script.py](verification_script.py) and requires four command line parameters:
- `--model_path`: Path to the ONNX file containing the Neural Network to verify.
An example can be found in [CDAD_ReLUNode_[32-8-32].onnx](models/CDAD_ReLUNode_[32-8-32].onnx).
- `--property_path`: Path to the .vnnlib file containing the property of interest. 
An example can be found in [CDAD_ReLUNode_[32-8-32]_e=0.1_d=0.2.vnnlib](properties/CDAD_ReLUNode_[32-8-32]_e=0.1_d=0.2.vnnlib).
- `--output_path`: Path to the Plain Text file that will contain the results of the verification process.
An example can be found in [output.csv](outputs/output.csv).
- `--config_path`: Path to the .ini file containing the configuration info for the script.
An example can be found in [default_config.ini](configs/default_config.ini).

The script to generate new properties can be executed launching [properties_gen.py](properties_gen.py) and requires three command line parameters:
- `--models_path`: Path to the folder containing the ONNX files to generate properties for.
- `--config_path`: Path to the .ini file containing the configuration info for the script. 
An example can be found in [default_config.ini](configs/default_config.ini).
- `--props_path`: Path to the folder in which save the properties.

## Important Notes

- If new variables are added in the config file, minor modification of the code may be needed
to manage them. The variable `ver_heuristic` should be one among *overapprox*, *mixed*, and *complete*.
- In the [data/](data/) folder, the .csv file containing the normalized dataset used in SW-107 should be provided.