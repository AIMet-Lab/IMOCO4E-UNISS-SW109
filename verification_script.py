import argparse
import os
import time
import configparser

import pynever.networks as pyn_networks
import pynever.strategies.conversion as pyn_conv
import pynever.strategies.verification as pyn_ver
import onnx

import utilities


def make_parser() -> argparse.ArgumentParser:

    parser = argparse.ArgumentParser("Verification Script for SW-109")

    model_path_help = "Path to the ONNX file containing the Neural Network to verify."
    parser.add_argument("--model_path", type=str, help=model_path_help, default="models/CDAD_ReLUNode_[32-8-32].onnx")

    property_path_help = "Path to the .vnnlib file containing the property of interest."
    parser.add_argument("--property_path", type=str, help=property_path_help,
                        default="properties/CDAD_ReLUNode_[32-8-32]_e=0.1_d=0.2.vnnlib")

    output_path_help = "Path to the Plain Text file that will contain the results of the verification process."
    parser.add_argument("--output_path", type=str, help=output_path_help, default="outputs/output.csv")

    config_path_help = "Path to the .ini file containing the configuration info for the script."
    parser.add_argument("--config_path", type=str, help=config_path_help, default="configs/default_config.ini")

    return parser


if __name__ == "__main__":

    # Parse the command line parameters.
    arg_parser = make_parser()
    args = arg_parser.parse_args()

    model_path = args.model_path
    model_id = model_path.split('/')[-1].replace('.onnx', '')

    property_path = args.property_path
    property_id = property_path.split('/')[-1].replace('.vnnlib', '')

    output_path = args.output_path
    config_path = args.config_path

    # Extract configuration info.
    config = configparser.ConfigParser()
    _ = config.read(config_path)

    ver_heuristic = config["VERIFICATION"]["ver_heuristic"]
    verbose = config["VERIFICATION"].getboolean("verbose")

    # Instantiate loggers.
    if not os.path.exists(output_path):
        stream_log, file_log = utilities.instantiate_logger(output_path)
        file_log.info("MODEL_PATH,PROPERTY_PATH,VER_HEURISTIC,IS_VERIFIED,TIME")
    else:
        stream_log, file_log = utilities.instantiate_logger(output_path)

    # Load the model in the pynever format.
    onnx_net = pyn_conv.ONNXNetwork(model_id, onnx.load(model_path))
    pyn_net = pyn_conv.ONNXConverter().to_neural_network(onnx_net)

    if not isinstance(pyn_net, pyn_networks.SequentialNetwork):
        raise NotImplementedError

    # Now we load the property of interest.
    never_prop = pyn_ver.NeVerProperty()
    never_prop.from_smt_file(property_path)

    stream_log.info(f"Verifying Model {model_path} and Property {property_path}...")

    # We prepare the verification strategy with the verification heuristic chosen.
    ver_strategy = pyn_ver.NeverVerification(heuristic=ver_heuristic)

    start = time.perf_counter()
    is_verified = ver_strategy.verify(pyn_net, never_prop)
    end = time.perf_counter()

    stream_log.info(f"Verification Result: {is_verified}")
    file_log.info(f"{model_path},{property_path},{ver_heuristic},{is_verified},{end - start}")