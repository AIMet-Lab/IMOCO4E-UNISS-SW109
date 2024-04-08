
import os
import argparse
import configparser
import random

import onnx
import torch
import numpy as np
import pynever.strategies.conversion as pyn_conv
import pynever.strategies.verification as pyn_ver

import datasets


def make_parser() -> argparse.ArgumentParser:

    parser = argparse.ArgumentParser("Properties Generation Script for SW-109")

    model_path_help = "Path to the folder containing the ONNX files to generate properties for."
    parser.add_argument("--models_path", type=str, help=model_path_help, default="models/")

    config_path_help = "Path to the .ini file containing the configuration info for the script."
    parser.add_argument("--config_path", type=str, help=config_path_help, default="configs/default_config.ini")

    prop_path_help = "Path to the folder in which save the properties."
    parser.add_argument("--props_path", type=str, help=prop_path_help, default="properties/")

    return parser


if __name__ == "__main__":

    # PARAMETERS ESTRACTION

    arg_parser = make_parser()
    args = arg_parser.parse_args()

    models_path = args.models_path
    config_path = args.config_path
    props_path = args.props_path

    config = configparser.ConfigParser()
    _ = config.read(config_path)

    dataset_path = config["PROP_GEN"]["dataset_path"]
    epsilons = config["PROP_GEN"]["epsilons"].strip().replace('[', '').replace(']', '').split(',')
    epsilons = [float(epsilon) for epsilon in epsilons]
    deltas = config["PROP_GEN"]["deltas"].strip().replace('[', '').replace(']', '').split(',')
    deltas = [float(delta) for delta in deltas]

    in_low_b = [-1 for i in range(8)]
    in_upp_b = [1 for i in range(8)]

    # Get random sample over which build the local robustness properties.
    dataset = datasets.ComponentDegradationAD(dataset_path, is_training=False)
    sample_index = random.randrange(dataset.__len__())
    ver_sample = dataset.__getitem__(sample_index)[0]

    # The local robustness property must be tailored for each model.
    onnx_ids = os.listdir(models_path)
    for onnx_id in onnx_ids:

        net_id = onnx_id.replace(".onnx", "")
        onnx_net = pyn_conv.ONNXNetwork(net_id, onnx.load(models_path + onnx_id))
        pyn_net = pyn_conv.ONNXConverter().to_neural_network(onnx_net)
        pyt_net = pyn_conv.PyTorchConverter().from_neural_network(pyn_net).pytorch_network

        tensor_input = torch.from_numpy(ver_sample)
        tensor_input.to("cpu").double()
        pyt_net.to("cpu").double()
        numpy_output = pyt_net(tensor_input).detach().numpy()

        for i in range(len(epsilons)):

            epsilon = epsilons[i]
            delta = deltas[i]

            in_pred_mat = []
            in_pred_bias = []
            data_size = len(ver_sample)

            for j in range(len(ver_sample)):

                lb_constraint = np.zeros(data_size)
                ub_constraint = np.zeros(data_size)
                lb_constraint[j] = -1
                ub_constraint[j] = 1
                in_pred_mat.append(lb_constraint)
                in_pred_mat.append(ub_constraint)

                if ver_sample[j] - epsilon < in_low_b[j]:
                    in_pred_bias.append([-in_low_b[j]])
                else:
                    in_pred_bias.append([-(ver_sample[j] - epsilon)])

                if ver_sample[j] + epsilon > in_upp_b[j]:
                    in_pred_bias.append([in_upp_b[j]])
                else:
                    in_pred_bias.append([ver_sample[j] + epsilon])

            in_pred_bias = np.array(in_pred_bias)
            in_pred_mat = np.array(in_pred_mat)

            out_pred_mat = []
            out_pred_bias = []
            out_size = len(numpy_output)

            # As usual we need to define the negation of the output property we want:
            # In this case the safe property is that the output do not present a deviation greater than delta.
            # The corresponding property is that at least one output variable present greater deviation.
            for j in range(out_size):
                lb_constraint = np.zeros(out_size)
                ub_constraint = np.zeros(out_size)
                lb_constraint[j] = 1
                ub_constraint[j] = -1
                out_pred_mat.append(np.array([lb_constraint]))
                out_pred_mat.append(np.array([ub_constraint]))

                out_pred_bias.append(np.array([[(numpy_output[j] - delta)]]))
                out_pred_bias.append(np.array([[-(numpy_output[j] + delta)]]))

            in_prop = pyn_ver.NeVerProperty(in_pred_mat, in_pred_bias, out_pred_mat, out_pred_bias)
            in_prop.to_smt_file(filepath=props_path + net_id + f"_e={epsilon}_d={delta}.vnnlib")
