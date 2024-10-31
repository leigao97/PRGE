# Copyright (c) Qualcomm Innovation Center, Inc.
# All rights reserved
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import json
import os
import re
import sys
from multiprocessing.connection import Client

import numpy as np
import torch

from utils import (
    build_executorch_binary,
    make_output_dir,
    setup_common_args_and_variables,
    SimpleADB,
    parse_skip_delegation_node,
)


def create_device_inputs(example_inputs):
    # inputs = [inp.to(torch.int32) for inp in example_inputs]
    # input_list = ""

    # inputs[0].numpy().tofile(f"{args.artifact}/input_0_0.raw")
    # input_list = "input_0_0.raw"
    # input_list += "\n"

    inputs = [inp.to(torch.int32) for inp in example_inputs]
    input_list = ""
    for i, inp in enumerate(inputs):
        inp.numpy().tofile(f"{args.artifact}/input_{i}_0.raw")
        input_list += f"input_{i}_0.raw\n"

    return tuple(inputs), input_list


if __name__ == "__main__":
    parser = setup_common_args_and_variables()
    parser.add_argument(
        "-a",
        "--artifact",
        help="path for storing generated artifacts by this example.",
        default="./llama2",
        type=str,
    )

    parser.add_argument(
        "--checkpoint",
        help="Pass llama2 checkpoint.",
        default="models/demo_params.pth",
        # default="models/llama2_1b_params.pth",
    )

    parser.add_argument(
        "--params",
        help="Pass llama2 params json file.",
        default="models/demo_config.json",
        # default="models/llama2_1b_config.json",
    )

    args = parser.parse_args()

    # ensure the working directory exist.
    os.makedirs(args.artifact, exist_ok=True)

    from models.llama2 import ModelArgs, Llama2Model

    with open(args.params, "r") as f:
        params = json.loads(f.read())

    model_args: ModelArgs = ModelArgs(
        max_seq_len=128,
        max_batch_size=1,
        **params,
    )

    model = Llama2Model(model_args)
    # model.load_state_dict(torch.load(args.checkpoint), strict=False)

    dataset = torch.ones(1, 1, 64, dtype=torch.long)

    inputs, input_list = create_device_inputs(dataset)

    pte_filename = "llama2_qnn"

    build_executorch_binary(
        model.eval(),
        (inputs[0],),
        args.model,
        f"{args.artifact}/{pte_filename}",
        dataset=None,
        # custom_annotations=(),
        quant_dtype=None,
        shared_buffer=args.shared_buffer,
        skip_node_id_set=None,
        skip_node_op_set=None,
    )

    if args.compile_only:
        sys.exit(0)

    adb = SimpleADB(
        qnn_sdk=os.getenv("QNN_SDK_ROOT"),
        build_path=f"{args.build_folder}",
        pte_path=f"{args.artifact}/{pte_filename}.pte",
        workspace=f"/data/local/tmp/executorch/{pte_filename}",
        device_id=args.device,
        host_id=args.host,
        soc_model=args.model,
        shared_buffer=args.shared_buffer,
    )
    adb.push(inputs=inputs, input_list=input_list)
    adb.execute()

    # collect output data
    output_data_folder = f"{args.artifact}/outputs"
    make_output_dir(output_data_folder)

    output_raws = []

    def post_process():
        for f in sorted(
            os.listdir(output_data_folder), key=lambda f: int(f.split("_")[1])
        ):
            filename = os.path.join(output_data_folder, f)
            if re.match(r"^output_[0-9]+_[1-9].raw$", f):
                os.remove(filename)
            else:
                output = np.fromfile(filename, dtype=np.float32)
                output_raws.append(output)

    adb.pull(output_path=args.artifact, callback=post_process)

    x86_golden = model.eval()(inputs[0])
    device_output = torch.from_numpy(output_raws[0]).reshape(x86_golden.size())
    result = torch.all(torch.isclose(x86_golden, device_output, atol=1e-2)).tolist()

    if args.ip and args.port != -1:
        with Client((args.ip, args.port)) as conn:
            conn.send(
                json.dumps(
                    {
                        "is_close": result,
                    }
                )
            )
    else:
        print(f"is_close? {result}")
        print(f"x86_golden {x86_golden}")
        print(f"device_out {device_output}")