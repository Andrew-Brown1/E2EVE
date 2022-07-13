# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
# --------------------------------------------------------

import argparse
import json
from E2EVE.main_functions import main_train_function

def get_parser(**parser_kwargs):
    def str2bool(v):
        if isinstance(v, bool):
            return v
        if v.lower() in ("yes", "true", "t", "y", "1"):
            return True
        elif v.lower() in ("no", "false", "f", "n", "0"):
            return False
        else:
            raise argparse.ArgumentTypeError("Boolean value expected.")

    parser = argparse.ArgumentParser(**parser_kwargs)

    parser.add_argument(
        "--parameters",
        type=str,
        const=True,
        default="configs/FFHQ/train_block_edit.json",
        nargs="?",
        help="postfix for logdir",
    )

    return parser


if __name__ == '__main__':

    parser = get_parser()
    opt, unknown = parser.parse_known_args()

    with open(opt.parameters) as json_file:
        data = json.load(json_file)

    parser = get_parser()
    opt, unknown = parser.parse_known_args()

    main_train_function((data['E2EVE_config'], "local", {'model':data['model'], 'data':data['data']}))
