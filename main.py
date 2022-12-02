"""
Copyright (c) 2021-2022 UCLouvain, ICTEAM
Licensed under GPL-3.0 [see LICENSE for details]
Written by Jonathan Samelson (2021-2022)
"""

import tracker_client_video as tcv
import tracker_client_online as tco

import argparse
import os
import logging
import coloredlogs

if __name__ == "__main__":
    logging.getLogger("matplotlib").setLevel(logging.WARNING)
    logging.getLogger("tensorflow").setLevel(logging.WARNING)
    logging.getLogger("fvcore.common.checkpoint").setLevel(logging.WARNING)  # Detectron2 logger
    logging.getLogger("utils.torch_utils").setLevel(logging.WARNING)  # yolov5 loggers
    logging.getLogger("models.yolo").setLevel(logging.WARNING)

    def dims(s):
        try:
            w, h = map(int, s.split(','))
            return w, h
        except:
            raise argparse.ArgumentTypeError("Dimensions must be expressed as widht,height without quotes or parantheses")

    ap = argparse.ArgumentParser()
    ap.add_argument("-d", "--detector", required=True,
                    help="path to detector config (json file)")
    ap.add_argument("-t", "--tracker", required=True,
                    help="path to tracker config (json file)")
    ap.add_argument("-c", "--classes", required=True,
                    help="path to classes (json file)")
    ap.add_argument("-l", "--log_level", type=str, default="INFO",
                    help="Log level."
                         "Possible values : \"NOTSET\", \"DEBUG\", \"INFO\", \"WARNING\", \"ERROR\", \"CRITICAL\".")
    args = vars(ap.parse_args())

    assert args["log_level"] in ["NOTSET", "DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"], \
        "Invalid log level. Possible values : \"NOTSET\", \"DEBUG\", \"INFO\", \"WARNING\", \"ERROR\", \"CRITICAL\"."

    log = logging.getLogger("aptitude-toolbox")
    coloredlogs.install(level=args["log_level"],
                        fmt="%(asctime)s %(hostname)s %(name)s[%(process)d] %(levelname)s %(message)s")


    tco.main(args["detector"], args["tracker"], args["classes"])
