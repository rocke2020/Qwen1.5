import json
import logging
import math
import os
import random
import re
import shutil
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
from pandas import DataFrame
from tqdm import tqdm

sys.path.append(os.path.abspath("."))
from utils_comm.file_util import file_util
from utils_comm.log_util import ic, logger

SEED = 0
random.seed(SEED)
np.random.seed(SEED)


def convert_to_qwen15():
    """
    https://github.com/QwenLM/Qwen1.5/issues/156

    Input: read from a jsonl file, each line is a dict as below:
        {"conversation_id": 1, "category": "Brainstorming",
            "conversation": [
                {"human": "如何保障工作中遵循正确的安全准则？",
                "assistant": "为了保障工作中遵循正确的安全准则，以下是一些建议：\n\n1.了解相关安全规定和标准：了解相关的安全规定和标准，并遵守它们。这可以包括公司和政府的安全标准，以及行业标准和最佳实践。\n\n2.培训和教育：确保您和您的同事接受了必要的培训和教育，以了解正确的安全准则和行为。\n\n3.使用正确的工具和设备：确保您使用正确的工具和设备，并且它们得到了正确的维护和保养。\n\n4.个人防护装备：确保您和您的同事穿戴正确的个人防护装备，如安全鞋、透明眼镜或面罩、手套等。\n\n5.创建一个安全文化：鼓励个人对安全的责任感和意识，并创建一个安全文化，使人们始终关注他们自己和他人的安全。\n\n6.持续监测和改进：持续监测和改进安全准则和程序，以确保它们保持最新，并适应变化的情况。\n\n这些措施可以帮助您和您的同事在工作中保障正确的安全准则，确保您的工作场所是一个安全的环境。"},
                {"human": "如何应对突发紧急情况？",
                "assistant": "突发紧急情况可能会在任何时间发生，为了应对这些情况，以下是一些建议：\n\n1.保持冷静：保持冷静和清醒，以便能够更好地应对情况。\n\n2.评估情况：评估紧急情况的性质和范围，以便能够采取适当的行动。\n\n3.采取行动：根据情况采取适当的行动，可能包括撤离、报警、提供急救等。\n\n4.寻求帮助：如果需要，寻求帮助，包括向紧急服务部门报警或联系其他人员。\n\n5.保护自己和他人：确保您和他人的安全，采取措施保护自己和他人的生命和财产。\n\n6.学习经验：在紧急情况发生后，学习经验，以便能够更好地应对类似情况。\n\n这些措施可以帮助您在突发紧急情况中保持冷静和应对情况，确保您和他人的安全。"}
            ]
        }

    Output: save to a jsonl file, each line is a dict as below:
        {
            "type": "chatml",
            "messages": [
                {
                    "role": "system",
                    "content": "You are a helpful assistant."
                },
                {
                    "role": "user",
                    "content": "Tell me something about large language models."
                },
                {
                    "role": "assistant",
                    "content": "Large language models are a type of language model that is trained on a large corpus of text data. They are capable of generating human-like text and are used in a variety of natural language processing tasks..."
                }
            ],
            "source": "unknown"
        }
    """
    orig_file = "/mnt/nas1/dong-qichang/corpus/fine-tune/moss-003-sft-data/moss-003-sft-data-small.jsonl"
    data = file_util.read_jsonl(orig_file)
    new_d = []
    system_prompt = {"role": "system", "content": "You are a helpful assistant."}
    for item in data:
        message = {"type": "chatml", "messages": [system_prompt], "source": "unknown"}
        conversation = item["conversation"]
        for turn in conversation:
            message["messages"].append({"role": "user", "content": turn["human"]})
            message["messages"].append({"role": "assistant", "content": turn["assistant"]})
        new_d.append(message)
    file_util.write_jsonl(new_d, "/mnt/nas1/dong-qichang/corpus/fine-tune/moss-003-sft-data/moss-003-sft-data-small-qwen1.5.jsonl")


if __name__ == "__main__":
    convert_to_qwen15()
