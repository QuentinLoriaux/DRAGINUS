import os
import json
import argparse
from tqdm import tqdm
from copy import copy
import logging
from data import StrategyQA, WikiMultiHopQA, HotpotQA, IIRC
from generate import *

logging.basicConfig(level=logging.INFO) 
logger = logging.getLogger(__name__)


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("-c", "--config_path", type=str, required=True)
    parser.add_argument("-i", "--intolerance", type=float, required=False)
    parser.add_argument("-o", "--output_no_rag", action='store_true', help="compare results with those without RAG", required=False)
    args = parser.parse_args()
    config_path = args.config_path
    intolerance = args.intolerance
    output_no_rag = args.output_no_rag
    with open(config_path, "r") as f:
        args = json.load(f)
    args = argparse.Namespace(**args)
    args.config_path = config_path
    args.intolerance = intolerance
    args.output_no_rag = output_no_rag 
    if "shuffle" not in args:
        args.shuffle = False 
    if "use_counter" not in args:
        args.use_counter = True
    return args


def main():
    args = get_args()
    logger.info(f"{args}")

    # output dir
    if os.path.exists(args.output_dir) is False:
        os.makedirs(args.output_dir)
    dir_name = os.listdir(args.output_dir)
    for i in range(10000):
        if str(i) not in dir_name:
            args.output_dir = os.path.join(args.output_dir, str(i))
            os.makedirs(args.output_dir)
            break
    logger.info(f"output dir: {args.output_dir}")
    # save config
    with open(os.path.join(args.output_dir, "config.json"), "w") as f:
        json.dump(args.__dict__, f, indent=4)
    # create output file
    output_file = open(os.path.join(args.output_dir, "output.txt"), "w")
    if args.output_no_rag:
        output_file_no_rag = open(os.path.join(args.output_dir, "output_no_rag.txt"), "w")
        print("The output without RAG will be saved in output_no_rag.txt")

    # load data
    if args.dataset == "strategyqa":
        data = StrategyQA(args.data_path)
    elif args.dataset == "2wikimultihopqa":
        data = WikiMultiHopQA(args.data_path)
    elif args.dataset == "hotpotqa":
        data = HotpotQA(args.data_path)
    elif args.dataset == "iirc":
        data = IIRC(args.data_path)
    else:
        raise NotImplementedError
    data.format(fewshot=args.fewshot)
    data = data.dataset
    if args.shuffle:
        data = data.shuffle()
    if args.sample != -1:
        samples = min(len(data), args.sample)
        data = data.select(range(samples))
   
    model = AttnWeightRAG(args)

    logger.info("start inference")
    for i in tqdm(range(953,len(data))):
        last_counter = copy(model.counter)
        batch = data[i]

        pred = model.inference(batch["question"], batch["demo"], batch["case"])
        pred = pred.strip()
        ret = {
            "qid": batch["qid"], 
            "prediction": pred,
        }
        if args.use_counter:
            ret.update(model.counter.calc(last_counter))
        output_file.write(json.dumps(ret)+"\n")

        if args.output_no_rag:
            pred = model.inference(batch["question"], batch["demo"], batch["case"], ragless=True)
            pred = pred.strip()
            ret = {
                "qid": batch["qid"], 
                "prediction": pred,
            }
            output_file_no_rag.write(json.dumps(ret)+"\n")
    

if __name__ == "__main__":
    main()
