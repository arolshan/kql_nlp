import argparse
import os
import logging
import matplotlib.pyplot as plt
import numpy as np
import torch
import Utils
import json
import time
import torch.nn.functional as F
from KQLDurationDataset import KQLDurationDataset
from KQLClassifier import KQLDurationBucketClassifier
from sklearn.model_selection import train_test_split
from tokenizers import (CharBPETokenizer)
from torch import nn
from torch.utils.data import DataLoader
from transformers import BertTokenizer, AdamW, get_linear_schedule_with_warmup
from collections import defaultdict
from Consts import *
from pathlib import Path

logFormatter = logging.Formatter("[%(asctime)s, %(threadName)s, %(levelname)s] %(message)s")
logging.basicConfig(level=logging.INFO)
rootLogger = logging.getLogger()

consoleHandler = logging.StreamHandler()
consoleHandler.setFormatter(logFormatter)
rootLogger.addHandler(consoleHandler)


def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


def extract_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("-o", "--output_path", required=False, default="./output", type=str, help="Output path")
    parser.add_argument("-hl", "--hidden_layers", type=int, default='1', help="Number of hidden layers in Fully connected")
    parser.add_argument("-hs", "--hidden_layer_sizes", nargs='+', default=[2048], help="Hidden Layer Output sizes")
    parser.add_argument("-b", "--batch_size", type=int, default='64', help="Number of batch_size")
    parser.add_argument("-e", "--epochs", type=int, default='3', help="Number of epochs")
    parser.add_argument("--cuda_device", type=int, default='0', help="CUDA device id")
    parser.add_argument("--lr", type=float, default='2e-3', help="Learning rate of adam")
    parser.add_argument("--seed", type=int, default='42', help="Download the data")
    parser.add_argument("--buckets_strategy", type=str, default=BUCKET_STRATEGY_KMEANS, help="Type of bucketing algorithm for duration. Possible values: 'kmeans', 'uniform'")
    parser.add_argument("--buckets_count", type=int, default='10', help="Number of duration buckets")
    parser.add_argument("--minimal_queries_per_container", type=int, default='50', help="Minimal amount of queries for container")
    parser.add_argument("--outliers_threshold", type=float, default='0.997', help="Threshold percentile for removing outliers")
    parser.add_argument("--dropout", type=float, default='0.1', help="Model dropout probability")
    parser.add_argument("--dl_worker_count", type=int, default='4', help="Number of data loader workers")
    parser.add_argument("--subset", type=int, default='-1', help="Run only on subset of data")
    parser.add_argument("--ast", type=str2bool, nargs='?', const=True, default=True, help="Uses AST data")
    parser.add_argument("--print_report", type=str2bool, nargs='?', const=True, default=True, help="Number of data loader workers")
    parser.add_argument("--use_weights", type=str2bool, nargs='?', const=True, default=False, help="Pass weights to loss function")
    parser.add_argument("--with_cuda", type=str2bool, nargs='?', const=True, default=True, help="Training with CUDA: true, or false")
    parser.add_argument("--download_data", type=str2bool, nargs='?', const=False, default=True, help="Download the data")
    parser.add_argument("--data_parallel", type=str2bool, nargs='?', const=True, default=True, help="Run batches with data parallel")
    parser.add_argument("--inference", type=str2bool, nargs='?', const=True, default=False, help="Is inference mode, i.e. evaluate last model state without training")
    args = parser.parse_args()
    return args


def create_data_loader(df, tokenizer, bucket_to_idx, container_to_idx, batch_size, num_workers=1):
    ds = KQLDurationDataset(
        kql_queries=df[QUERY].to_numpy(),
        bucket_labels=df[DURATION_BUCKET].to_numpy(),
        raw_durations=df[DURATION].to_numpy(),
        container_labels=df[CONTAINER].to_numpy(),
        bucket_to_idx=bucket_to_idx,
        container_to_idx=container_to_idx,
        tokenizer=tokenizer,
        tokenizer_max_len=512)

    return DataLoader(ds, batch_size=batch_size, num_workers=num_workers)


def train_epoch(
        model,
        data_loader,
        loss_fn,
        optimizer,
        device,
        scheduler,
        n_examples):
    model = model.train()

    losses = []
    correct_predictions = 0

    logging.info(f"Batches={len(data_loader)}")
    i = 0
    print_every = 1000
    for d in data_loader:
        input_ids = d[DL_INPUT_IDS].to(device)
        attention_mask = d[DL_ATTN_MASK].to(device)
        targets = d[DL_BUCKETS].to(device)
        containers_onehot = d[DL_CONTAINERS].to(device)
        print_info = True if i % print_every == 0 else False

        outputs = model(
            input_ids=input_ids,
            containers_onehot=containers_onehot,
            attention_mask=attention_mask,
            print_info=print_info,
            batch_number=i
        )

        _, preds = torch.max(outputs, dim=1)
        loss = loss_fn(outputs, targets)

        correct_predictions += torch.sum(preds == targets)
        losses.append(loss.item())

        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        scheduler.step()
        optimizer.zero_grad()
        i += 1

    return correct_predictions.double() / n_examples, np.mean(losses)


def eval_model(model, data_loader, loss_fn, device, n_examples, bucket_average_durations, mode='train'):
    model = model.eval()
    losses = []
    correct_predictions = 0
    correct_predictions_test = torch.tensor(0)
    test_applicative_success = torch.tensor(0)
    dist = []
    with torch.no_grad():
        for d in data_loader:
            input_ids = d[DL_INPUT_IDS].to(device)
            attention_mask = d[DL_ATTN_MASK].to(device)
            targets = d[DL_BUCKETS].to(device)
            real_durations = d[DL_RAW_DURATIONS].to(device)
            containers_onehot = d[DL_CONTAINERS].to(device)

            outputs = model(
                input_ids=input_ids,
                containers_onehot=containers_onehot,
                attention_mask=attention_mask)

            _, preds = torch.max(outputs, dim=1)
            loss = loss_fn(outputs, targets)
            correct_predictions += torch.sum(preds == targets)
            losses.append(loss.item())

            if (mode != 'train'):
                probs = F.softmax(outputs, dim=1)  # [batch_size, class_num]
                top_probs_values, top_probs_indices = torch.topk(probs, 7)
                top_probs_values_sum = torch.sum(top_probs_values, dim=1).cpu()
                top_probs_values_sum_ones = torch.ones(len(top_probs_values_sum)).cpu()
                top_probs_resize = top_probs_values_sum_ones / top_probs_values_sum
                top_probs_resize = (torch.eye(len(top_probs_resize)) * top_probs_resize)
                top_probs_resize = torch.mm(top_probs_resize, top_probs_values.cpu())
                top_bucket_average_durations = torch.tensor(bucket_average_durations)[top_probs_indices]
                predicted_duration = torch.sum(top_probs_resize.cpu() * torch.tensor(top_bucket_average_durations), dim=1)
                expected_duration = real_durations.cpu()

                predicted_bucket = get_test_bucket(predicted_duration)
                expected_bucket = get_test_bucket(expected_duration)
                correct_predictions_test += torch.sum(predicted_bucket == expected_bucket)
                dist += [(torch.dist(expected_duration / 1000, predicted_duration / 1000) ** 2)]
                test_applicative_success += torch.sum((expected_duration < predicted_duration * 1.2) & (expected_duration > predicted_duration * 0.8))

    return (correct_predictions.double() / n_examples), \
           (correct_predictions_test.double() / n_examples), \
           np.mean(losses), \
           (((sum(dist) / n_examples) ** 0.5) * 1000), \
           test_applicative_success.double() / n_examples


def get_test_bucket(value):
    fast = (value <= 1000).type(torch.uint8)
    intermediate = ((value < 10000) & (value > 1000)).type(torch.uint8)
    slow = (value >= 10000).type(torch.uint8)

    return fast + intermediate * 2 + slow * 3


def get_predictions(model, data_loader, device):
    model = model.eval()

    review_texts = []
    predictions = []
    prediction_probs = []
    real_values = []

    with torch.no_grad():
        for d in data_loader:
            texts = d[DL_KQL_QUERY]
            input_ids = d[DL_INPUT_IDS].to(device)
            attention_mask = d[DL_ATTN_MASK].to(device)
            targets = d[DL_BUCKETS].to(device)
            containers_onehot = d[DL_CONTAINERS].to(device)

            outputs = model(
                input_ids=input_ids,
                containers_onehot=containers_onehot,
                attention_mask=attention_mask)

            probs = F.softmax(outputs, dim=1)
            _, preds = torch.max(outputs, dim=1)
            review_texts.extend(texts)
            predictions.extend(preds)
            prediction_probs.extend(probs)
            real_values.extend(targets)

    predictions = torch.stack(predictions).cpu()
    prediction_probs = torch.stack(prediction_probs).cpu()
    real_values = torch.stack(real_values).cpu()

    return review_texts, predictions, prediction_probs, real_values


def train():
    args = extract_args()

    DATA_PATH = "./data"
    TOKENIZER_PATH = "./tokenizer"
    OUT_PATH = args.output_path

    os.makedirs(OUT_PATH, exist_ok=True)
    os.makedirs(DATA_PATH, exist_ok=True)
    os.makedirs(TOKENIZER_PATH, exist_ok=True)

    with open(f'{OUT_PATH}/best_model_state.bin', mode='ab+'):
        pass

    fileHandler = logging.FileHandler(f'{OUT_PATH}/out_{int(round(time.time() * 1000))}.log', mode='w')
    fileHandler.setFormatter(logFormatter)
    rootLogger.addHandler(fileHandler)

    with open(f'{OUT_PATH}/args.json', 'w') as out:
        json.dump(args.__dict__, out, indent=4)

    RANDOM_SEED = args.seed
    np.random.seed(RANDOM_SEED)
    torch.manual_seed(RANDOM_SEED)

    device = torch.device(f"cuda:{args.cuda_device}" if args.with_cuda else "cpu")

    data_uri = DATA_COMPRESSED_URI
    data_folder = "base"
    data_file = "data"
    if (args.ast):
        data_uri = DATA_AST_COMPRESSED_URI
        data_folder = "ast"
        data_file ="data_ast"
    if (args.download_data):
        Utils.download_data(data_uri, f'{DATA_PATH}/{data_file}.zip')

    Utils.unzip_file(f'{DATA_PATH}/{data_file}.zip', f'{DATA_PATH}/{data_folder}')
    df = Utils.load_json_files_to_df(f'{DATA_PATH}/{data_folder}', ["query", "container", "duration"])

    if (args.subset != -1):
        df = df[:args.subset]

    df, bucket_names, bucket_to_idx, container_names, container_to_idx, bucket_weights, bucket_average_durations = Utils.preprocess_kql_data(args, df)

    logging.info("training tokenizer")
    corpus_path = f'{DATA_PATH}/corpus.txt'
    with open(corpus_path, 'wb') as out:
        out.write(os.linesep.join(df[QUERY]).encode())

    with open(corpus_path, 'r') as file:
        data = file.read().replace('\n', ' ')
        vocab_len = len(set(data.split(' ')))
        logging.info(f'evaluated number of distinct words in corpus: {vocab_len}')

    tokenizer = CharBPETokenizer()
    tokenizer.train(files=corpus_path, vocab_size=int(vocab_len * 1.3), min_frequency=2, special_tokens=[
        "[PAD]",
        "[SEP]",
        "[UNK]",
        "[unused1]",
        "[CLS]"
    ])
    tokenizer.save_model(TOKENIZER_PATH)
    tokenizer.enable_padding(length=MAX_TOKENS_LEN)
    tokenizer.enable_truncation(max_length=MAX_TOKENS_LEN)

    logging.info(f"finished training tokenizer, files created in {TOKENIZER_PATH}: {os.listdir(TOKENIZER_PATH)}")
    logging.info(f'bucket_names:{bucket_names}')
    logging.info(f'bucket_to_idx:{bucket_to_idx}')
    logging.info(f'bucket_average_durations:{bucket_average_durations}')
    logging.info(f'bucket_weights:{bucket_weights}')
    logging.info(f'containers_count:{len(container_names)}')

    df_train, df_test = train_test_split(df, test_size=0.1, random_state=RANDOM_SEED)
    df_val, df_test = train_test_split(df_test, test_size=0.5, random_state=RANDOM_SEED)

    train_data_loader = create_data_loader(df_train, tokenizer, bucket_to_idx, container_to_idx, args.batch_size, args.dl_worker_count)
    val_data_loader = create_data_loader(df_val, tokenizer, bucket_to_idx, container_to_idx, args.batch_size, args.dl_worker_count)
    test_data_loader = create_data_loader(df_test, tokenizer, bucket_to_idx, container_to_idx, args.batch_size, args.dl_worker_count)

    model = KQLDurationBucketClassifier(
        len(bucket_names),
        len(container_names),
        n_linear=args.hidden_layers + 1,
        vocab_size=tokenizer.get_vocab_size(),
        h_sizes=args.hidden_layer_sizes,
        drop_p=args.dropout)
    logging.info(model)
    if (args.data_parallel and args.with_cuda):
        model = torch.nn.DataParallel(model)
    model = model.to(device)

    EPOCHS = args.epochs
    optimizer = AdamW(model.parameters(), lr=args.lr, correct_bias=False)
    total_steps = len(train_data_loader) * EPOCHS

    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=2500,
        num_training_steps=total_steps)

    loss_fn = None
    if (args.use_weights):
        loss_fn = nn.CrossEntropyLoss(weight=torch.tensor(bucket_weights).to(device)).to(device)
    else:
        loss_fn = nn.CrossEntropyLoss().to(device)

    history = defaultdict(list)
    best_accuracy = 0

    if (not args.inference):
        for epoch in range(EPOCHS):

            logging.info(f'Epoch {epoch + 1}/{EPOCHS}')
            logging.info('-' * 10)

            train_acc, train_loss = train_epoch(
                model,
                train_data_loader,
                loss_fn,
                optimizer,
                device,
                scheduler,
                len(df_train))

            logging.info(f'Train loss {train_loss} accuracy {train_acc}')

            val_acc, _, val_loss, _, _ = eval_model(
                model,
                val_data_loader,
                loss_fn,
                device,
                len(df_val),
                bucket_average_durations)

            logging.info(f'Val loss {val_loss} accuracy {val_acc}')

            history['train_acc'].append(train_acc)
            history['train_loss'].append(train_loss)
            history['val_acc'].append(val_acc)
            history['val_loss'].append(val_loss)

            if val_acc > best_accuracy:
                torch.save(model.state_dict(), f'{OUT_PATH}/best_model_state.bin')
                best_accuracy = val_acc

    else:
        model.load_state_dict(torch.load(f'{OUT_PATH}/best_model_state.bin'))


    test_acc, test_acc_big_buckets, test_loss, root_mean_squared, test_applicative_success = eval_model(
        model,
        val_data_loader,
        loss_fn,
        device,
        len(df_val),
        bucket_average_durations,
        "test")

    if (args.print_report):
        logging.info(f'----------------------   Validation  REPORT   ----------------------')
        logging.info(f'Model test loss: {np.mean(test_loss.item())}')
        logging.info(f'Model test accuracy: {test_acc.item()}')
        logging.info(f'Root mean squared error, between real duration and predicted weighted average duration:\n{root_mean_squared.item()}')
        logging.info(f'Accuracy over 3 buckets, (,10sec], (10sec,1sec], (1sec,0sec]:\n{test_acc_big_buckets.item()}')
        logging.info(f'Applicative success, count of prediction with less than 20% difference from actual:\n{test_applicative_success.item()}')
        logging.info(f'-------------------------------------------------------------')

    if (args.inference):
        test_acc, test_acc_big_buckets, test_loss, root_mean_squared, test_applicative_success = eval_model(
            model,
            test_data_loader,
            loss_fn,
            device,
            len(df_test),
            bucket_average_durations,
            "test")

        if (args.print_report):
            logging.info(f'----------------------   TEST REPORT   ----------------------')
            logging.info(f'Model test loss: {np.mean(test_loss.item())}')
            logging.info(f'Model test accuracy: {test_acc.item()}')
            logging.info(f'Root mean squared error, between real duration and predicted weighted average duration:\n{root_mean_squared.item()}')
            logging.info(f'Accuracy over 3 buckets, (,10sec], (10sec,1sec], (1sec,0sec]:\n{test_acc_big_buckets.item()}')
            logging.info(f'Applicative success, count of prediction with less than 20% difference from actual:\n{test_applicative_success.item()}')
            logging.info(f'-------------------------------------------------------------')


def main():
    train()


if __name__ == "__main__":
    main()
