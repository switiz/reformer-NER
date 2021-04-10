from __future__ import absolute_import, division, print_function

import argparse
import logging
import os
import random
from io import open

import numpy as np
import torch
from torch.utils.data import (DataLoader, RandomSampler, SequentialSampler)
from tqdm import tqdm

from model.mlm_ner import ReformerNERModel
from transformers.optimization import AdamW
from util.schedule import WarmupLinearSchedule
from transformers import BertTokenizer
from util.arg import ModelConfig
from finetuning.ner.data_loader import load_and_cache_examples, convert_examples_to_features, get_labels
from finetuning.ner.utils import compute_metrics, show_report, get_test_texts

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def save(config, epoch, model, optimizer, losses, train_step):
    logger.info("** ** * Saving file * ** **")
    model_checkpoint = f"{config.model_name}.pt"
    logger.info(model_checkpoint)
    output_model_file = os.path.join(config.output_dir, model_checkpoint)

    if os.path.exists(output_model_file):
        os.remove(output_model_file)

    torch.save({
        'epoch': epoch,  # 현재 학습 epoch
        'model_state_dict': model.state_dict(),  # 모델 저장
        'optimizer_state_dict': optimizer.state_dict(),  # 옵티마이저 저장
        'losses': losses,  # Loss 저장
        'train_step': train_step,  # 현재 진행한 학습
    }, output_model_file)


def eval(model, config, epoch, eval_dataset, test_texts, predict_batch_size):
    logger.info("***** Evaluating *****")
    logger.info("  Num features = %d", len(eval_dataset))
    logger.info("  Batch size = %d", predict_batch_size)

    pad_token_label_id = torch.nn.CrossEntropyLoss().ignore_index
    sampler = SequentialSampler(eval_dataset)
    eval_dataloader = DataLoader(eval_dataset, sampler=sampler, batch_size=predict_batch_size, drop_last=True)

    model.eval()
    model.to(device)
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    if n_gpu > 0:
        torch.cuda.manual_seed_all(SEED)

    logger.info("Start evaluating!")
    preds = None
    out_label_ids = None
    nb_eval = 0
    label_lst = get_labels(config)
    for batch in tqdm(eval_dataloader,
                      desc="Evaluating"):  # tqdm(dataloader, desc="Evaluating", leave=True, position=1):
        batch = tuple(t.to(device) for t in batch)
        with torch.no_grad():
            input_ids, attention_mask, token_type_ids, labels = batch
            logits = model(input_ids)
        if preds is None:
            preds = logits.detach().cpu().numpy()
            out_label_ids = labels.detach().cpu().numpy()
        else:
            preds = np.append(preds, logits.detach().cpu().numpy(), axis=0)
            out_label_ids = np.append(out_label_ids, labels.detach().cpu().numpy(), axis=0)

        # Slot result
        preds = np.argmax(preds, axis=2)
        slot_label_map = {i: label for i, label in enumerate(label_lst)}
        out_label_list = [[] for _ in range(out_label_ids.shape[0])]
        preds_list = [[] for _ in range(out_label_ids.shape[0])]

        results = {}
        for i in range(out_label_ids.shape[0]):
            for j in range(out_label_ids.shape[1]):
                if out_label_ids[i, j] != pad_token_label_id:
                    out_label_list[i].append(slot_label_map[out_label_ids[i][j]])
                    preds_list[i].append(slot_label_map[preds[i][j]])

        if config.write_pred:
            if not os.path.exists(config.pred_dir):
                os.makedirs(config.pred_dir)
        with open(os.path.join(config.pred_dir, "pred_{}.txt".format(nb_eval)), "w", encoding="utf-8") as f:
            for text, true_label, pred_label in zip(test_texts, out_label_list, preds_list):
                for t, tl, pl in zip(text, true_label, pred_label):
                    f.write("{} {} {}\n".format(t, tl, pl))
                f.write("\n")

        result = compute_metrics(out_label_list, preds_list)
        results.update(result)
        nb_eval+=1
        logger.info("***** Eval results *****")
        for key in sorted(results.keys()):
            logger.info("  %s = %s", key, str(results[key]))
            logger.info("\n" + show_report(out_label_list, preds_list))  # Get the report for each tag result

        return results


def train(model, config, args):
    # 시작 Epoch
    num_params = count_parameters(model)
    logger.info("Total Parameter: %d" % num_params)

    train_dataset, _ = load_and_cache_examples(config, tokenizer, 'train')
    test_dataset, test_features = load_and_cache_examples(config, tokenizer, 'test')

    num_train_optimization_steps = int(len(train_dataset) / train_batch_size) * num_train_epochs

    # Prepare optimizer
    param_optimizer = list(model.named_parameters())
    no_decay = ['bias', 'LayerNorm.weight']
    optimizer_grouped_parameters = [
        {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)],
         'weight_decay': 0.01},
        {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
    ]

    optimizer = AdamW(optimizer_grouped_parameters,
                      lr=learning_rate,
                      eps=adam_epsilon)
    scheduler = WarmupLinearSchedule(optimizer,
                                     warmup_steps=num_train_optimization_steps * 0.1,
                                     t_total=num_train_optimization_steps)
    global_step = 0
    start_epoch = 0
    start_step= 0
    # Reformer NER 모델
    args.load_ner_checkpoint = False
    if os.path.isfile(f'{config.checkpoint_path}/{config.model_name}.pth') and args.load_ner_checkpoint is False:
        checkpoint = torch.load(f'{config.checkpoint_path}/{config.model_name}.pth', map_location=device)
        model.reformer.load_state_dict(checkpoint['model_state_dict'], strict=False)
        logger.info(f'Load Reformer Model')



    train_sampler = RandomSampler(train_dataset)
    train_dataloader = DataLoader(train_dataset, sampler=train_sampler, batch_size=train_batch_size, drop_last=True)

    if args.load_ner_checkpoint:
        checkpoint = torch.load(config.ner_checkpoint_path, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'], strict=False)
        start_epoch = checkpoint['epoch']
        global_step = checkpoint['train_step']
        start_step = global_step if start_epoch == 0 else global_step * train_batch_size % len(train_dataloader)
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        logger.info(f'Load Reformer[NER] Model')

    logger.info("***** Running training *****")
    logger.info("  Batch size = %d", train_batch_size)
    logger.info("  Num steps = %d", num_train_optimization_steps)
    num_train_step = num_train_optimization_steps

    test_texts = get_test_texts(config.data_dir, config.test_file)

    for epoch in range(start_epoch, int(num_train_epochs)):
        global_step, model = train_on_epoch(model, train_dataloader, optimizer, scheduler, start_step, global_step, epoch,
                                            num_train_step)
        start_step = 0
        eval(model, config, epoch, test_dataset, test_texts, train_batch_size)


def train_on_epoch(model, train_dataloader, optimizer, scheduler, start_step, global_step, epoch, num_train_step):
    model.train()
    iter_bar = tqdm(train_dataloader, desc=f"Epoch-{epoch} Train(XX Epoch) Step(XX/XX) (Mean loss=X.X) (loss=X.X)")
    tr_step, total_loss, mean_loss = 0, 0., 0.

    for step, batch in enumerate(iter_bar):
        if step < start_step:
            continue
        batch = tuple(t.to(device) for t in batch)  # multi-gpu does scattering it-self
        inputs_ids, attention_mask, token_type_ids, labels = batch
        outputs = model(inputs_ids, attention_mask, labels)
        loss = outputs
        loss.backward()
        total_loss += loss.item()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
        optimizer.step()
        scheduler.step()# Update learning rate schedule
        model.zero_grad()
        global_step += 1
        if config.ckpt_steps > 0 and global_step % config.ckpt_steps == 0:
            save(config, epoch, model, optimizer, loss, global_step)
        iter_bar.set_description(f"Epoch-{epoch} Train Step(%d / %d) (Mean loss=%5.5f) (loss=%5.5f)" %
                                 (global_step, num_train_step, mean_loss, loss.item()))
    save(config, epoch, model, optimizer, loss, global_step)
    return global_step, model


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--load_ner_checkpoint', action="store_true", help="load NER Model checkpoint")
    parser.add_argument('--resume', action="store_true", help="resume func")
    args = parser.parse_args()

    logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                        datefmt='%m/%d/%Y %H:%M:%S',
                        level=logging.INFO)
    logger = logging.getLogger(__name__)

    # 1. config load
    config_path = "../config/mlm/ner-pretrain-small.json"
    config = ModelConfig(config_path).get_config()
    if not os.path.exists(config.output_dir):
        os.makedirs(config.output_dir)

    # 2. hyperparameter
    train_batch_size = config.batch_size
    num_train_epochs = config.epochs
    learning_rate = 2e-4
    warmup_proportion = 0.1
    max_grad_norm = 1.0
    adam_epsilon = 1e-6
    weight_decay = 0.01

    SEED = 42
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)

    # 3. Tokenizer
    tokenizer = BertTokenizer(vocab_file=config.vocab_path, do_lower_case=False)

    # 4. Prepare model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    n_gpu = torch.cuda.device_count()
    logger.info("device: {} n_gpu: {}".format(device, n_gpu))

    model = ReformerNERModel(
        num_tokens=tokenizer.vocab_size,
        dim=config.dim,
        depth=config.depth,
        heads=config.n_head,
        max_seq_len=config.max_seq_len,
        causal=False,  # auto-regressive 학습을 위한 설정
        num_labels=len(get_labels(config))
    ).to(device)

    # 5. train
    train(model, config, args)
