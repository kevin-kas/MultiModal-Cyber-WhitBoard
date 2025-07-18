import os
import argparse
import random
import numpy as np
from tensorboardX import SummaryWriter
import yaml
from .Network.model1 import CAN
import torch
import time
import pickle as pkl
from torch.utils.data import DataLoader, Dataset, RandomSampler
from difflib import SequenceMatcher

from tqdm import tqdm
import math

def update_lr(optimizer, current_epoch, current_step, steps, epochs, initial_lr):
    if current_epoch < 1:
        new_lr = initial_lr / steps * (current_step + 1)
    elif 1 <= current_epoch <= 200:
        new_lr = 0.5 * (1 + math.cos((current_step + 1 + (current_epoch - 1) * steps) * math.pi / (200 * steps))) * initial_lr
    else:
        new_lr = 0.5 * (1 + math.cos((current_step + 1 + (current_epoch - 1) * steps) * math.pi / (epochs * steps))) * initial_lr   
    for param_group in optimizer.param_groups:
        param_group['lr'] = new_lr
class Meter:
    def __init__(self, alpha=0.9):
        self.nums = []
        self.exp_mean = 0
        self.alpha = alpha
    @property
    def mean(self):
        return np.mean(self.nums)

    def add(self, num):
        if len(self.nums) == 0:
            self.exp_mean = num
        self.nums.append(num)
        self.exp_mean = self.alpha * self.exp_mean + (1 - self.alpha) * num

def cal_score(word_probs, word_label, mask):
    line_right = 0
    if word_probs is not None:
        _, word_pred = word_probs.max(2)
    word_scores = [SequenceMatcher(None, s1[:int(np.sum(s3))], s2[:int(np.sum(s3))], autojunk=False).ratio() * (len(s1[:int(np.sum(s3))]) + len(s2[:int(np.sum(s3))])) / len(s1[:int(np.sum(s3))]) / 2
              for s1, s2, s3 in zip(word_label.cpu().detach().numpy(), word_pred.cpu().detach().numpy(), mask.cpu().detach().numpy())]
    batch_size = len(word_scores)
    for i in range(batch_size):
        if word_scores[i] == 1:
            line_right += 1
    ExpRate = line_right / batch_size
    word_scores = np.mean(word_scores)
    return word_scores, ExpRate
def train(params, model, optimizer, epoch, train_loader, writer=None):
    model.train()
    device = params['device']
    loss_meter = Meter()
    word_right, exp_right, length, cal_num = 0, 0, 0, 0

    with tqdm(train_loader, total=len(train_loader)//params['train_parts']) as pbar:
        for batch_idx, (images, image_masks, labels, label_masks) in enumerate(pbar):
            images, image_masks, labels, label_masks = images.to(device), image_masks.to(
                device), labels.to(device), label_masks.to(device)
            batch, time = labels.shape[:2]
            if not 'lr_decay' in params or params['lr_decay'] == 'cosine':
                update_lr(optimizer, epoch, batch_idx, len(train_loader), params['epochs'], params['lr'])
            optimizer.zero_grad()
            probs, counting_preds, word_loss, counting_loss = model(images, image_masks, labels, label_masks)
            loss = word_loss + counting_loss
            loss.backward()

            if params['gradient_clip']:
                torch.nn.utils.clip_grad_norm_(model.parameters(), params['gradient'])
            optimizer.step()
            loss_meter.add(loss.item())

            wordRate, ExpRate = cal_score(probs, labels, label_masks)
            word_right = word_right + wordRate * time
            exp_right = exp_right + ExpRate * batch
            length = length + time
            cal_num = cal_num + batch

            if writer:
                current_step = epoch * len(train_loader) // params['train_parts'] + batch_idx + 1
                writer.add_scalar('train/word_loss', word_loss.item(), current_step)
                writer.add_scalar('train/counting_loss', counting_loss.item(), current_step)
                writer.add_scalar('train/loss', loss.item(), current_step)
                writer.add_scalar('train/WordRate', wordRate, current_step)
                writer.add_scalar('train/ExpRate', ExpRate, current_step)
                writer.add_scalar('train/lr', optimizer.param_groups[0]['lr'], current_step)

            pbar.set_description(f'{epoch+1} word_loss:{word_loss.item():.4f} counting_loss:{counting_loss.item():.4f} WRate:{word_right / length:.4f} '
                                 f'ERate:{exp_right / cal_num:.4f}')
            if batch_idx >= len(train_loader) // params['train_parts']:
                break

        if writer:
            writer.add_scalar('epoch/train_loss', loss_meter.mean, epoch+1)
            writer.add_scalar('epoch/train_WordRate', word_right / length, epoch+1)
            writer.add_scalar('epoch/train_ExpRate', exp_right / cal_num, epoch + 1)
        return loss_meter.mean, word_right / length, exp_right / cal_num


def eval(params, model, epoch, eval_loader, writer=None):
    model.eval()
    device = params['device']
    loss_meter = Meter()
    word_right, exp_right, length, cal_num = 0, 0, 0, 0

    with tqdm(eval_loader, total=len(eval_loader)//params['valid_parts']) as pbar, torch.no_grad():
        for batch_idx, (images, image_masks, labels, label_masks) in enumerate(pbar):
            images, image_masks, labels, label_masks = images.to(device), image_masks.to(
                device), labels.to(device), label_masks.to(device)
            batch, time = labels.shape[:2]
            probs, counting_preds, word_loss, counting_loss = model(images, image_masks, labels, label_masks, is_train=False)
            loss = word_loss + counting_loss
            loss_meter.add(loss.item())

            wordRate, ExpRate = cal_score(probs, labels, label_masks)
            word_right = word_right + wordRate * time
            exp_right = exp_right + ExpRate * batch
            length = length + time
            cal_num = cal_num + batch

            if writer:
                current_step = epoch * len(eval_loader)//params['valid_parts'] + batch_idx + 1
                writer.add_scalar('eval/word_loss', word_loss.item(), current_step)
                writer.add_scalar('eval/counting_loss', counting_loss.item(), current_step)
                writer.add_scalar('eval/loss', loss.item(), current_step)
                writer.add_scalar('eval/WordRate', wordRate, current_step)
                writer.add_scalar('eval/ExpRate', ExpRate, current_step)

            pbar.set_description(f'{epoch+1} word_loss:{word_loss.item():.4f} counting_loss:{counting_loss.item():.4f} WRate:{word_right / length:.4f} '
                                 f'ERate:{exp_right / cal_num:.4f}')
            if batch_idx >= len(eval_loader) // params['valid_parts']:
                break

        if writer:
            writer.add_scalar('epoch/eval_loss', loss_meter.mean, epoch + 1)
            writer.add_scalar('epoch/eval_WordRate', word_right / length, epoch + 1)
            writer.add_scalar('epoch/eval_ExpRate', exp_right / len(eval_loader.dataset), epoch + 1)
        return loss_meter.mean, word_right / length, exp_right / cal_num
        
class HMERDataset(Dataset):
    def __init__(self, params, image_path, label_path, words, is_train=True):
        super(HMERDataset, self).__init__()
        if image_path.endswith('.pkl'):
            with open(image_path, 'rb') as f:
                self.images = pkl.load(f)
        elif image_path.endswith('.list'):
            with open(image_path, 'r') as f:
                lines = f.readlines()
            self.images = {}
            print(f'data files: {lines}')
            for line in lines:
                name = line.strip()
                print(f'loading data file: {name}')
                start = time.time()
                with open(name, 'rb') as f:
                    images = pkl.load(f)
                self.images.update(images)
                print(f'loading {name} cost: {time.time() - start:.2f} seconds!')

        with open(label_path, 'r') as f:
            self.labels = f.readlines()

        self.words = words
        self.is_train = is_train
        self.params = params

    def __len__(self):
        assert len(self.images) == len(self.labels)
        return len(self.labels)

    def __getitem__(self, idx):
        name, *labels = self.labels[idx].strip().split()
        name = name.split('.')[0] if name.endswith('jpg') else name
        image = self.images[name]
        image = torch.Tensor(255-image) / 255
        image = image.unsqueeze(0)
        labels.append('eos')
        words = self.words.encode(labels)
        words = torch.LongTensor(words)
        return image, words

def get_crohme_dataset(params):
    words = Words(params['word_path'])
    params['word_num'] = len(words)
    print(f"training data path images: {params['train_image_path']} labels: {params['train_label_path']}")
    print(f"val data path images: {params['eval_image_path']} labels: {params['eval_label_path']}")

    train_dataset = HMERDataset(params, params['train_image_path'], params['train_label_path'], words, is_train=True)
    eval_dataset = HMERDataset(params, params['eval_image_path'], params['eval_label_path'], words, is_train=False)

    train_sampler = RandomSampler(train_dataset)
    eval_sampler = RandomSampler(eval_dataset)

    train_loader = DataLoader(train_dataset, batch_size=params['batch_size'], sampler=train_sampler,
                              num_workers=params['workers'], collate_fn=collate_fn_dict[params['collate_fn']], pin_memory=True)
    eval_loader = DataLoader(eval_dataset, batch_size=1, sampler=eval_sampler,
                              num_workers=params['workers'], collate_fn=collate_fn_dict[params['collate_fn']], pin_memory=True)

    print(f'train dataset: {len(train_dataset)} train steps: {len(train_loader)} '
          f'eval dataset: {len(eval_dataset)} eval steps: {len(eval_loader)} ')
    return train_loader, eval_loader


def collate_fn(batch_images):
    max_width, max_height, max_length = 0, 0, 0
    batch, channel = len(batch_images), batch_images[0][0].shape[0]
    proper_items = []
    for item in batch_images:
        if item[0].shape[1] * max_width > 1600 * 320 or item[0].shape[2] * max_height > 1600 * 320:
            continue
        max_height = item[0].shape[1] if item[0].shape[1] > max_height else max_height
        max_width = item[0].shape[2] if item[0].shape[2] > max_width else max_width
        max_length = item[1].shape[0] if item[1].shape[0] > max_length else max_length
        proper_items.append(item)

    images, image_masks = torch.zeros((len(proper_items), channel, max_height, max_width)), torch.zeros((len(proper_items), 1, max_height, max_width))
    labels, labels_masks = torch.zeros((len(proper_items), max_length)).long(), torch.zeros((len(proper_items), max_length))

    for i in range(len(proper_items)):
        _, h, w = proper_items[i][0].shape
        images[i][:, :h, :w] = proper_items[i][0]
        image_masks[i][:, :h, :w] = 1
        l = proper_items[i][1].shape[0]
        labels[i][:l] = proper_items[i][1]
        labels_masks[i][:l] = 1
    return images, image_masks, labels, labels_masks


class Words:
    def __init__(self, words_path):
        with open(words_path) as f:
            words = f.readlines()
            print(f'共 {len(words)} 类符号。')
        self.words_dict = {words[i].strip(): i for i in range(len(words))}
        self.words_index_dict = {i: words[i].strip() for i in range(len(words))}

    def __len__(self):
        return len(self.words_dict)

    def encode(self, labels):
        label_index = [self.words_dict[item] for item in labels]
        return label_index

    def decode(self, label_index):
        label = ' '.join([self.words_index_dict[int(item)] for item in label_index])
        return label


collate_fn_dict = {
    'collate_fn': collate_fn
}


def save_checkpoint(model, optimizer, word_score, ExpRate_score, epoch, optimizer_save=False, path='checkpoints', multi_gpu=False, local_rank=0):
    filename = f'{os.path.join(path, model.name)}/{model.name}_WordRate-{word_score:.4f}_ExpRate-{ExpRate_score:.4f}_{epoch}.pth'
    if optimizer_save:
        state = {
            'model': model.state_dict(),
            'optimizer': optimizer.state_dict(),
        }
    else:
        state = {
            'model': model.state_dict()
        }
    torch.save(state, filename)
    print(f'Save checkpoint: {filename}\n')
    return filename


def load_checkpoint(model, optimizer, path):
    state = torch.load(path, map_location='cpu')
    if optimizer is not None and 'optimizer' in state:
        optimizer.load_state_dict(state['optimizer'])
    else:
        print(f'No optimizer in the pretrained model')
    model.load_state_dict(state['model'])


def load_config(yaml_path):
    try:
        with open(yaml_path, 'r') as f:
            params = yaml.load(f, Loader=yaml.FullLoader)
    except:
        print('try utf-8 encoding....')
        with open(yaml_path, 'r', encoding='UTF-8') as f:
            params = yaml.load(f, Loader=yaml.FullLoader)
    if not params['experiment']:
        print('name of experiment cannot be empty!')
        exit(-1)
    if not params['train_image_path']:
        print('train image path cannot be empty!')
        exit(-1)
    if not params['train_label_path']:
        print('train label path cannot be empty!')
        exit(-1)
    if not params['word_path']:
        print('word dict path cannot be empty!')
        exit(-1)
    if 'train_parts' not in params:
        params['train_parts'] = 1
    if 'valid_parts' not in params:
        params['valid_parts'] = 1
    if 'valid_start' not in params:
        params['valid_start'] = 0
    if 'word_conv_kernel' not in params['attention']:
        params['attention']['word_conv_kernel'] = 1
    return params


parser = argparse.ArgumentParser(description='model training')
parser.add_argument('--dataset', default='CROHME', type=str, help='name of the dataset')
parser.add_argument('--check', action='store_true', help='test code option')
args = parser.parse_args()

if not args.dataset:
    print('Please provide the dataset name')
    exit(-1)

if args.dataset == 'CROHME':
    config_file = 'config.yaml'

"""load config"""
params = load_config(config_file)

"""set random seed"""
random.seed(params['seed'])
np.random.seed(params['seed'])
torch.manual_seed(params['seed'])
torch.cuda.manual_seed(params['seed'])

os.environ['CUDA_VISIBLE_DEVICES'] = '0'
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
params['device'] = device

if args.dataset == 'CROHME':
    train_loader, eval_loader = get_crohme_dataset(params)

model = CAN(params)
now = time.strftime("%Y-%m-%d-%H-%M", time.localtime())
model.name = f'{params["experiment"]}_{now}_decoder-{params["decoder"]["net"]}'

print(model.name)
model = model.to(device)

if args.check:
    writer = None
else:
    writer = SummaryWriter(f'{params["log_dir"]}/{model.name}')

optimizer = getattr(torch.optim, params['optimizer'])(model.parameters(), lr=float(params['lr']),
                                                      eps=float(params['eps']), weight_decay=float(params['weight_decay']))

if params['finetune']:
    print('load pretrained model')
    print(f'pretrain model weight path: {params["checkpoint"]}')
    load_checkpoint(model, optimizer, params['checkpoint'])

if not args.check:
    if not os.path.exists(os.path.join(params['checkpoint_dir'], model.name)):
        os.makedirs(os.path.join(params['checkpoint_dir'], model.name), exist_ok=True)
    os.system(f'cp {config_file} {os.path.join(params["checkpoint_dir"], model.name, model.name)}.yaml')

"""training on CROHME"""
if args.dataset == 'CROHME':
    min_score, init_epoch = 0, 0

    for epoch in range(init_epoch, params['epochs']):
        train_loss, train_word_score, train_exprate = train(params, model, optimizer, epoch, train_loader, writer=writer)

        if epoch >= params['valid_start']:
            eval_loss, eval_word_score, eval_exprate = eval(params, model, epoch, eval_loader, writer=writer)
            print(f'Epoch: {epoch+1} loss: {eval_loss:.4f} word score: {eval_word_score:.4f} ExpRate: {eval_exprate:.4f}')
            if eval_exprate > min_score and not args.check and epoch >= params['save_start']:
                min_score = eval_exprate
                save_checkpoint(model, optimizer, eval_word_score, eval_exprate, epoch+1,
                                optimizer_save=params['optimizer_save'], path=params['checkpoint_dir'])
