from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import _init_paths

import os

import torch
import torch.utils.data
from opts import opts
from models.model import create_model, load_model, save_model
from models.data_parallel import DataParallel
from logger import Logger
from datasets.dataset_factory import get_dataset
from trains.train_factory import train_factory

from tensorboardX import SummaryWriter       # importing tensorboard
from utils.early_stopping import EarlyStopping
from test import test

#torch.backends.cudnn.enabled   = True

def main(opt, tb):
  torch.manual_seed(opt.seed)
  torch.backends.cudnn.benchmark = not opt.not_cuda_benchmark and not opt.test
  Dataset = get_dataset(opt.dataset, opt.task)
  opt = opts().update_dataset_info_and_set_heads(opt, Dataset)
  print(opt)

  logger = Logger(opt)

  os.environ['CUDA_VISIBLE_DEVICES'] = opt.gpus_str
  opt.device = torch.device('cuda' if opt.gpus[0] >= 0 else 'cpu')
  
  print('Creating model...')
  model = create_model(opt.arch, opt.heads, opt.head_conv)
  optimizer = torch.optim.Adam(model.parameters(), opt.lr)
  start_epoch = 0
  if opt.load_model != '':
    model, optimizer, start_epoch = load_model(
      model, opt.load_model, optimizer, opt.resume, opt.lr, opt.lr_step)

  Trainer = train_factory[opt.task]
  trainer = Trainer(opt, model, optimizer)
  trainer.set_device(opt.gpus, opt.chunk_sizes, opt.device)

  print('Setting up data...')
  val_loader = torch.utils.data.DataLoader(
      Dataset(opt, 'val'), 
      batch_size=1, 
      shuffle=False,
      num_workers=1,
      pin_memory=True
  )

  if opt.test:
    _, preds = trainer.val(0, val_loader)
    val_loader.dataset.run_eval(preds, opt.save_dir)
    return

  train_loader = torch.utils.data.DataLoader(
      Dataset(opt, 'train'), 
      batch_size=opt.batch_size, 
      shuffle=True,
      num_workers=opt.num_workers,
      pin_memory=True,
      drop_last=True
  )

  print('Starting training...')
  best = 1e10

  if opt.es:
    early_stopping = EarlyStopping(patience=10, verbose=True)
  
  best_model_dir = None
  iteration = 0

  for epoch in range(start_epoch + 1, opt.num_epochs + 1):
    mark = epoch if opt.save_all else 'last'
    log_dict_train, _, iteration = trainer.train(epoch, train_loader, iteration, tb)
    logger.write('epoch: {} |'.format(epoch))
    for k, v in log_dict_train.items():
      logger.scalar_summary('train_{}'.format(k), v, epoch)
      logger.write('{} {:8f} | '.format(k, v))
    if opt.val_intervals > 0 and epoch % opt.val_intervals == 0:
      #save_model(os.path.join(opt.save_dir, 'model_{}.pth'.format(mark)), 
      #           epoch, model, optimizer)
      with torch.no_grad():
        log_dict_val, preds, iteration = trainer.val(epoch, val_loader, iteration, tb)

        if opt.es:
          dir = early_stopping(log_dict_val[opt.metric], model, epoch, opt.arch, opt.task, opt.lr_step)
          if dir == None:
            best_model_dir = best_model_dir
          else:
            best_model_dir = dir
        else:
          if log_dict_val[opt.metric] < best:
            best = log_dict_val[opt.metric]
            best_model_dir = save_checkpoint(best, model, epoch, opt.arch, opt.task, opt.lr_step, es = False)

        if epoch % 1 == 0:
          best_model_dir = os.path.join(os.getcwd(), "..", "exp", best_model_dir)
          stats = test(opt, best_model_dir, tb)
          print(stats[0])
          map_avg = stats[0]
          map_50 = stats[1]
          map_75 = stats[2]
          map_small = stats[3]
          map_medium = stats[4]
          map_large = stats[5]
          mar_1 = stats[6]
          mar_10 = stats[7]
          mar_100 = stats[8]
          mar_small = stats[9]
          mar_medium = stats[10]
          mar_large = stats[11]
          tb.add_scalar('Average mAP vs Epoch', map_avg, epoch)
          tb.add_scalar('mAP (IoU 0.5) vs Epoch', map_50, epoch)
          tb.add_scalar('mAP (IoU 0.75) vs Epoch', map_75, epoch)
          tb.add_scalar('mAP (Area = Small) vs Epoch', map_small, epoch)
          tb.add_scalar('mAP (Area = Medium) vs Epoch', map_medium, epoch)
          tb.add_scalar('mAP (Area = Large) vs Epoch', map_large, epoch)
          tb.add_scalar('mAR (Max Detection = 1) vs Epoch', mar_1, epoch)
          tb.add_scalar('mAR (Max Detection = 10) vs Epoch', mar_10, epoch)
          tb.add_scalar('mAR (Max Detection = 100) vs Epoch', mar_100, epoch)
          tb.add_scalar('mAR (Area = Small) vs Epoch', mar_small, epoch)
          tb.add_scalar('mAR (Area = Medium) vs Epoch', mar_medium, epoch)
          tb.add_scalar('mAR (Area = Large) vs Epoch', mar_large, epoch)

      for k, v in log_dict_val.items():
        logger.scalar_summary('val_{}'.format(k), v, epoch)
        logger.write('{} {:8f} | '.format(k, v))
      #if log_dict_val[opt.metric] < best:
      #  best = log_dict_val[opt.metric]
      #  save_model(os.path.join(opt.save_dir, 'model_best.pth'), 
      #             epoch, model)

    else:
      save_model(os.path.join(opt.save_dir, 'model_last.pth'), 
                 epoch, model, optimizer)
    logger.write('\n')

    if opt.es and early_stopping.early_stop:
      print("Early stopping")
      break

    if epoch in opt.lr_step:
      #save_model(os.path.join(opt.save_dir, 'model_{}.pth'.format(epoch)), 
      #           epoch, model, optimizer)
      save_checkpoint(log_dict_val[opt.metric], model, epoch, opt.arch, opt.task, opt.lr_step, es = False)
      lr = opt.lr * (0.1 ** (opt.lr_step.index(epoch) + 1))
      print('Drop LR to', lr)
      for param_group in optimizer.param_groups:
          param_group['lr'] = lr
  logger.close()

if __name__ == '__main__':
  opt = opts().parse()

  tb = SummaryWriter(comment=' Model = <' + str(opt.arch) + '>, batch_size = ' + str(opt.batch_size) +
                      ', learning_rate = ' + str(opt.lr) + 
                      ', reg_loss = ' + str(opt.reg_loss) +
                      ', hm_weight = ' + str(opt.hm_weight) + 
                      ', off_weight = ' + str(opt.off_weight) + 
                      ', wh_weight = ' + str(opt.wh_weight))

  main(opt, tb)