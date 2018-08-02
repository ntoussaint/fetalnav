import os
import shutil
import time
import torch
import torch.backends.cudnn as cudnn
import torch.nn.parallel
import torch.optim
import torch.utils.data
import torchnet as tnt
from tqdm import tqdm
import pickle

import numpy as np


def FGE_learning_rate(epoch, lr_init=5e-2, lr_final=5e-5,
                      max_epoch=30, cycle_length=6,
                      percentage_before_cycles=.3, number_of_drops=2):

    def cyclic_learning_rate(lr1, lr2, i, c):
        t = 1./c * ( (i-1)%c + 1 )
        if t < .5:
            lr = (1 - 2*t) * lr1 + 2*t * lr2
        else:
            lr = (2 - 2*t) * lr2 + (2*t - 1) * lr1
        return lr

    epoch_switch = int(percentage_before_cycles*max_epoch)

    if epoch >= epoch_switch:
        lr1 = lr_init / (2**int((epoch_switch-1)*number_of_drops/(epoch_switch)))
        lr = cyclic_learning_rate(lr1, lr_final, epoch - epoch_switch, cycle_length)
    else:
        lr = lr_init / (2**int(epoch*number_of_drops/epoch_switch))
    return lr


class Engine(object):
    def __init__(self, state={}):
        self.state = state
        if self._state('use_gpu') is None:
            self.state['use_gpu'] = torch.cuda.is_available()

        if self._state('evaluate') is None:
            self.state['evaluate'] = False

        if self._state('start_epoch') is None:
            self.state['start_epoch'] = 0

        if self._state('max_epochs') is None:
            self.state['max_epochs'] = 100

        if self._state('epoch_step') is None:
            self.state['epoch_step'] = [5,10,15,20,25]

        if self._state('maximize') is None:
            self.state['maximize'] = False

        # meters
        self.state['meter_loss'] = tnt.meter.AverageValueMeter()
        # time measure
        self.state['batch_time'] = tnt.meter.AverageValueMeter()
        self.state['data_time'] = tnt.meter.AverageValueMeter()
        # display parameters
        if self._state('use_pb') is None:
            self.state['use_pb'] = True

    def _state(self, name):
        if name in self.state:
            return self.state[name]

    def on_start_epoch(self, training, model, criterion, data_loader, optimizer=None):
        self.state['meter_loss'].reset()
        self.state['batch_time'].reset()
        self.state['data_time'].reset()

    def on_end_epoch(self, training, model, criterion, data_loader, optimizer=None):
        loss = self.state['meter_loss'].value()[0]
        if training:
            print_strs = 'Epoch: [{0}]\t trainLoss {loss:.4f}'.format(self.state['epoch'], loss=loss)
            log_str = 'epoch/loss/train'
        else:
            print_strs = 'testLoss {loss:.4f}'.format(loss=loss)
            log_str = 'epoch/loss/validate'
        print(print_strs)

        if self.state['Logger'] is not None:
            self.state['Logger'].add_scalar(log_str, loss, self.state['epoch']+1)

        return loss

    def on_start_batch(self, training, model, criterion, data_loader, optimizer=None):
        pass

    def on_end_batch(self, training, model, criterion, data_loader, optimizer=None):
        # record loss
        self.state['loss_batch'] = self.state['loss'].data[0]
        self.state['meter_loss'].add(self.state['loss_batch'])
        if training:
            log_str = 'batch/loss/train'
        else:
            log_str = 'batch/loss/validate'
        if self.state['Logger'] is not None:
            self.state['Logger'].add_scalar(log_str, self.state['loss_batch'], self.state['total_iteration_test']+1)

    def on_forward(self, training, model, criterion, data_loader, optimizer=None):

        input_var = torch.autograd.Variable(self.state['input'])
        target_var = torch.autograd.Variable(self.state['target'])

        if not training:
            input_var.volatile = True
            target_var.volatile = True

        # compute output
        self.state['output'] = model(input_var)
        self.state['loss'] = criterion(self.state['output'], target_var)

        if training:
            optimizer.zero_grad()
            self.state['loss'].backward()
            optimizer.step()

    def init_learning(self, model, criterion):
        self.state['best_score'] = 0 if self.state['maximize'] else np.inf
        self.state['total_iteration_train'] = 0
        self.state['total_iteration_test'] = 0

    def learning(self, model, criterion, train_loader, val_loader, optimizer=None):

        self.init_learning(model, criterion)

        # optionally resume from a checkpoint
        if self._state('resume') is not None:
            if os.path.isfile(self.state['resume']):
                print("=> loading checkpoint '{}'".format(self.state['resume']))
                checkpoint = torch.load(self.state['resume'])
                self.state['start_epoch'] = checkpoint['epoch']
                self.state['best_score'] = checkpoint['best_score']
                model.load_state_dict(checkpoint['state_dict'])
                print("=> loaded checkpoint '{}' (epoch {})"
                      .format(self.state['evaluate'], checkpoint['epoch']))
            else:
                print("=> no checkpoint found at '{}'".format(self.state['resume']))

        if self.state['use_gpu']:
            train_loader.pin_memory = True
            val_loader.pin_memory = True
            cudnn.benchmark = True
            model = torch.nn.DataParallel(model).cuda()
            criterion = criterion.cuda()

        if self.state['evaluate']:
            if not 'epoch' in self.state:
                self.state['epoch'] = 0
            self.validate(val_loader, model, criterion)

        for epoch in range(self.state['start_epoch'], self.state['max_epochs']):
            self.state['epoch'] = epoch
            self.adjust_learning_rate(optimizer)

            # train for one epoch
            self.train(train_loader, model, criterion, optimizer, epoch)

            self.save_checkpoint({
                'epoch': epoch + 1,
                'arch': self._state('arch'),
                'state_dict': model.module.state_dict() if self.state['use_gpu'] else model.state_dict(),
                'best_score': self.state['best_score'],
            }, False, filename='checkpoint_{:02d}.pth.tar'.format(epoch + 1))

            if ((epoch+1) % 5 == 0) or (epoch == self.state['max_epochs']-1):
                # evaluate on validation set
                prec1 = self.validate(val_loader, model, criterion)

                # remember best prec@1 and save checkpoint
                if self.state['maximize']:
                    is_best = prec1 > self.state['best_score']
                    self.state['best_score'] = max(prec1, self.state['best_score'])
                else:
                    is_best = prec1 < self.state['best_score']
                    self.state['best_score'] = min(prec1, self.state['best_score'])
                print(' *** best={best:.3f}'.format(best=self.state['best_score']))

    def train(self, data_loader, model, criterion, optimizer, epoch):
        # switch to train mode
        model.train()
        self.on_start_epoch(True, model, criterion, data_loader, optimizer)
        if self.state['use_pb']:
            data_loader = tqdm(data_loader, desc='Training')
        end = time.time()
        for i, (input, target) in enumerate(data_loader):
            # measure data loading time
            self.state['iteration'] = i
            self.state['data_time_batch'] = time.time() - end
            self.state['data_time'].add(self.state['data_time_batch'])
            self.state['total_iteration_train'] = self.state['total_iteration_train']+1
            self.state['input'] = input
            self.state['target'] = target
            if 'train_class_weights' in self.state.keys():
                weights = np.array([ self.state['train_class_weights'][idx] for idx in np.argmax(target.numpy(), axis=1) ])
                self.state['weights'] = torch.Tensor(weights)
            else:
                self.state['weights'] = None

            self.on_start_batch(True, model, criterion, data_loader, optimizer)
            if self.state['use_gpu']:
                self.state['target'] = self.state['target'].cuda(async=True)

            self.on_forward(True, model, criterion, data_loader, optimizer)
            # measure elapsed time
            self.state['batch_time_current'] = time.time() - end
            self.state['batch_time'].add(self.state['batch_time_current'])
            end = time.time()
            # measure accuracy
            self.on_end_batch(True, model, criterion, data_loader, optimizer)
        self.on_end_epoch(True, model, criterion, data_loader, optimizer)

    def validate(self, data_loader, model, criterion):

        # switch to evaluate mode
        model.eval()

        self.on_start_epoch(False, model, criterion, data_loader)

        nsamples = len(data_loader.dataset)
        nclasses = len(self.state['classes'])
        y_gt = np.zeros((nsamples, nclasses))
        y_scores = np.zeros((nsamples, nclasses))

        if self.state['use_pb']:
            data_loader = tqdm(data_loader, desc='Test')

        class_correct = [0.] * len(self.state['classes'])
        class_total = [0.] * len(self.state['classes'])

        end = time.time()

        for i, (input, target) in enumerate(data_loader):
            # measure data loading time
            self.state['iteration'] = i
            self.state['data_time_batch'] = time.time() - end
            self.state['data_time'].add(self.state['data_time_batch'])
            self.state['total_iteration_test'] = self.state['total_iteration_test']+1

            self.state['input'] = input
            self.state['target'] = target
            if 'val_class_weights' in self.state.keys():
                weights = np.array([ self.state['val_class_weights'][idx] for idx in np.argmax(target.numpy(), axis=1) ])
                self.state['weights'] = torch.Tensor(weights)
            else:
                self.state['weights'] = None
            self.on_start_batch(False, model, criterion, data_loader)

            target = self.state['target']

            if self.state['use_gpu']:
                self.state['target'] = self.state['target'].cuda(async=True)

            self.on_forward(False, model, criterion, data_loader)

            output = self.state['output']

            # measure elapsed time
            self.state['batch_time_current'] = time.time() - end
            self.state['batch_time'].add(self.state['batch_time_current'])
            end = time.time()
            # measure accuracy
            self.on_end_batch(False, model, criterion, data_loader)

            _, gt        = torch.max(target, 1)
            _, predicted = torch.max(output, 1)
            c = [float(predicted[i] == gt[i]) for i in range(len(gt))]
            for i in range(len(c)):
                label = int(gt[i])
                class_correct[label] += c[i]
                class_total[label] += 1

            y_gt[self.state['iteration']*self.state['batch_size']:(self.state['iteration']+1)*self.state['batch_size'],:] = target
            y_scores[self.state['iteration']*self.state['batch_size']:(self.state['iteration']+1)*self.state['batch_size'],:] = output

        for i in range(len(self.state['classes'])):
            print('Accuracy of %5s : %2d %%' % (
                self.state['classes'][i], 100 * class_correct[i] / class_total[i]))

        score = self.on_end_epoch(False, model, criterion, data_loader)

        prec_recall = {
            'targets': y_gt,
            'outputs': y_scores,
        }

        pickle.dump(prec_recall, open(os.path.join(self.state['save_model_path'],
            'checkpoint_{:02d}_test_outputs.pkl'.format(self.state['epoch']+1)),
            'wb'), protocol=pickle.HIGHEST_PROTOCOL)

        return score

    def save_checkpoint(self, state, is_best, filename='checkpoint.pth.tar'):
        if self._state('save_model_path') is not None:
            filename_ = filename
            filename = os.path.join(self.state['save_model_path'], filename_)
            if not os.path.exists(self.state['save_model_path']):
                os.makedirs(self.state['save_model_path'])
        print('save model {filename}'.format(filename=filename))
        torch.save(state, filename)
        if is_best:
            filename_best = 'model_best.pth.tar'
            if self._state('save_model_path') is not None:
                filename_best = os.path.join(self.state['save_model_path'], filename_best)
            shutil.copyfile(filename, filename_best)
            if self._state('save_model_path') is not None:
                if self._state('filename_previous_best') is not None:
                    os.remove(self._state('filename_previous_best'))
                filename_best = os.path.join(self.state['save_model_path'], 'model_best_{score:.4f}_{time}.pth.tar'.format(score=state['best_score'], time=time.strftime("%Y%m%d%H%M%S", time.localtime())))
                shutil.copyfile(filename, filename_best)
                self.state['filename_previous_best'] = filename_best

    def adjust_learning_rate(self, optimizer):
        """Sets the learning rate to the initial LR decayed by 10 every 10 epochs"""
        # lr = args.lr * (0.1 ** (epoch // 10))
        if self.state['epoch'] is not 0 and self.state['epoch'] in self.state['epoch_step']:
            print('update learning rate / 10...')
            for param_group in optimizer.param_groups:
                param_group['lr'] = param_group['lr'] * 0.1
            print('update learning rate: lr={}'.format(param_group['lr']))
        # lr = FGE_learning_rate(epoch=self.state['epoch'],
        #                        max_epoch=self.state['max_epochs'],
        #                        cycle_length=6,
        #                        percentage_before_cycles=.5,
        #                        lr_init=self.state['learning_rate'],
        #                        lr_final=5e-5,
        #                        number_of_drops=3)
        # print('update learning rate: lr={}'.format(lr))
        # for param_group in optimizer.param_groups:
        #     param_group['lr'] = lr

class MultiLabelMAPEngine(Engine):
    def __init__(self, state):
        Engine.__init__(self, state)

        self.state['ap_meter'] = tnt.meter.mAPMeter()
        self.state['maximize'] = True

    def init_learning(self, model, criterion):
        Engine.init_learning(self, model, criterion)

    def on_start_epoch(self, training, model, criterion, data_loader, optimizer=None):
        Engine.on_start_epoch(self, training, model, criterion, data_loader, optimizer)
        self.state['ap_meter'].reset()
        if 'classes' in self.state:
            self.state['occurences'] = np.zeros(len(self.state['classes']))

    def on_end_epoch(self, training, model, criterion, data_loader, optimizer=None):
        map = 100 * self.state['ap_meter'].value()
        loss = self.state['meter_loss'].value()[0]
        if training:
            strs = 'Epoch: [{0}]\t trainLoss {loss:.4f}\t trainMAP {map:.3f}'.format(self.state['epoch'], loss=loss, map=map)
        else:
            strs = 'testLoss {loss:.4f}\t testMAP {map:.3f}'.format(loss=loss, map=map)
        print(strs)

        if self.state['Logger'] is not None:
            info = {}
            info['loss'] = loss
            info['mAP'] = self.state['ap_meter'].value()
            if training:
                self.state['Logger'].add_scalars('train/epoch', info, self.state['epoch']+1)
            else:
                self.state['Logger'].add_scalars('validate/epoch', info, self.state['epoch']+1)

        return map

    def on_start_batch(self, training, model, criterion, data_loader, optimizer=None):
        Engine.on_start_batch(self, training, model, criterion, data_loader, optimizer)

    def on_end_batch(self, training, model, criterion, data_loader, optimizer=None):
        # record loss
        self.state['loss_batch'] = self.state['loss'].data[0]
        self.state['meter_loss'].add(self.state['loss_batch'])
        self.state['ap_meter'].add(self.state['output'].data, self.state['target'], self.state['weights'])

        if self.state['Logger'] is not None:
            info = {}
            info['loss'] = self.state['meter_loss'].value()[0]
            info['mAP'] = self.state['ap_meter'].value()
            if training:
                self.state['Logger'].add_scalars('train/batch', info, self.state['total_iteration_train']+1)
            else:
                self.state['Logger'].add_scalars('validate/batch', info, self.state['total_iteration_test']+1)

            if ('classes' in self.state):
                self.state['occurences'] = self.state['occurences'] + np.sum(self.state['target'].cpu().numpy(), 0)
                info = {}
                for idx, oc in enumerate(self.state['occurences']):
                    info['{}'.format(self.state['classes'][idx])] = oc
                if training:
                    self.state['Logger'].add_scalars('train/occurences', info, self.state['total_iteration_train']+1)
                else:
                    self.state['Logger'].add_scalars('validate/occurences', info, self.state['total_iteration_test']+1)
