# Copyright 2019 Image Analysis Lab, German Center for Neurodegenerative Diseases (DZNE), Bonn
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


# =========================================================================
# solver.py — 训练编排工具
# =========================================================================
# 本文件包含训练神经网络所需的全套工具：
# 1. 辅助函数（创建目录、评估指标计算）
# 2. 可视化函数（分割结果、混淆矩阵）
# 3. Solver 类 —— 训练循环核心
#
# 注意：DDParcel 推理（DDSurfer_Pred.py）完全不使用本文件。
# 本文件只在训练新模型时使用。
# =========================================================================
import os
import torch
import time
import matplotlib.pyplot as plt
import numpy as np
import itertools
import glob

from torch.autograd import Variable
from torch.optim import lr_scheduler
from torchvision import utils
from skimage import color
from models.losses import CombinedLoss


# =========================================================================
# 辅助函数
# =========================================================================

def create_exp_directory(exp_dir_name):
    """
    创建实验目录（如果不存在）。
    用于存放训练过程中的 checkpoints 和日志。
    """
    if not os.path.exists(exp_dir_name):
        try:
            os.makedirs(exp_dir_name)
            print("Successfully Created Directory @ {}".format(exp_dir_name))
        except:
            print("Directory Creation Failed - Check Path")
    else:
        print("Directory {} Exists ".format(exp_dir_name))


# =========================================================================
# 评价指标函数
# =========================================================================

def dice_confusion_matrix(batch_output, labels_batch, num_classes):
    """
    计算 Dice 混淆矩阵。
    CM[i][j] = 类别 i 和类别 j 之间的 Dice 系数。
    - i == j 的对角线 = 每个类别的 Dice 分数
    - i != j 反映模型把 i 类误分为 j 类的程度

    :param batch_output: 离散预测标签 [N, H, W]
    :param labels_batch: 真实标签 [N, H, W]
    :param num_classes: 类别数
    :return: avg_dice (标量), dice_cm (num_classes × num_classes 矩阵)
    """
    dice_cm = torch.zeros(num_classes, num_classes)

    for i in range(num_classes):
        gt = (labels_batch == i).float()

        for j in range(num_classes):
            pred = (batch_output == j).float()
            inter = torch.sum(torch.mul(gt, pred)) + 0.0001
            union = torch.sum(gt) + torch.sum(pred) + 0.0001
            dice_cm[i, j] = 2 * torch.div(inter, union)

    avg_dice = torch.mean(torch.diagflat(dice_cm))

    return avg_dice, dice_cm


def iou_score(pred_cls, true_cls, nclass=79):
    """
    计算交并比（IoU / Jaccard 系数）。
    IoU = |P ∩ T| / |P ∪ T|
    常用于评估分割质量。

    :param pred_cls: 预测标签（离散值）
    :param true_cls: 真实标签
    :param nclass: 类别数（默认 79，不含背景）
    :return: (intersect_, union_)，每个类别的交集和并集像素数
    """
    intersect_ = []
    union_ = []

    for i in range(1, nclass):
        intersect = ((pred_cls == i).float() + (true_cls == i).float()).eq(2).sum().item()
        union = ((pred_cls == i).float() + (true_cls == i).float()).ge(1).sum().item()
        intersect_.append(intersect)
        union_.append(union)

    return np.array(intersect_), np.array(union_)


def precision_recall(pred_cls, true_cls, nclass=79):
    """
    计算每个类别的精确率（Precision）和召回率（Recall）：
      Recall = TP / (TP + FN)
      Precision = TP / (TP + FP)

    :return: (tpos, tpos_fneg, tpos_fpos)
      - tpos:      每类的真阳性数
      - tpos_fneg: 每类的真阳性 + 假阴性（即该类在 GT 中的总数）
      - tpos_fpos: 每类的真阳性 + 假阳性（即该类在预测中的总数）
    """
    tpos_fneg = []
    tpos_fpos = []
    tpos = []

    for i in range(1, nclass):
        all_pred = (pred_cls == i).float()
        all_gt = (true_cls == i).float()

        tpos.append((all_pred + all_gt).eq(2).sum().item())
        tpos_fpos.append(all_pred.sum().item())
        tpos_fneg.append(all_gt.sum().item())

    return np.array(tpos), np.array(tpos_fneg), np.array(tpos_fpos)


# =========================================================================
# 可视化函数
# =========================================================================

def plot_predictions(images_batch, labels_batch, batch_output, plt_title, file_save_name):
    """
    绘制验证集的分割结果对比图：
    - 左图：输入切片
    - 中图：真实分割
    - 右图：模型预测

    保存为 PDF 文件用于训练过程中人工检查模型效果。
    """
    f = plt.figure(figsize=(20, 20))
    n, c, h, w = images_batch.shape
    mid_slice = c // 2
    images_batch = torch.unsqueeze(images_batch[:, mid_slice, :, :], 1)
    grid = utils.make_grid(images_batch.cpu(), nrow=4)

    plt.subplot(131)
    plt.imshow(grid.numpy().transpose((1, 2, 0)))
    plt.title('Slices')

    grid = utils.make_grid(labels_batch.unsqueeze_(1).cpu(), nrow=4)[0]
    color_grid = color.label2rgb(grid.numpy(), bg_label=0)
    plt.subplot(132)
    plt.imshow(color_grid)
    plt.title('Ground Truth')

    grid = utils.make_grid(batch_output.unsqueeze_(1).cpu(), nrow=4)[0]
    color_grid = color.label2rgb(grid.numpy(), bg_label=0)
    plt.subplot(133)
    plt.imshow(color_grid)
    plt.title('Prediction')

    plt.suptitle(plt_title)
    plt.tight_layout()

    f.savefig(file_save_name, bbox_inches='tight')
    plt.close(f)
    plt.gcf().clear()


def plot_confusion_matrix(cm, classes, title='Confusion matrix', cmap=plt.cm.Blues, file_save_name="temp.pdf"):
    """
    绘制混淆矩阵热图。
    行 = 真实类别，列 = 预测类别。
    对角线越亮 = 该类别预测越准确。
    """
    f = plt.figure(figsize=(35, 35))

    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))

    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    fmt = '.2f'
    thresh = cm.max() / 2.

    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')

    f.savefig(file_save_name, bbox_inches='tight')
    plt.close(f)
    plt.gcf().clear()


# =========================================================================
# Solver 类 — 训练循环核心
# =========================================================================
# Solver 封装了完整的训练流程：
# - 优化器、学习率调度器
# - epoch 级训练/验证循环
# - checkpoint 保存与恢复
# - 指标记录与可视化
#
# 使用方式：
#   solver = Solver(num_classes=82)
#   solver.train(model, train_loader, val_loader, ...)
# =========================================================================
class Solver(object):
    """
    Class for training neural networks
    """

    # gamma 是学习率衰减因子，step_size 是衰减步长
    # 每 5 个 epoch，学习率乘以 0.3
    default_lr_scheduler_args = {"gamma": 0.3,
                                 "step_size": 5}

    def __init__(self, num_classes, optimizer=torch.optim.Adam, optimizer_args={}, loss_func=CombinedLoss(), lr_scheduler_args={}):

        # Merge and update the default arguments - optimizer
        self.optimizer_args = optimizer_args

        lr_scheduler_args_merged = Solver.default_lr_scheduler_args.copy()
        lr_scheduler_args_merged.update(lr_scheduler_args)

        # Merge and update the default arguments - lr scheduler
        self.lr_scheduler_args = lr_scheduler_args_merged

        self.optimizer = optimizer
        self.loss_func = loss_func          # 默认 CombinedLoss（Dice + CE）
        self.num_classes = num_classes
        self.classes = list(range(self.num_classes))

    def train(self, model, train_loader, validation_loader, class_names, num_epochs, log_params, expdir, scheduler_type, torch_v11, resume=True):
        """
        训练模型的完整循环。

        主要步骤：
        0. 创建实验/日志目录，恢复已有 checkpoint（可选）
        1. 对每个 epoch：
           a. 训练模式：逐 batch 前向→计算损失→反向传播→更新参数
           b. 验证模式：逐 batch 前向→计算 IoU、Dice、Precision/Recall
           c. 每 log_iter 个 epoch 保存一次 checkpoint
        2. 返回最佳模型

        :param model: PyTorch 模型
        :param train_loader: 训练数据 DataLoader
        :param validation_loader: 验证数据 DataLoader
        :param class_names: 类别名称列表
        :param num_epochs: 训练总轮数
        :param log_params: 日志配置字典（包括 logger 和 logdir）
        :param expdir: checkpoint 保存目录
        :param scheduler_type: 学习率调度器类型（如 "StepLR"）
        :param torch_v11: PyTorch 版本是否为 1.1 以下（影响 scheduler 调用时机）
        :param resume: 是否从最近的 checkpoint 恢复训练
        """
        create_exp_directory(expdir)               # 存放 checkpoint
        create_exp_directory(log_params["logdir"]) # 存放日志和可视化

        # 实例化优化器（默认 Adam）
        optimizer = self.optimizer(model.parameters(), **self.optimizer_args)

        # 实例化学习率调度器
        if scheduler_type == "StepLR":
            scheduler = lr_scheduler.StepLR(optimizer, step_size=self.lr_scheduler_args["step_size"], gamma=self.lr_scheduler_args["gamma"])

            log_params["logger"].info("Scheduler: StepLR, step_size: {}, gamma: {}".format(self.lr_scheduler_args["step_size"], self.lr_scheduler_args["gamma"]))
        else:
            scheduler = None
            log_params["logger"].info("Scheduler: None")

        # 日志格式化模板（用于打印所有类别的指标）
        a = "{}\t" * (self.num_classes - 2) + "{}"

        epoch = -1  # 从 -1 开始，因为下面 while 循环先 +1
        print('-------> Starting to train')

        # 恢复模型：扫描 expdir 下最新的 checkpoint 文件并加载
        if resume:
            try:
                prior_model_paths = sorted(glob.glob(os.path.join(expdir, 'Epoch_*')), key=os.path.getmtime)
                if prior_model_paths:
                    current_model = prior_model_paths.pop()

                state = torch.load(current_model)

                # 恢复模型权重、优化器状态、调度器状态
                model.load_state_dict(state["model_state_dict"])
                optimizer.load_state_dict(state["optimizer_state_dict"])
                if scheduler is not None:
                    scheduler.load_state_dict(state["scheduler_state_dict"])
                epoch = state["epoch"]

                print("Successfully restored the model state. Resuming training from Epoch {}".format(epoch + 1))

            except Exception as e:
                print("No model to restore. Resuming training from Epoch 0. {}".format(e))

        log_params["logger"].info("{} parameters in total".format(sum(x.numel() for x in model.parameters())))

        # =====================================================================
        # 主训练循环
        # =====================================================================
        while epoch < num_epochs:

            epoch = epoch + 1
            epoch_start = time.time()

            # PyTorch < 1.2：在 epoch 开始时更新学习率
            if torch_v11 and scheduler is not None:
                scheduler.step()

            loss_batch = np.zeros(1)

            # ──────────── 训练阶段 ────────────
            for batch_idx, sample_batch in enumerate(train_loader):

                # 从 DataLoader 获取数据
                images_batch, labels_batch, weights_batch = sample_batch['image'], sample_batch['label'], sample_batch['weight']

                # 包装为 Variable（PyTorch 0.4.0 之前的语法，现在可以省略）
                images_batch = Variable(images_batch)
                labels_batch = Variable(labels_batch)
                weights_batch = Variable(weights_batch)

                if torch.cuda.is_available():
                    images_batch, labels_batch, weights_batch = images_batch.cuda(), labels_batch.cuda(), weights_batch.type(torch.FloatTensor).cuda()

                model.train()
                optimizer.zero_grad()
                predictions = model(images_batch)
                loss_total, loss_dice, loss_ce = self.loss_func(predictions, labels_batch, weights_batch)
                loss_total.backward()  # 反向传播
                optimizer.step()       # 参数更新

                loss_batch += loss_total.item()

                if batch_idx % (len(train_loader) // 2) == 0 or batch_idx == len(train_loader) - 1:
                    log_params["logger"].info("Train Epoch: {} [{}/{}] ({:.0f}%)] with loss: {}".format(epoch, batch_idx,
                                              len(train_loader), 100. * batch_idx / len(train_loader), loss_batch / (batch_idx + 1)))

                del images_batch, labels_batch, weights_batch, predictions, loss_total, loss_dice, loss_ce

            # PyTorch >= 1.2：在 epoch 结束时更新学习率
            if not torch_v11 and scheduler is not None:
                scheduler.step()

            epoch_finish = time.time() - epoch_start

            log_params["logger"].info("Train Epoch {} finished in {:.04f} seconds.".format(epoch, epoch_finish))

            # ──────────── 验证阶段 ────────────
            model.eval()

            val_loss_total = 0
            val_loss_dice = 0
            val_loss_ce = 0

            ints_ = np.zeros(self.num_classes - 1)
            unis_ = np.zeros(self.num_classes - 1)
            per_cls_counts_gt = np.zeros(self.num_classes - 1)
            per_cls_counts_pred = np.zeros(self.num_classes - 1)
            accs = np.zeros(self.num_classes - 1)

            # ─── 保存 checkpoint ───
            # 每 log_iter 个 epoch 保存一次
            if epoch % log_params["log_iter"] == 0:
                save_name = os.path.join(expdir, 'Epoch_' + str(epoch).zfill(2) + '_training_state.pkl')
                checkpoint = {"model_state_dict": model.state_dict(),
                              "optimizer_state_dict": optimizer.state_dict(),
                              "epoch": epoch}
                if scheduler is not None:
                    checkpoint["scheduler_state_dict"] = scheduler.state_dict()

                torch.save(checkpoint, save_name)

            # ─── 验证循环（当前被 if False 禁用，以加速训练）───
            # 如果需要，可以改为 epoch % 10 == 0 每 10 轮做一次完整验证
            if False:
                with torch.no_grad():

                    if validation_loader is not None:

                        val_start = time.time()
                        cnf_matrix_validation = torch.zeros(self.num_classes, self.num_classes)

                        for batch_idx, sample_batch in enumerate(validation_loader):

                            images_batch, labels_batch, weights_batch = sample_batch['image'], sample_batch['label'], sample_batch['weight']

                            images_batch = Variable(images_batch)
                            labels_batch = Variable(labels_batch)
                            weights_batch = Variable(weights_batch)

                            if torch.cuda.is_available():
                                images_batch, labels_batch, weights_batch = images_batch.cuda(), labels_batch.cuda(), weights_batch.type(torch.FloatTensor).cuda()

                            predictions = model(images_batch)
                            loss_total, loss_dice, loss_ce = self.loss_func(predictions, labels_batch, weights_batch)
                            val_loss_total += loss_total.item()
                            val_loss_dice += loss_dice.item()
                            val_loss_ce += loss_ce.item()

                            _, batch_output = torch.max(predictions, dim=1)

                            # 累积验证指标
                            int_, uni_ = iou_score(batch_output, labels_batch, self.num_classes)
                            ints_ += int_
                            unis_ += uni_

                            tpos, pcc_gt, pcc_pred = precision_recall(batch_output, labels_batch, self.num_classes)
                            accs += tpos
                            per_cls_counts_gt += pcc_gt
                            per_cls_counts_pred += pcc_pred

                            _, cm_batch = dice_confusion_matrix(batch_output, labels_batch, self.num_classes)
                            cnf_matrix_validation += cm_batch.cpu()

                            # 第一 batch 绘制分割结果可视化
                            if batch_idx == 0:
                                plt_title = 'Validation Results Epoch ' + str(epoch)
                                file_save_name = os.path.join(log_params["logdir"], 'Epoch_' + str(epoch) + '_Validations_Predictions.pdf')
                                plot_predictions(images_batch, labels_batch, batch_output, plt_title, file_save_name)

                            del images_batch, labels_batch, weights_batch, predictions, batch_output, \
                                int_, uni_, tpos, pcc_gt, pcc_pred, loss_total, loss_dice, loss_ce

                        # 计算最终指标并记录
                        ious = ints_ / unis_
                        val_loss_total /= (batch_idx + 1)
                        val_loss_dice /= (batch_idx + 1)
                        val_loss_ce /= (batch_idx + 1)
                        cnf_matrix_validation = cnf_matrix_validation / (batch_idx + 1)
                        val_end = time.time() - val_start

                        print("Completed Validation Dataset in {:0.4f} s".format(val_end))

                        save_name = os.path.join(log_params["logdir"], 'Epoch_' + str(epoch) + '_Validation_Dice_CM.pdf')
                        plot_confusion_matrix(cnf_matrix_validation.cpu().numpy(), self.classes, file_save_name=save_name)

                        # 日志记录
                        log_params["logger"].info("[Epoch {} stats]: MIoU: {:.4f}; "
                                                  "Mean Recall: {:.4f}; "
                                                  "Mean Precision: {:.4f}; "
                                                  "Avg loss total: {:.4f}; "
                                                  "Avg loss dice: {:.4f}; "
                                                  "Avg loss ce: {:.4f}".format(epoch, np.mean(ious),
                                                                               np.mean(accs / per_cls_counts_gt),
                                                                               np.mean(accs / per_cls_counts_pred),
                                                                               val_loss_total, val_loss_dice, val_loss_ce))

                        log_params["logger"].info(a.format(*class_names))
                        log_params["logger"].info(a.format(*ious))

            # 切回训练模式
            model.train()
