import time
import optuna
import random
import numpy as np
import torch
import os

from torch.nn import BCEWithLogitsLoss
from sklearn.metrics import f1_score, matthews_corrcoef
from torch.utils.data import DataLoader
from torch.utils.tensorboard import  SummaryWriter
from torch.utils.data import RandomSampler
from ablation_mpnn_predictor import MPNNPredictorWithProtein
from ablation_loading_data import MolecularDataset
import dgl
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc,precision_score, recall_score, confusion_matrix



def set_seed(seed=42):
    """固定随机种子，确保每次训练结果一致"""
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    dgl.random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    os.environ["PYTHONHASHSEED"] = str(seed)





def early_stopping(avg_valid_loss,
                   avg_valid_acc,
                   best_valid_loss,
                   model,
                   epoch,
                   epochs_without_improvement,
                   best_model_state,
                   best_valid_acc,
                   patience=10):

    if best_valid_acc < avg_valid_acc:
        best_valid_acc = avg_valid_acc
        best_valid_loss = avg_valid_loss
        best_model_state = model.state_dict()
        epochs_without_improvement = 0
        return 'continue', best_valid_loss, epochs_without_improvement, best_model_state, best_valid_acc
    else:
        if avg_valid_loss < best_valid_loss:
            # best_valid_loss = avg_valid_loss
            epochs_without_improvement = 0
            # best_model_state = model.state_dict()  # Save the best model
            # torch.save(model.state_dict(), 'best_model{}.pth'.format(epoch + 1))  # Save model weights
            return 'continue', best_valid_loss, epochs_without_improvement, best_model_state,best_valid_acc
        else:
            epochs_without_improvement += 1
            if epochs_without_improvement >= patience:
                print(f"Early stopping triggered at epoch {epoch + 1}")
                return 'end', best_valid_loss, epochs_without_improvement, best_model_state,best_valid_acc
            else:
                return 'continue', best_valid_loss, epochs_without_improvement, best_model_state,best_valid_acc


def train_valid(model,
                loss_trans,
                learning_rate,
                weight_decay,
                batch_size,
                epochs,
                device,
                dataset_train,
                dataset_valid,
                writer,
                trial_id=None,
                seed=None,
                trial_number='default'
                ):

    if trial_id is None:
        if seed is not None:
            set_seed(seed)
        elif seed is None:
            seed = 42
            set_seed(seed)

    elif trial_id is not None:
        seed = trial_id
        set_seed(seed)



    # writer = SummaryWriter('train_logs')
    time_start = time.time()
    generator = torch.Generator()
    generator.manual_seed(seed)

    # 🎯 使用 RandomSampler 确保每个 trial 数据顺序不同



    dataloader_train = DataLoader(dataset_train,batch_size=batch_size, collate_fn=MolecularDataset.collate_fn,
                                  sampler=RandomSampler(dataset_train, generator=generator),

                                  num_workers=0,  pin_memory=False)# 指定自定义的 collate_fn
    dataloader_valid = DataLoader(dataset_valid, batch_size=batch_size,collate_fn=MolecularDataset.collate_fn,
                                  num_workers=0,  pin_memory=False)
    # dataloader_kras_test = DataLoader(kras_test, batch_size=batch_size, collate_fn=MolecularDataset.collate_fn,
    #                                 num_workers=0, pin_memory=False)

    optim = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)      #优化器选择：AdamW

    best_valid_loss = float('inf')  # 初始化最佳验证损失
    best_valid_acc= float('-inf')
    epochs_without_improvement = 0  # 计数器，记录多少轮没有提升
    best_model_state = None  # 用于保存最好的模型

    epoch_rounds=0
    train_round=0



    for epoch in range(epochs):
        model.train()

        running_loss = 0
        epoch_rounds+=1
        print('——————————Start {}th rounds of training——————————'.format(epoch_rounds))

        for i in dataloader_train:          #训练模型
            graph, node_feats, edge_feats, protein_feats, labels ,adj_matrix= i  # 解包 tuple
            graph = graph.to(device)
            protein_feats = protein_feats.to(device)
            node_feats = node_feats.to(device)
            edge_feats = edge_feats.to(device)
            label = labels.to(device)
            label = label.view(-1,1)
            adj_matrix = adj_matrix.to(device)

            output = model(graph,node_feats,edge_feats,protein_feats,adj_matrix)

            loss = loss_trans(output, label)
            optim.zero_grad()  # 清除先前梯度参数为0

            loss.backward()  # 反向传播计算 损失对所有可训练参数的梯度
            optim.step()

            running_loss += loss.item()
            train_round+=1
            if train_round % 1000==0:
                time_end = time.time()
                print('Training time:{}s'.format(time_end-time_start))
                print('The cross entropy loss for the {}th training is:{:.4f}'.format(train_round,loss))
            writer.add_scalar('realtime_Loss/Train_{}'.format(trial_id), loss.item(), global_step=train_round)       #输出每次训练的损失图像

        avg_train_loss = running_loss / len(dataloader_train)
        print('The average loss of the {}th round of training is {:.4f}'.format(epoch+1,avg_train_loss))

        valid_loss_total=0
        acc_total=0
        labelscore_predicted = []
        labels_true = []
        model.eval()

        with torch.no_grad():
            for i in dataloader_valid:
                graph, node_feats, edge_feats, protein_feats, labels ,adj_matrix= i  # 解包 tuple
                graph = graph.to(device)
                protein_feats = protein_feats.to(device)
                node_feats = node_feats.to(device)
                edge_feats = edge_feats.to(device)
                label = labels.to(device)
                label = label.view(-1,1)      # 数据准备
                adj_matrix = adj_matrix.to(device)  # 数据准备

                output = model(graph, node_feats, edge_feats, protein_feats, adj_matrix)
                loss = loss_trans(output, label)
                valid_loss_total += loss.item()  * label.size(0)   #验证集总损失

                label_probs = torch.sigmoid(output)  # 将 logits 转为概率
                predicted = (label_probs > 0.5).float()      #是否有活性
                acc = (predicted == label).sum().item()         #验证集是否正确
                acc_total += acc        #验证集总正确个数
                labels_true.extend(label.cpu().numpy().flatten())  # ROC
                labelscore_predicted.extend(label_probs.cpu().numpy().flatten())

        avg_valid_loss = valid_loss_total / len(dataloader_valid.dataset)
        avg_valid_acc = acc_total / len(dataloader_valid.dataset)

        auc = ROC_AUC(labelscore_predicted, labels_true,num=trial_number)
        print('The average accuracy on the validation set is:{:.4f}'.format(avg_valid_acc))
        print('Total loss on the validation set:{:.4f}'.format(valid_loss_total))
        print('Average loss on the validation set:{:.4f}'.format(avg_valid_loss))
        print('AUC on the validation set:{:.4f}'.format(auc))

        writer.add_scalar('avg_Loss/Train_{}'.format(trial_id), avg_train_loss, global_step=epoch)
        writer.add_scalar('avg_Loss/Valid_{}'.format(trial_id), avg_valid_loss, global_step=epoch)
        writer.add_scalar('Accuracy/Valid_{}'.format(trial_id), avg_valid_acc, global_step=epoch)

        # Early stopping check
        stop_flag, best_valid_loss, epochs_without_improvement, best_model_state, best_valid_acc, = (
            early_stopping(avg_valid_loss,
                           avg_valid_acc,
                           best_valid_loss,
                           model,
                           epoch,
                           epochs_without_improvement,
                           best_model_state=best_model_state,
                           best_valid_acc=best_valid_acc
                           ))
        if stop_flag == 'end':
            break

    # 恢复最好的模型
    if best_model_state is not None:
        model.load_state_dict(best_model_state)
        model.eval()

        save_dir = "ablation_model_pth"
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        if trial_id is not None:
            torch.save(model.state_dict(), os.path.join(save_dir, "best_model{}.pth".format(trial_id)))
        else:
            torch.save(model.state_dict(), "ablation_model_pth/best_model.pth")

        print('========================================================')
        print('Average loss on the validation set for the best model:{:.5f}'.format(best_valid_loss))
        print('The accuracy of the best model:{:.4f}'.format(best_valid_acc))
        print("The best model has been saved as best_model.pth")


    return best_valid_loss, best_valid_acc


def Bayesian(loss_trans,
             trials,
             device,
             dataset_train,
             dataset_valid
             ):
    writer_by_path = 'log_by'
    writer_by = SummaryWriter(writer_by_path)
    def objective(trial):       # 贝叶斯优化目标函数
        trial_num = trial.number
        writer_path='log_by/log_by_{}'.format(trial_num)
        writer = SummaryWriter(writer_path)
        mpnn = MPNNPredictorWithProtein().to(device)  # 🎯 重新初始化模型

        mpnn.load_state_dict(torch.load('best_model0.pth', weights_only=False))  # 加载训练好的权重

        learning_rate = trial.suggest_float('learning_rate', 1e-5, 1e-3,log=False)
        weight_decay = trial.suggest_float('weight_decay', 1e-4, 1e-2,log=False)
        batch_size = trial.suggest_categorical('batch_size', [64,128])
        epochs = trial.suggest_int('epochs', 50, 100)

        best_valid_loss, best_valid_acc = train_valid(
            model=mpnn,
            loss_trans=loss_trans,
            learning_rate=learning_rate,
            weight_decay=weight_decay,
            batch_size=batch_size,
            epochs=epochs,
            device=device,
            dataset_train=dataset_train,
            dataset_valid=dataset_valid,
            trial_id=trial.number,
            writer=writer,
            trial_number=trial_num
        )
        trial.set_user_attr("valid_acc", float(best_valid_acc))

        writer_by.add_scalar('best_Loss/Valid', best_valid_loss, global_step=trial_num)
        writer_by.add_scalar('best_Accuracy/Valid', best_valid_acc, global_step=trial_num)
        writer_by.add_scalar('best_Accuracy/lr', best_valid_acc,
                             global_step=trial.params['learning_rate'])
        writer_by.add_scalar('best_Accuracy/wd', best_valid_acc,
                             global_step=trial.params['weight_decay'])

        return  best_valid_acc

    # study = optuna.create_study(direction='minimize')  # 最小化验证损失
    study = optuna.create_study(direction='maximize',
                                pruner=optuna.pruners.HyperbandPruner(min_resource=10),
                                sampler=optuna.samplers.TPESampler())  # 最大化正确率
    study.optimize(objective, n_trials=trials)  # 优化20次试验

    print('Best trial:')
    best_trial = study.best_trial
    print(f'  Trail: {best_trial.number}')
    print(f'  Average Accuracy: {best_trial.user_attrs["valid_acc"]:.4f}')
    print(f'  Loss Value: {best_trial.value}')
    print(f'  Params: {best_trial.params}')


def test(model,
         dataset_test,
         batch_size,
         loss_trans,
         device,
         Threshold,
         seed=None
         ):
    if seed is not None:
        set_seed()
    else:
        set_seed(seed)

    writer = SummaryWriter('test_logs')

    dataloader_test = DataLoader(dataset_test, batch_size=batch_size, collate_fn=MolecularDataset.collate_fn,
                                 num_workers=0, pin_memory=False)
    valid_loss_total = 0
    acc_total = 0
    labels_true = []
    labelscore_predicted = []
    counter = 0

    # <<< 新加：保存预测的标签
    labels_pred_binary = []

    # model.eval()
    with torch.no_grad():
        for i in dataloader_test:
            graph, node_feats, edge_feats, protein_feats, labels ,adj_matrix= i  # 解包 tuple
            graph = graph.to(device)
            protein_feats = protein_feats.to(device)
            node_feats = node_feats.to(device)
            edge_feats = edge_feats.to(device)
            label = labels.to(device)
            label = label.view(-1, 1)
            adj_matrix = adj_matrix.to(device)      # 数据准备

            output = model(graph, node_feats, edge_feats, protein_feats,adj_matrix)
            loss = loss_trans(output, label)
            valid_loss_total += loss.item() * label.size(0)  # 测试集集总损失

            label_probs = torch.sigmoid(output)
            predicted = (label_probs > Threshold).float()  # 是否有活性

            acc = (predicted == label).sum().item()  # 测试集是否正确
            acc_total += acc  # 验证集总正确个数

            labels_true.extend(label.cpu().numpy().flatten())  # ROC
            labelscore_predicted.extend(label_probs.cpu().numpy().flatten())

            counter += 1

            # <<< 新加：收集预测标签用于计算f1/mcc
            labels_pred_binary.extend(predicted.cpu().numpy().flatten())

            if counter % 100 == 0:
                print('已计算{}组数据'.format(counter))



    tn, fp, fn, tp = confusion_matrix(labels_true, labels_pred_binary).ravel()
    # <<< 新加：最终统计f1分数和mcc
    precision = precision_score(labels_true, labels_pred_binary)
    recall = recall_score(labels_true, labels_pred_binary)
    specificity = tn / (tn + fp)
    f1 = f1_score(labels_true, labels_pred_binary)
    acc=acc_total / len(dataloader_test.dataset)
    mcc = matthews_corrcoef(labels_true, labels_pred_binary)
    auc = ROC_AUC(labelscore_predicted, labels_true)
    print('测试集上的正确率：{:.4f}'.format(acc))
    print('测试集上的总损失：{:.4f}'.format(valid_loss_total))
    print('测试集F1分数:{:.4f}'.format(f1))
    print('测试集MCC分数:{:.4f}'.format(mcc))
    print("测试集上的 ROC-AUC: {:.4f}".format(auc))
    print('测试集上的recall:{:.4f}'.format(recall))
    print('测试集上的precision:{:.4f}'.format(precision))
    print('测试集上的Specificity:{:.4f}'.format(specificity))

    writer.add_scalar('test_loss_total', valid_loss_total)  # 输出每次测试的总损失图
    writer.add_scalar('test_acc', acc_total / len(dataloader_test.dataset))  # 输出每次测试的正确率

    return auc,acc,mcc,f1


def ROC_AUC( labelscore_predicted,labels_true,num=0):
    save_dir = "auc"
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    # 计算 FPR, TPR 和 阈值
    fpr, tpr, thresholds = roc_curve(labels_true, labelscore_predicted)
    roc_auc = auc(fpr, tpr)  # 计算 AUC 值

    # 绘制 ROC 曲线
    plt.figure()
    plt.plot(fpr, tpr, color='darkorange', lw=1.5, label=f'ROC curve (AUC = {roc_auc:.4f})')
    plt.plot([0, 1], [0, 1], color='navy', linestyle='--')  # 随机分类器参考线
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate (FPR)')
    plt.ylabel('True Positive Rate (TPR)')
    plt.title('ROC Curve')
    plt.legend(loc='lower right')
    plt.savefig('auc/roc_curve{}.png'.format(num), dpi=500)
    plt.close()
    return roc_auc

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print('using device:', device)

if __name__ == '__main__':
    while True:
        code= input('mission:')
        # code='bayesian'
        if code == 'train':
            writer_path = 'ablation_log_train'
            writer = SummaryWriter(writer_path)

            loss_trans = BCEWithLogitsLoss().to(device)
            mpnn = MPNNPredictorWithProtein().to(device)
            train_file_path = 'trainset.csv'
            valid_file_path = 'validset.csv'
            dataset_train = MolecularDataset.loading_data(train_file_path)
            dataset_valid = MolecularDataset.loading_data(valid_file_path)

            train_valid(model=mpnn,
                    loss_trans=loss_trans,
                    device=device,
                    learning_rate=0.0007451954932192549,
                    weight_decay=0.005574660133633517,
                    batch_size=64,
                    epochs=20,
                    dataset_train=dataset_train,
                    dataset_valid=dataset_valid,
                    writer=writer
                    )
            break

        elif code == 'bayesian':

            loss_trans = BCEWithLogitsLoss().to(device)

            train_file_path = 'train.csv'
            valid_file_path = 'val.csv'

            dataset_train = MolecularDataset.loading_data(train_file_path, device=device)
            dataset_valid = MolecularDataset.loading_data(valid_file_path, device=device)
            Bayesian(
                loss_trans=loss_trans,
                device=device,
                trials=50,
                dataset_train=dataset_train,
                dataset_valid=dataset_valid,
                )
            break

        elif code == 'test':
            loss_trans = BCEWithLogitsLoss().to(device)

            mpnn = MPNNPredictorWithProtein().to(device)
            mpnn.load_state_dict(torch.load('best_model.pth',weights_only=False) ) # 加载训练好的权重
            mpnn.eval()

            test_file_path = 'testset.csv'
            dataset_test = MolecularDataset.loading_data(test_file_path)
            test(model=mpnn,
                dataset_test=dataset_test,
                batch_size=64,
                loss_trans=loss_trans,
                device=device,
                Threshold=0.5,
                seed=4
                )
            break

        else:
            print('code erro')




