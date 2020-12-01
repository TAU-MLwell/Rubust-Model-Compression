import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from nn_train import eval_model, calc_num_correct, dnn_train_loop
from nn_models import ResNet18, Lenet5, VGG16, MobileNetV2
from nn_datasets import get_dataloader, get_train_test_val_dataloaders
import argparse
import time
from utils import preds2lables


def count_parameters(model):
    parms = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f'# Parameters in model = {parms}')


def create_model(args):
    if args['model_name'] == 'resnet18':
        model = ResNet18()
    elif args['model_name'] == 'lenet':
        model = Lenet5()
    elif args['model_name'] == 'VGG16':
        model = VGG16()
    elif args['model_name'] == 'mobilenetv2':
        model = MobileNetV2()
    else:
        raise NotImplementedError('No such model')

    model.to(args['device'])
    return model


def get_depths(M, train_dataloader, device='cpu', temperature=1):
    M.eval()
    depths = None

    for i, (samples, labels) in enumerate(train_dataloader):
        samples = samples.to(device, dtype=torch.float)
        with torch.no_grad():
            M_outputs = M(samples)
            M_preds = F.softmax(M_outputs / temperature, dim=1)

        if depths is None:
            depths = M_preds
        else:
            depths = torch.cat([depths, M_preds], dim=0)


    depths[depths < 0.01] = 0
    depths[depths > 0.9] = 1
    depths = depths.flatten().sort()[0].unique()
    print(f'Len depths {len(depths)}')
    return depths


def is_consistent(model, M, train_dataloader, d, device, soft):
    model.eval()
    M.eval()

    for i, (samples, labels) in enumerate(train_dataloader):
        samples = samples.to(device, dtype=torch.float)
        with torch.no_grad():
            M_outputs = M(samples)
            M_preds = F.softmax(M_outputs, dim=1)
            Y = preds2lables(M_preds, d, soft=soft).to(device)

            predictions = model(samples)
            _, predicted = torch.max(predictions.data, 1)
            for j in range(Y.shape[0]):
                if Y[j][predicted[j]] == 0:
                    return False

    return True


def train_large_model(args):
    start_time = time.time()
    train_dataloader, test_dataloader, num_outputs = get_dataloader(args['dataset_name'],  args['batch_size'],
                                                                    pin_memory=args['pin_memory'], num_workers=args['num_workers'])
    model = create_model(args)
    count_parameters(model)
    model.to(args['device'])
    dnn_train_loop(args, model, train_dataloader, test_dataloader, args['device'])
    print(f'Train time {time.time() - start_time}')


def train_hypothesis(args, model, M, d, train_dataloader, test_dataloader, device):
    M.eval()
    optimizer = optim.Adam(model.parameters(), lr=args['lr'])
    criterion = nn.BCEWithLogitsLoss()
    bce_criterion = nn.BCELoss()
    ce_criterion = nn.CrossEntropyLoss()
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=args['step_size'], gamma=args['gamma'])
    temperature = args['temperature']
    comment = f'{args["experiment_name"]}_{args["dataset_name"]}_epochs_{args["epochs"]}_device_{device}'
    print(comment)

    best = 0
    for epoch in range(args['epochs']):
        model.train()
        epoch_loss = 0
        num_samples = 0
        correct_pred = 0
        for i, (samples, labels) in enumerate(train_dataloader):
            samples = samples.to(device, dtype=torch.float)
            labels = labels.to(device, dtype=torch.long)

            optimizer.zero_grad()
            outputs = model(samples)

            with torch.no_grad():
                M_outputs = M(samples)
            M_preds = F.softmax(M_outputs / temperature, dim=1)
            if args['experiment'] == 'kd' or args['kd']:
                soft_output = F.softmax(outputs / temperature, dim=1)

                # soft prediction loss
                soft_loss = bce_criterion(soft_output, M_preds) * (temperature ** 2)

                # cross entropy loss (hard predictions)
                ce_loss = ce_criterion(outputs, labels)

                loss = soft_loss + ce_loss
            else:
                Y = preds2lables(M_preds, d, args['soft']).to(device)
                if temperature == 1:
                    loss = criterion(outputs, Y)
                else:
                    loss = bce_criterion(F.softmax(outputs / temperature, dim=1), Y)

            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()
            num_samples += labels.size(0)
            correct_pred += calc_num_correct(outputs, labels)

        accuracy = correct_pred / num_samples
        print(f'Epoch: {epoch} Epoch loss: {epoch_loss}. Epoch Accuracy: {accuracy}')

        if (epoch + 1) % args['validate_every'] == 0:
            val_accuracy = eval_model(model, test_dataloader, device)
            print(f'Epoch: {epoch} Validation Accuracy: {val_accuracy}')
            if val_accuracy > best:
                best = val_accuracy
                best_path = args['save_dir'] + args['experiment_name'] + '.pt'
                torch.save(model.state_dict(), best_path)

        scheduler.step()

    save_path = args['save_dir'] + args['experiment_name'] + f'_Final_acc_{val_accuracy}' '.pt'
    torch.save(model.state_dict(), save_path)

    return model


def memo(args, M, train_dataloader, test_dataloader):
    thresholds = get_depths(M, train_dataloader, device=args['device'], temperature=args['temperature'])

    start = 0
    end = len(thresholds) - 1
    f_out = None
    idx = 0
    while start <= end:
        m = (end + start) // 2
        d = thresholds[m]
        model = create_model(args)
        f = train_hypothesis(args, model, M, d, train_dataloader, test_dataloader, args['device'])

        # check for consistency
        consistent = is_consistent(f, M, train_dataloader, d, args['device'],  soft=args['soft'])

        if consistent:
            f_out = f
            idx = m
            start = m + 1
        else:
            end = m - 1

    print(f'MEMO depth: {thresholds[idx]}')
    if f_out is None:
        f_out = model

    return f_out, thresholds, idx


def crembo(args, M, train_dataloader, test_dataloader, val_loader, delta=250):
    f, thresholds, idx = memo(args, M, train_dataloader, test_dataloader)

    start = idx
    end = len(thresholds) - 1
    best = eval_model(f, val_loader, args['device'])

    for m in range(start, end, delta):
        model = create_model(args)
        d = thresholds[m]
        h = train_hypothesis(args, model, M, d, train_dataloader, test_dataloader, args['device'])

        # evaluate score on validation set
        score = eval_model(h, val_loader, args['device'])

        if score > best:
            best = score
            idx = m
            f = h

    print(f'CREMBO val: {best} depth: {thresholds[idx].item()}')
    return f, thresholds, idx


def run_compression(args):
    start_time = time.time()
    if args['large_model'] == 'resnet18':
        M = ResNet18()
        model_path = 'outputs/models/M_resnet18_cifar10_0.9315.pt'
    elif args['large_model'] == 'VGG16':
        M = VGG16()
        model_path = 'outputs/models/M_vgg16_cifar10_0.9222.pt'
    else:
        raise NotImplementedError('No such experiment')

    M.load_state_dict(torch.load(model_path, map_location=args['device']))
    M.to(args['device'])
    M.eval()

    train_loader, test_loader, valid_loader = get_train_test_val_dataloaders(args['dataset_name'], args['batch_size'],
                                                                             args['pin_memory'], num_workers=args['num_workers'],
                                                                             valid_size=0.1)
    f, thresholds, idx = crembo(args, M, train_loader, test_loader, valid_loader, delta=20)

    f_score = eval_model(f, test_loader, args['device'])
    M_score = eval_model(M, test_loader, args['device'])

    print(f'M score: {M_score}, CREMBO: {f_score}')
    exp_name = args['experiment_name']
    torch.save(f.state_dict(), f'outputs/models/crembo_{exp_name}.pt')
    print(f'Depth: {thresholds[idx]}')
    print(f'Train time {time.time() - start_time}')


def run_kd(args):
    if args['large_model'] == 'resnet18':
        M = ResNet18()
        model_path = 'outputs/models/M_resnet18_cifar10_0.9315.pt'
    elif args['large_model'] == 'VGG16':
        M = VGG16()
        model_path = 'outputs/models/M_vgg16_cifar10_0.9222.pt'
    else:
        raise NotImplementedError('No such large model')

    M.load_state_dict(torch.load(model_path, map_location=args['device']))
    M.to(args['device'])
    M.eval()

    train_loader, test_loader, valid_loader = get_train_test_val_dataloaders(args['dataset_name'], args['batch_size'],
                                                                             args['pin_memory'], num_workers=args['num_workers'],
                                                                             valid_size=0.1)

    model = create_model(args)
    f = train_hypothesis(args, model, M, 0, train_loader, test_loader, args['device'])

    # performance
    f_score = eval_model(f, test_loader, args['device'])
    M_score = eval_model(M, test_loader, args['device'])

    print(f'M score: {M_score}, KD: {f_score}')
    exp_name = args['experiment_name']
    torch.save(f.state_dict(), f'outputs/models/kd_{exp_name}.pt')


def remote_runner():
    p = argparse.ArgumentParser()

    #required arguments
    p.add_argument('-dataset_name', type=str, required=True, help='Dataset name')
    p.add_argument('-experiment_name', type=str, required=True, help='Experiment name')
    p.add_argument("-model_name", type=str, required=True, help="Model to train")
    p.add_argument("-experiment", type=str, required=True, help="Type of experiment to run")

    #optional arguments
    p.add_argument("-large_model", type=str, default='resnet18', help="Large model to compress")
    p.add_argument("-disable_gpu", action='store_true', help="Dont use gpu")
    p.add_argument("-pin_memory", action='store_true', help="Load whole dataset to memory flag")
    p.add_argument("-save_dir", type=str, default="outputs/models/", help="Path to save model")
    p.add_argument("-device", type=str, default='cuda:0', help="Cuda gpu id to use")
    p.add_argument("-epochs", type=int, default=90, help="Number of training epochs")
    p.add_argument("-num_workers", type=int, default=0, help="Number of workers")
    p.add_argument("-validate_every", type=int, default=2, help="Run validation every this number of epochs")
    p.add_argument("-lr", type=float, default=1e-2, help="Training learning rate")
    p.add_argument("-momentum", type=float, default=0.9, help="SGD optimizer momentum")
    p.add_argument("-weight_decay", type=float, default=1e-5, help="L2 regularization coefficient")
    p.add_argument("-batch_size", type=int, default=128, help="Train batch size")
    p.add_argument("-step_size", type=int, default=60, help="lr step size")
    p.add_argument("-gamma", type=float, default=0.1, help="lr step value")
    p.add_argument("-temperature", type=int, default=1, help="Temperature")
    p.add_argument("-soft", action='store_true', help="Use soft labels")
    p.add_argument("-kd", action='store_true', help="Knowledge distillation")

    args = p.parse_args()
    args = vars(args)
    if args['experiment'] == 'compress':
        run_compression(args)
    elif args['experiment'] == 'train':
        train_large_model(args)
    elif args['experiment'] == 'kd':
        run_kd(args)
    else:
        raise NotImplementedError('No such experiment')


if __name__ == '__main__':
    remote_runner()
