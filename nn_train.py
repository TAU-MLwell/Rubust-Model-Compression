import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as data_utils
from sklearn.preprocessing import StandardScaler
from oracles import Oracle


def calc_num_correct(predictions, labels):
    _, predicted = torch.max(predictions.data, 1)
    correct = (predicted == labels).sum().item()
    return correct


def eval_model(model, data_set, device):
    model.eval()

    correct_pred = 0
    num_samples = 0
    with torch.no_grad():
        for i, (samples, labels) in enumerate(data_set):
            labels = labels.to(device, dtype=torch.long)
            samples = samples.to(device, dtype=torch.float)
            output = model(samples)

            num_samples += labels.size(0)
            correct_pred += calc_num_correct(output, labels)

    return correct_pred / num_samples


def train_loop(args, model, train_dataloader, test_dataloader, device):
    optimizer = optim.Adam(model.parameters(), lr=args['lr'])
    criterion = nn.CrossEntropyLoss()

    comment = f'{args["experiment_name"]}_{args["dataset_name"]}_epochs_{args["epochs"]}_device_{device}'
    print(comment)

    best = 0
    best_path = None
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
            loss = criterion(outputs, labels)

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
                best_path = args['save_dir'] + args['experiment_name'] + f'_Best_acc_{best}' '.pt'
                torch.save(model.state_dict(), best_path)

    save_path = args['save_dir'] + args['experiment_name'] + f'_Final_acc_{val_accuracy}' '.pt'
    torch.save(model.state_dict(), save_path)

    best_path = save_path if best_path is None else best_path
    return best_path, best



def dnn_train_loop(args, model, train_dataloader, test_dataloader, device):
    optimizer = optim.Adam(model.parameters(), lr=args['lr'])
    criterion = nn.CrossEntropyLoss()

    comment = f'{args["experiment_name"]}_{args["dataset_name"]}_epochs_{args["epochs"]}_device_{device}'
    print(comment)

    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=args['step_size'], gamma=args['gamma'])
    best = 0
    best_path = None
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
            loss = criterion(outputs, labels)

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
                best_path = args['save_dir'] + args['experiment_name'] + f'_Best_acc_{best}' '.pt'
                torch.save(model.state_dict(), best_path)

        scheduler.step()

    save_path = args['save_dir'] + args['experiment_name'] + f'_Final_acc_{val_accuracy}' '.pt'
    torch.save(model.state_dict(), save_path)

    best_path = save_path if best_path is None else best_path
    return best_path, best


def train_oracle_and_predict(dataset_name ,X_train, y_train, X_test, y_test, num_outputs, device, model_name='uci'):
    args = {'dataset_name': dataset_name, 'experiment_name': dataset_name + '_CV', 'disable_gpu': False, 'model_name': model_name,
            'save_dir': 'outputs/models/', 'gpu': device, 'epochs': 10, 'lr': 0.01, 'batch_size': 32, 'validate_every': 1}

    o = Oracle(num_outputs, None, model_name=args['model_name'], device=device, in_features=X_train.shape[1])
    scaler = StandardScaler()

    nn_x_train = scaler.fit_transform(X_train)
    nn_x_train = torch.from_numpy(nn_x_train).float().to(device)
    nn_y_train = torch.from_numpy(y_train)
    train_dataset = data_utils.TensorDataset(nn_x_train, nn_y_train)
    train_dataloader = data_utils.DataLoader(train_dataset, batch_size=args['batch_size'], shuffle=True)

    nn_x_test = scaler.fit_transform(X_test)
    nn_x_test = torch.from_numpy(nn_x_test).float().to(device)
    nn_y_test = torch.from_numpy(y_test)
    test_dataset = data_utils.TensorDataset(nn_x_test, nn_y_test)
    test_dataloader = data_utils.DataLoader(test_dataset, batch_size=args['batch_size'], shuffle=False)


    o.set_train()
    best_path, best = train_loop(args, o.model, train_dataloader, test_dataloader, args['gpu'])

    o.load_model(best_path)
    train_dataloader = data_utils.DataLoader(train_dataset, batch_size=args['batch_size'], shuffle=False)
    p_ic = o.predict(train_dataloader)

    return p_ic, best
