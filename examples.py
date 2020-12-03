import torch.optim as optim
from robust_compression import *
from nn_models import ResNet18, Lenet5, VGG16, MobileNetV2
from nn_datasets import get_train_test_val_dataloaders
from dnn2dnn_compression import train_hypothesis
from nn_train import eval_model, calc_num_correct
from allowed_lables_loss import allowed_labels_loss
import datasets
from sklearn.ensemble import RandomForestClassifier
from consistency import MCConsistentTree
from utils import calc_allowed_labels


def crembo_dnn2dnn_example():
    # set device
    device = 'cuda'

    # set arguments
    args = {
        'device': device,
        'epochs': 90,
        'batch_size': 128,
        'pin_memory': True,
        'num_workers': 4,
        'step_size': 70,
        'gamma': 0.1,
        'lr': 1e-2,
        'model_name': 'lenet'
    }

    # Load large trained model
    M = ResNet18()
    model_path = 'outputs/models/M_resnet18_cifar10_0.9315.pt'
    M.load_state_dict(torch.load(model_path, map_location=device))
    M.to(device)
    M.eval()

    # Get dataLoaders
    train_loader, test_loader, valid_loader = get_train_test_val_dataloaders('cifar10', args['batch_size'],
                                                                             args['pin_memory'], num_workers=args['num_workers'],
                                                                             valid_size=0.1)
    # define create_model method
    create_model_func = create_model

    # define train_hypothesis method
    train_hypothesis_func = train_hypothesis

    # define evel_model method
    eval_model_func = eval_model

    # initiate CREMBO class
    crembo = CREMBO(create_model_func, train_hypothesis_func, eval_model_func, args)

    # run crembo
    f = crembo(M, train_loader, test_loader, valid_loader, device)

    # print scores
    f_score = eval_model(f, test_loader, device)
    M_score = eval_model(M, test_loader, device)
    print(f'M score: {M_score}, CREMBO: {f_score}')

    return f



# Example of training function that uses allowed labels loss
def train_hypothesis(args, model, M, d, train_dataloader, test_dataloader, device):
    M.eval()
    optimizer = optim.Adam(model.parameters(), lr=args['lr'])
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=args['step_size'], gamma=args['gamma'])

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

            # allowed label loss
            loss = allowed_labels_loss(outputs, M_outputs, d, device)

            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()
            num_samples += labels.size(0)
            correct_pred += calc_num_correct(outputs, labels)

        accuracy = correct_pred / num_samples
        print(f'Epoch: {epoch} Epoch loss: {epoch_loss}. Epoch Accuracy: {accuracy}')
        val_accuracy = eval_model(model, test_dataloader, device)
        print(f'Epoch: {epoch} Validation Accuracy: {val_accuracy}')

        scheduler.step()

    save_path = f'outputs/Final_acc_{val_accuracy}.pt'
    torch.save(model.state_dict(), save_path)

    return model


# Example for create_model function
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


# sklearn example
def crembo_sklearn_example():
    # set arguments
    dataset = 'dermatology'
    args = {
        'dataset': 'iris',
        'num_trees': 100,
        'tree_depth': 4,
        'forest_depth': 10,
        'weight': 'balanced',
        'sklearn': True
    }

    # Create train, test, val sets
    x, y, X_test, y_test = datasets.prepare_data(dataset, return_test=True)
    X_train, X_val, y_train, y_val = datasets.prepare_val(x, y)
    train_loader = (X_train, y_train)
    test_loader = (X_test, y_test)
    val_loader = (X_val, y_val)

    # train large model
    M = RandomForestClassifier(n_estimators=args['num_trees'], max_depth=args['forest_depth'], class_weight=args['weight'])
    M.fit(X_train, y_train)

    # define create_model method
    # Here we create a tree model. All sklearn models should be wrapped by a class that
    # inherits from the MCSkLearnConsistensy class.
    create_model_func = MCConsistentTree(depth=args['tree_depth'], class_weight=args['weight']).get_clone

    # define train_hypothesis method
    train_hypothesis_func = train_sklearn

    # define evel_model method
    eval_model_func = eval_sklearn

    # initiate CREMBO class
    crembo = CREMBO(create_model_func, train_hypothesis_func, eval_model_func, args, delta=1)

    # run crembo
    f = crembo(M, train_loader, test_loader, val_loader, device=None)

    # print scores
    f_score = eval_model_func(f, test_loader, None)
    M_score = eval_model_func(M, test_loader, None)
    print(f'M score: {M_score}, CREMBO: {f_score}')

    return f


# Example of a training function for sklearn models
def train_sklearn(args, model, M, d, train_dataloader, test_dataloader, device=None):
    X_train, y_train = train_dataloader

    # get M predictions
    preds = M.predict_proba(X_train)

    # calculate allowed labels
    Y, y = calc_allowed_labels(preds, d)

    # train m
    model.train(X_train, y)

    return model

# Example of an evaluation function for sklearn models
def eval_sklearn(model, loader, device=None):
    X, y = loader
    score = model.score(X, y)
    return score


if __name__ == '__main__':
    crembo_sklearn_example()
    # crembo_dnn2dnn_example()
