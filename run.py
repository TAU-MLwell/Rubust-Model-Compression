import argparse
import nn_compression
import rf_compression


def run_experiments():
    p = argparse.ArgumentParser()

    # required arguments
    p.add_argument('-model', type=str, required=True, help='Type of model to compress (rf or nn)')
    p.add_argument('-experiment_name', type=str, required=True, help='Experiment Name')
    p.add_argument('-dataset_name', type=str, required=True, help='Dataset name')

    # optional arguments
    p.add_argument("-n_experiments", type=int, default=20, help="Number of experiments")
    p.add_argument("-tree_depth", type=int, default=4, help="Maximum tree depth")
    p.add_argument("-score", type=str, default='accuracy', help="Score to calculate")
    p.add_argument("-weight", type=str, default='balanced', help="Weight")
    p.add_argument("-delta", type=int, default=1, help="Delta on optimization stage")
    p.add_argument("-rf_depth", type=int, default=12, help="Depth of Random Forest trees")
    p.add_argument("-rf_trees", type=int, default=100, help="Number of trees in Random Forest")
    args = p.parse_args()

    if args.model == 'nn':
        if args.experiment_name == 'generalization':
            experiment = nn_compression.generalization_exp
        elif args.experiment_name == 'robustness':
            experiment = nn_compression.robustness_agreement_exp
        else:
            raise NotImplementedError('No such experiment')

        experiment(args.tree_depth, num_experiments=args.n_experiments, dataset=args.dataset_name,
                   score=args.score, weight=args.weight, delta=args.delta)
    elif args.model == 'rf':
        if args.experiment_name == 'generalization':
            experiment = rf_compression.generalization_exp
        elif args.experiment_name == 'robustness':
            experiment = rf_compression.robustness_agreement_exp
        else:
            raise NotImplementedError('No such experiment')

        experiment(args.dataset_name, args.tree_depth, args.rf_depth, args.rf_trees,
                   num_experiments=args.n_experiments, score=args.score, weight=args.weight,
                   delta=args.delta)
    else:
        raise NotImplementedError('No such model')


if __name__ == '__main__':
    print('ewt')
    run_experiments()
