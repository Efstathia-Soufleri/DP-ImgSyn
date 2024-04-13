import os
import multiprocessing

def main():
    import argparse
    import torch
    import logging
    from utils.str2bool import str2bool
    from synthesizer.syn_fix import SynthesisModuleValidator
    import synthesizer.syn_GN_validator
    from utils.inference import inference
    import math
    import registry
    from opacus.validators import ModuleValidator
    from opacus import PrivacyEngine
    from opacus.utils.batch_memory_manager import BatchMemoryManager
    import numpy as np
    import random
    from prv_accountant import PoissonSubsampledGaussianMechanism, PRVAccountant

    parser = argparse.ArgumentParser(description='Train', formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    # Generation parameters
    parser.add_argument('--dataset',                default='mnist',      type=str,       help='Set dataset to use')
    parser.add_argument('--public_set',             default='SVHN',         type=str,       help='Public set used as initialization')
    parser.add_argument('--root_path',              default="<set_path>",
                                                                            type=str,       help="Root path for the datasets")
    parser.add_argument('--gpu',                    default=1,              type=int,       help='Which GPU to use')


    # Dataloader args
    parser.add_argument('--train_batch_size',       default=64,             type=int,       help='Train batch size')
    parser.add_argument('--test_batch_size',        default=64,            type=int,       help='Test batch size')
    parser.add_argument('--augment',                default=False,          type=str2bool,  help='Random horizontal flip and random crop')
    parser.add_argument('--shuffle',                default=False,          type=str2bool,  help='Shuffle the training dataset')
    parser.add_argument('--random_seed',            default=0,              type=int,       help='Initializing the seed for reproducibility')

    # Model parameters
    parser.add_argument('--save_seed',              default=False,          type=str2bool,  help='Save the seed')
    parser.add_argument('--use_seed',               default=False,          type=str2bool,  help='For Random initialization')
    parser.add_argument('--suffix',                 default='t',            type=str,       help='Appended to model name')
    parser.add_argument('--model',                  default='resnet18',     type=str,       help='Network architecture')

    # Differential Privacy Parameters
    parser.add_argument('--noise_multiplier',       default=0.8,          type=float,     help='Differential Privacy How much noise to add?')
    parser.add_argument('--max_norm',               default=1,            type=float,     help='Differential Privacy Maximum norm')

    global args
    args = parser.parse_args()

    MAX_GRAD_NORM = 1.2
    DELTA = 1e-5
    MAX_PHYSICAL_BATCH_SIZE = 64

    os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'
    torch.manual_seed(args.random_seed)
    torch.cuda.manual_seed(args.random_seed)
    np.random.seed(args.random_seed)
    random.seed(args.random_seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    version_list = list(map(float, torch.__version__.split(".")))
    if  version_list[0] == 1 and version_list[1] < 8: ## pytorch 1.8.0 or below
        torch.set_deterministic(True)
    else:
        torch.use_deterministic_algorithms(True)

    # Create a logger
    logger = logging.getLogger(f'Synthesis Logger')
    logger.setLevel(logging.INFO)

    handler = logging.FileHandler(os.path.join('./logs', f'Synthesis_DP_{args.suffix}.log'), encoding="UTF-8")
    formatter = logging.Formatter(
        fmt=u"%(asctime)s %(levelname)-8s %(message)s",
        datefmt='%Y-%m-%d %H:%M:%S')
    handler.setFormatter(formatter)
    logger.addHandler(handler)

    logger.info(args)

    # Parameters
    gpu_id = args.gpu

    # Setup right device to run on
    device = torch.device(f'cuda:{gpu_id}' if torch.cuda.is_available() else 'cpu')

    ############################################
    # Setup dataset
    ############################################
    num_classes, train_dataset, val_dataset = registry.get_dataset(name=args.dataset, data_root=args.root_path)

    train_sampler = None
    
    train_loader = torch.utils.data.DataLoader(
        train_dataset, 
        batch_size=args.train_batch_size, 
        shuffle=(train_sampler is None),
        num_workers=0, 
        pin_memory=True, 
        sampler=train_sampler)

    test_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=args.train_batch_size, 
        shuffle=False,
        num_workers=0, 
        pin_memory=True)
    
    args.suffix = f"nm_{args.noise_multiplier}_{args.suffix}"

    # Instantiate model 
    net = registry.get_model(args.model, num_classes=num_classes, pretrained=False)
    model_name = f"{args.dataset.lower()}_{args.model}_{args.suffix}"

    # Load model
    net = net.to(device)

    net = ModuleValidator.fix(net)
    ModuleValidator.validate(net, strict=False)

    dummy_optimizer = torch.optim.SGD(net.parameters(), lr=0.0)

    # Need to do this because privacy engine renames the net layer
    # So cannot load without
    privacy_engine = PrivacyEngine()
    net, dummy_optimizer, train_loader = privacy_engine.make_private(
        module=net,
        optimizer=dummy_optimizer,
        data_loader=train_loader,
        noise_multiplier=args.noise_multiplier,
        max_grad_norm=MAX_GRAD_NORM,
    )
    
    # Load pre-trained model
    load_path = os.path.join(
        "./checkpoints/", 
        model_name + "_.dppt")

    load_data = torch.load(load_path)
    net.load_state_dict(load_data['model'])

    # Start synthesis model
    net.eval()
    logger.info('')

    # Replace the GroupNorm Layer with GroupNorm_Record_Stats
    # This layer adds noise and saves the batch stats in a differentially private way
    net = SynthesisModuleValidator.fix(
        module=net, 
        **{
            'noise_multiplier': args.noise_multiplier,
            'max_norm': args.max_norm,
        })
    
    SynthesisModuleValidator.validate(net)

    # sampling_probability = batch_size (not physical, but logical batch size) / length_of_dataset
    sampling_probability = 1 / len(train_loader) 
    prv = PoissonSubsampledGaussianMechanism(
        sampling_probability=sampling_probability,
        noise_multiplier=args.noise_multiplier)
    
    train_acc = load_data['dp_accountant']
    train_sampling_prob = train_acc.history[0][1]
    train_noise_mp = train_acc.history[0][0]
    train_comp = train_acc.history[0][2]

    prv_for_training = PoissonSubsampledGaussianMechanism(
        sampling_probability=train_sampling_prob,
        noise_multiplier=train_noise_mp)
    
    prev_acc = PRVAccountant(
        prvs=[prv_for_training],
        max_self_compositions=[train_comp],
        eps_error=0.01,
        delta_error=1e-8
    )

    _, _, eps_up = prev_acc.compute_epsilon(delta=DELTA, num_self_compositions=[train_comp])
    logger.info(f"ε = {eps_up:.2f}, δ = {DELTA} from training")

    # First pass the dataset through the model, calculate the batch stats
    # in a differentially private way
    # Pass each image, get the batch stat, add gaussian noise to the batch stat
    # We do not update model parameters, thus this is an efficient way to
    # obtain the batch stats
    net.train()
    num_epochs = 5
    with torch.no_grad():
        for epochs in range(num_epochs):
            with BatchMemoryManager(
                data_loader=train_loader,
                max_physical_batch_size=MAX_PHYSICAL_BATCH_SIZE, 
                optimizer=dummy_optimizer
            ) as memory_safe_data_loader:
                for _, (data, _) in enumerate(memory_safe_data_loader):
                    data = data.to(device)
                    _ = net(data)

    # DP 
    net.eval()
    logger.info("Done, updated batch stats with DP")

    num_comp = math.ceil(len(train_loader) * num_epochs)
    verify_acc = PRVAccountant(
        prvs=[prv_for_training, prv],
        max_self_compositions=[train_comp, num_comp],
        eps_error=0.01,
        delta_error=1e-8
    )

    eps_low, eps_est, eps_up = verify_acc.compute_epsilon(delta=DELTA, num_self_compositions=[train_comp, num_comp])
    logger.info(f'Total ε = {eps_up:.2f}, δ = {DELTA}')    

    # Make net standard and not keep GradSampleModule from Opacus
    net = net.to('cpu')

    save_net = registry.get_model(args.model, num_classes=num_classes, pretrained=False)
    save_net = ModuleValidator.fix(save_net)
    save_net = SynthesisModuleValidator.fix(
        save_net,
        **{
            'noise_multiplier': args.noise_multiplier,
            'max_norm': args.max_norm,
        })
    
    save_net.load_state_dict(net._module.state_dict())
    save_net.to(device)
    test_correct, test_total, test_accuracy = inference(
        net=save_net,
        data_loader=test_loader,
        device=device)

    logger.info(
        "Test set: Accuracy: {}/{} ({:.2f}%)".format(
            test_correct,
            test_total,
            test_accuracy))

    # Save save net
    save_dict = {
                'model': save_net.state_dict(),
                'dp_accountant': verify_acc,
                'bn_nm': args.noise_multiplier,
                'eps': eps_up,
                'delta': DELTA,
                'training_dp_acc': prev_acc,
                'bn_epochs': num_epochs
            }

    torch.save(save_dict, './checkpoints/'+ model_name + f'_bn_eps{eps_up}.dppt')

if __name__ == "__main__":
    if os.name == 'nt':
        # On Windows calling this function is necessary for multiprocessing
        multiprocessing.freeze_support()
    
    main()

