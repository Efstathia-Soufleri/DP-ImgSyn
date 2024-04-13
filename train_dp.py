import os
import multiprocessing

def main():
    import argparse
    import torch
    import logging
    from utils.str2bool import str2bool
    from utils.inference import inference
    from utils.averagemeter import AverageMeter
    from opacus.validators import ModuleValidator
    from opacus import PrivacyEngine
    from opacus.utils.batch_memory_manager import BatchMemoryManager
    import numpy as np
    import random
    from prv_accountant import PoissonSubsampledGaussianMechanism, PRVAccountant
    import registry

    parser = argparse.ArgumentParser(description='Train', formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    # Training parameters
    parser.add_argument('--epochs',                 default=3,             type=int,       help='Set number of epochs')
    parser.add_argument('--dataset',                default='CIFAR10',      type=str,       help='Set dataset to use')
    parser.add_argument('--lr',                     default=0.01,           type=float,     help='Learning Rate')
    parser.add_argument('--test_accuracy_display',  default=True,           type=str2bool,  help='Test after each epoch')
    parser.add_argument('--optimizer',              default='SGD',          type=str,       help='Optimizer for training')
    parser.add_argument('--loss',                   default='crossentropy', type=str,       help='Loss function for training')
    parser.add_argument('--resume',                 default=False,          type=str2bool,  help='Resume training from a saved checkpoint')
    parser.add_argument('--gpu',                    default=1,              type=int,       help='Which GPU to use')

    # Dataloader args
    parser.add_argument('--train_batch_size',       default=128,            type=int,       help='Train batch size')
    parser.add_argument('--test_batch_size',        default=256,            type=int,       help='Test batch size')
    parser.add_argument('--val_split',              default=0.1,            type=float,     help='Fraction of training dataset split as validation')
    parser.add_argument('--augment',                default=True,           type=str2bool,  help='Random horizontal flip and random crop')
    parser.add_argument('--padding_crop',           default=4,              type=int,       help='Padding for random crop')
    parser.add_argument('--shuffle',                default=True,           type=str2bool,  help='Shuffle the training dataset')
    parser.add_argument('--random_seed',            default=0,              type=int,       help='Initializing the seed for reproducibility')
    parser.add_argument('--root_path',              default="<set_path>",
                                                                            type=str,       help="Root path for the datasets")

    # Model parameters
    parser.add_argument('--save_seed',              default=False,          type=str2bool,  help='Save the seed')
    parser.add_argument('--use_seed',               default=False,          type=str2bool,  help='For Random initialization')
    parser.add_argument('--suffix',                 default='t',            type=str,       help='Appended to model name')
    parser.add_argument('--model',                  default='resnet34',     type=str,       help='Network architecture')

    # Summary Writer Tensorboard
    parser.add_argument('--comment',                default="",             type=str,       help='Comment for tensorboard')

    # Differential Privacy Parameters
    parser.add_argument('--noise_multiplier',       default=1.2,            type=float,     help='How much noise to add')
    parser.add_argument('--max_norm',               default=1.2,            type=float,     help='How much clip grad')

    global args
    args = parser.parse_args()

    MAX_GRAD_NORM = args.max_norm
    EPSILON = 50.0
    DELTA = 1e-5
    MAX_PHYSICAL_BATCH_SIZE = 128

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
    logger = logging.getLogger(f'Train Logger')
    logger.setLevel(logging.INFO)

    handler = logging.FileHandler(os.path.join('./logs', f'train_dp_{args.dataset}_{args.model}_nm{args.noise_multiplier}_{args.suffix}.log'), encoding="UTF-8")
    formatter = logging.Formatter(
        fmt=u"%(asctime)s %(levelname)-8s \t %(message)s",
        datefmt='%Y-%m-%d %H:%M:%S')
    handler.setFormatter(formatter)
    logger.addHandler(handler)

    logger.info(args)

    # Parameters
    num_epochs = args.epochs
    learning_rate = args.lr

    # Setup right device to run on
    device = torch.device(f'cuda:{args.gpu}' if torch.cuda.is_available() else 'cpu')

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

    # Loss
    criterion = torch.nn.CrossEntropyLoss()

    net = net.to(device)

    net = ModuleValidator.fix(net)
    ModuleValidator.validate(net, strict=False)
    privacy_engine = PrivacyEngine()

    # Optimizer
    optimizer = torch.optim.SGD(
        net.parameters(),
        lr=learning_rate,
        momentum=0.9,
        weight_decay=5e-4)

    # Learning rate scheduler
    scheduler = torch.optim.lr_scheduler.MultiStepLR(
        optimizer,
        milestones=[int(0.6*args.epochs), int(0.8*args.epochs)],
        gamma=0.1)

    net, optimizer, train_loader = privacy_engine.make_private(
        module=net,
        optimizer=optimizer,
        data_loader=train_loader,
        noise_multiplier=args.noise_multiplier,
        max_grad_norm=MAX_GRAD_NORM,
    )

    # sampling_probability = batch_size (not physical, but logical batch size) / length_of_dataset
    sampling_probability = 1 / len(train_loader) 
    prv = PoissonSubsampledGaussianMechanism(
        sampling_probability=sampling_probability,
        noise_multiplier=args.noise_multiplier)

    # Train model
    for epoch in range(0, num_epochs, 1):
        net.train()
        train_correct = 0.0
        train_total = 0.0
        save_ckpt = False
        losses = AverageMeter('Loss', ':.4e')
        logger.info('')
        with BatchMemoryManager(
            data_loader=train_loader, 
            max_physical_batch_size=MAX_PHYSICAL_BATCH_SIZE, 
            optimizer=optimizer
        ) as memory_safe_data_loader:
            for batch_idx, (data, labels) in enumerate(memory_safe_data_loader):
                data = data.to(device)
                labels = labels.to(device)
                
                # Clears gradients of all the parameter tensors
                optimizer.zero_grad()
                out = net(data)
                loss = criterion(out, labels)
                loss.backward()
                optimizer.step()
                losses.update(loss.item())

                train_correct += (out.max(-1)[1] == labels).sum().long().item()
                train_total += labels.shape[0]

                if (batch_idx + 1) % 100 == 0:
                    curr_acc = 100. * train_correct / train_total
                    epsilon = privacy_engine.get_epsilon(DELTA)
                    logger.info(
                        f"Train Epoch: {epoch} \t"
                        f"Loss: {losses.avg:.6f} "
                        f"Acc@1: {curr_acc:.6f} "
                        f"(ε = {epsilon:.2f}, δ = {DELTA})"
                    )

                    # num_comp = privacy_engine.accountant.history[0][2]
                    # verify_acc = PRVAccountant(
                    #     prvs=[prv],
                    #     max_self_compositions=[num_comp],
                    #     eps_error=0.01,
                    #     delta_error=1e-8
                    # )

                    # _, _, eps_up = verify_acc.compute_epsilon(delta=DELTA, num_self_compositions=[num_comp])
                    # logger.info(f'Verify Opacus Calc (ε = {eps_up:.2f}, δ = {DELTA})')
        
        train_accuracy = float(train_correct) * 100.0 / float(train_total)
        logger.info(
            'Train Epoch: {} Accuracy : {}/{} [ {:.2f}%)]\tLoss: {:.6f}'.format(
                epoch,
                train_correct,
                train_total,
                train_accuracy,
                losses.avg))
       
        # Step the scheduler by 1 after each epoch
        scheduler.step()
        
        val_correct, val_total, val_accuracy, val_loss = -1, -1, -1, -1
        val_accuracy= float('inf')
        save_ckpt = True

        def save_model_with_dp_accountant(model, accountant, args, model_name):
            save_dict = {
                'model': model.state_dict(),
                'dp_accountant': accountant
            }

            torch.save(save_dict, './checkpoints/'+ model_name + '_.dppt')

        saved_training_state = {    
            'epoch'     : epoch + 1,
            'optimizer' : optimizer.state_dict(),
            'model'     : net.state_dict(),
            'dp_accountant': privacy_engine.accountant }

        torch.save(saved_training_state, './checkpoints/' + model_name  + '.temp')
        
        if save_ckpt:
            logger.info("Saving checkpoint...")
            save_model_with_dp_accountant(net, privacy_engine.accountant, args, model_name)
            if args.test_accuracy_display:
                # Test model
                # Set the model to eval mode
                test_correct, test_total, test_accuracy = inference(
                    net=net,
                    data_loader=test_loader,
                    device=device)

                logger.info(
                    " Training set accuracy: {}/{}({:.2f}%) \n" 
                    " Validation set accuracy: {}/{}({:.2f}%)\n"
                    " Test set: Accuracy: {}/{} ({:.2f}%)".format(
                        train_correct,
                        train_total,
                        train_accuracy,
                        val_correct,
                        val_total,
                        val_accuracy,
                        test_correct,
                        test_total,
                        test_accuracy))

    logger.info("End of training without reusing Validation set")
       

if __name__ == "__main__":
    if os.name == 'nt':
        # On Windows calling this function is necessary for multiprocessing
        multiprocessing.freeze_support()
    
    main()

