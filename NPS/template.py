def set_template(args):
    # Set the templates here
    if args.template == 'resnet':
        args.epochs = 200
        args.lr = 5e-5
        args.lr_decay = 150

