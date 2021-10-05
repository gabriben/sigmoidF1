import logging
import timm

logger = logging.getLogger(__name__)

from ..tresnet import TResnetM, TResnetL, TResnetXL


def create_model(args):
    """Create a model
    """
    model_params = {'args': args, 'num_classes': args.num_classes}
    args = model_params['args']
    args.model_name = args.model_name.lower()

    if args.model_name=='tresnet_m':
        model = TResnetM(model_params)
    elif args.model_name=='tresnet_l':
        model = TResnetL(model_params)
    elif args.model_name=='tresnet_xl':
        model = TResnetXL(model_params)
    elif args.model_name == 'resnet101':
        # check if we are training
        if '/dbfs/models/' in args.model_path:
            print("create timm model")
            model = timm.create_model('resnet101', pretrained=True, num_classes=args.num_classes)
        # or validating
        else:
            print("load timm model")
            model = timm.create_model('resnet101', pretrained=False, num_classes=args.num_classes, checkpoint_path = args.model_path)
    else:
        print("model: {} not found !!".format(args.model_name))
        exit(-1)

    return model
