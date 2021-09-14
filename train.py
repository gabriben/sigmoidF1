import numpy
import os
import argparse
import torch
import torch.nn.parallel
import torch.optim
import torch.utils.data.distributed
import torchvision.transforms as transforms
from torch.optim import lr_scheduler
from src.helper_functions.helper_functions import mAP, CocoDetection, CutoutPIL, ModelEma, add_weight_decay
from src.models import create_model
from src.loss_functions.losses import AsymmetricLoss, sigmoidF1
from randaugment import RandAugment
from torch.cuda.amp import GradScaler, autocast

#mlflow
import mlflow
import mlflow.pytorch
mlflow.set_experiment("/Users/gabriel.benedict@rtl.nl/multilabel/PASCAL-VOC/ASL run")
import tempfile
import tensorflow as tf
import pytorch_lightning

parser = argparse.ArgumentParser(description='PyTorch MS_COCO Training')
parser.add_argument('--data', help='path to dataset', default='/dbfs/datasets/coco', type=str) # , metavar='DIR'
parser.add_argument('--lr', default=1e-4, type=float)
parser.add_argument('--model-name', default='tresnet_m')
parser.add_argument('--model-path', default='/dbfs/models/tresnet_m.pth', type=str)
parser.add_argument('--num-classes', default=80)
parser.add_argument('-j', '--workers', default=8, type=int, metavar='N',
                    help='number of data loading workers (default: 16)')
parser.add_argument('--image-size', default=224, type=int,
                    metavar='N', help='input image size (default: 448)')
parser.add_argument('--thre', default=0.8, type=float,
                    metavar='N', help='threshold value')
parser.add_argument('-b', '--batch-size', default=128, type=int,
                    metavar='N', help='mini-batch size (default: 16)')
parser.add_argument('--print-freq', '-p', default=64, type=int,
                    metavar='N', help='print frequency (default: 64)')
parser.add_argument('--num-epochs', '-e', default=1, type=int,
                    metavar='N', help='number of epochs (default: 1)')
parser.add_argument('--stop-epoch', '-se', default=40, type=int,
                    metavar='N', help='? stop epoch ? (default: 40)')
parser.add_argument('--weight-decay', '-wd', default=1e-4, type=int,
                    metavar='N', help='weight decay (default: 1e-4)')
parser.add_argument('--loss-function', '-lo', default="ASL", type=str,
                    metavar='N', help='loss function a.k.a criterion (default: ASL)')
parser.add_argument('--slope', '-s', default=-1, type=float,
                    metavar='N', help='Slope of the sigmoid function loss')
parser.add_argument('--offset', '-off', default=0, type=float,
                    metavar='N', help='offset of the sigmoid function loss')

#https://stackoverflow.com/questions/44542605/python-how-to-get-all-default-values-from-argparse/44543594#:~:text=def%20get_argparse_defaults(parser)%3A%0A%20%20%20%20defaults%20%3D%20%7B%7D%0A%20%20%20%20for%20action%20in%20parser._actions%3A%0A%20%20%20%20%20%20%20%20if%20not%20action.required%20and%20action.dest%20!%3D%20%22help%22%3A%0A%20%20%20%20%20%20%20%20%20%20%20%20defaults%5Baction.dest%5D%20%3D%20action.default%0A%20%20%20%20return%20defaults
def get_argparse_defaults(parser):
    defaults = {}
    for action in parser._actions:
        if not action.required and action.dest != "help":
            defaults[action.dest] = action.default
    return defaults

#https://stackoverflow.com/questions/2352181/how-to-use-a-dot-to-access-members-of-dictionary#:~:text=to%20help%20you%3A-,class,-Map(dict)%3A%0A%20%20%20%20%22%22%22%0A%20%20%20%20Example
class Map(dict):
    """
    Example:
    m = Map({'first_name': 'Eduardo'}, last_name='Pool', age=24, sports=['Soccer'])
    """
    def __init__(self, *args, **kwargs):
        super(Map, self).__init__(*args, **kwargs)
        for arg in args:
            if isinstance(arg, dict):
                for k, v in arg.items():
                    self[k] = v

        if kwargs:
            for k, v in kwargs.items():
                self[k] = v

    def __getattr__(self, attr):
        return self.get(attr)

    def __setattr__(self, key, value):
        self.__setitem__(key, value)

    def __setitem__(self, key, value):
        super(Map, self).__setitem__(key, value)
        self.__dict__.update({key: value})

    def __delattr__(self, item):
        self.__delitem__(item)

    def __delitem__(self, key):
        super(Map, self).__delitem__(key)
        del self.__dict__[key]

def main(ep = 1, loss = "ASL", data = '/dbfs/datasets/coco', num_classes = 80, E = 1, S = -9):
    # try: # run from shell with arguments
    #   args = parsxer.parse_args()
    #   args.do_bottleneck_head = False
    #   #mlflow
    #   for key, value in vars(args).items():
    #       mlflow.log_param(key, value)

        
    # except Exception as e: #run as import from python
    args = get_argparse_defaults(parser)
    args = Map(args)
    args.num_epochs = ep
    args.loss_function = loss
    args.data = data
    args.num_classes = num_classes
    args.S = S
    args.E = E
    args.do_bottleneck_head = False
    print(args)

    #mlflow
    for key, value in args.items(): #vars(args).items()
        mlflow.log_param(key, value)

    
    # Setup model
    print('creating model...')
    model = create_model(args).cuda()
    print(model)
    if args.model_path:  # make sure to load pretrained ImageNet model
        #if "tresnet_m.pth" in args.model_path:
        #    state = torch.load(args.model_path, map_location='cpu')
        #    model.load_state_dict(state, strict=False)
        #else:
        state = torch.load(args.model_path, map_location='cpu')
        filtered_dict = {k: v for k, v in state['model'].items() if
                         (k in model.state_dict() and 'head.fc' not in k)}
        model.load_state_dict(filtered_dict, strict=False)
    print('done\n')

    os.makedirs("models", exist_ok=True)

    if "coco" in data:
        # COCO Data loading
        instances_path_val = os.path.join(args.data, 'annotations/instances_val2014.json')
        instances_path_train = os.path.join(args.data, 'annotations/instances_train2014.json')
        # data_path_val = args.data
        # data_path_train = args.data
        data_path_val = f'{args.data}/val2014'    # args.data
        data_path_train = f'{args.data}/train2014'  # args.data        
    elif "PASCAL" in data:
        # /dbfs/datasets/PASCAL-VOC
        
        instances_path_val = os.path.join(args.data, 'VOCasCOCO/annotations_val.json')
        instances_path_train = os.path.join(args.data, 'VOCasCOCO/annotations_train.json')
        data_path_val = f'{args.data}/VOCdevkit/VOC2007/JPEGImages'    # args.data
        data_path_train = f'{args.data}/VOCdevkit/VOC2007/JPEGImages'  # args.data        
    
    val_dataset = CocoDetection(args,
                                data_path_val,
                                instances_path_val,
                                transforms.Compose([
                                    transforms.Resize((args.image_size, args.image_size)),
                                    transforms.ToTensor(),
                                    # normalize, # no need, toTensor does normalization
                                ]))
    train_dataset = CocoDetection(args,
                                  data_path_train,
                                  instances_path_train,
                                  transforms.Compose([
                                      transforms.Resize((args.image_size, args.image_size)),
                                      CutoutPIL(cutout_factor=0.5),
                                      RandAugment(),
                                      transforms.ToTensor(),
                                      # normalize,
                                  ]))
    print("len(val_dataset)): ", len(val_dataset))
    print("len(train_dataset)): ", len(train_dataset))

    # Pytorch Data loader
    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=True,
        num_workers=args.workers, pin_memory=True)

    val_loader = torch.utils.data.DataLoader(
        val_dataset, batch_size=args.batch_size, shuffle=False,
        num_workers=args.workers, pin_memory=False)

    # Actuall Training
    train_multi_label_coco(model, train_loader, val_loader, args)


def train_multi_label_coco(model, train_loader, val_loader, args):


    #mlflow
    # sess = tf.compat.v1.InteractiveSession()
    # output_dir = tempfile.mkdtemp()
    # print("Writing TensorFlow events locally to %s\n" % output_dir)
    # writer = tf.summary.create_file_writer(output_dir, graph=sess.graph)
    mlflow.pytorch.autolog()
    
    ema = ModelEma(model, 0.9997)  # 0.9997^641=0.82

    # set optimizer
    Epochs = args.num_epochs
    Stop_epoch = args.stop_epoch
    weight_decay = args.weight_decay
    lr = args.lr
    S = args.S
    E = args.E
    if args.loss_function == "ASL":
        criterion = AsymmetricLoss(gamma_neg=4, gamma_pos=0, clip=0.05, disable_torch_grad_focal_loss=True)
    elif args.loss_function == "sigmoidF1":
        criterion = sigmoidF1(S = -1, E = 0)
        
    parameters = add_weight_decay(model, weight_decay)
    optimizer = torch.optim.Adam(params=parameters, lr=lr, weight_decay=0)  # true wd, filter_bias_and_bn
    steps_per_epoch = len(train_loader)
    scheduler = lr_scheduler.OneCycleLR(optimizer, max_lr=lr, steps_per_epoch=steps_per_epoch, epochs=Epochs,
                                        pct_start=0.2)


    highest_mAP = 0
    trainInfoList = []
    scaler = GradScaler()
    for epoch in range(Epochs):
        if epoch > Stop_epoch:
            break
        for i, (inputData, target) in enumerate(train_loader):
            inputData = inputData.cuda()
            target = target.cuda()  # (batch,3,num_classes)
            target = target.max(dim=1)[0]
            with autocast():  # mixed precision
                output = model(inputData).float()  # sigmoid will be done in loss !
            loss = criterion(output, target)
            model.zero_grad()

            scaler.scale(loss).backward()
            # loss.backward()

            scaler.step(optimizer)
            scaler.update()
            # optimizer.step()

            scheduler.step()

            ema.update(model)
            # store information
            if i % 100 == 0:
                trainInfoList.append([epoch, i, loss.item()])
                print('Epoch [{}/{}], Step [{}/{}], LR {:.1e}, Loss: {:.1f}'
                      .format(epoch, Epochs, str(i).zfill(3), str(steps_per_epoch).zfill(3),
                              scheduler.get_last_lr()[0], \
                              loss.item()))
            # if i == 50:
            #     break

        try:
            p = os.path.join(
                'models/', 'model-{}-{}.ckpt'.format(epoch + 1, i + 1))
            torch.save(model.state_dict(), p)
            mlflow.log_artifact(p)
        except:
            pass

        model.eval()
        mAP_score = validate_multi(val_loader, model, ema)
        model.train()
        if mAP_score > highest_mAP:
            highest_mAP = mAP_score
            try:
                torch.save(model.state_dict(), os.path.join(
                    'models/', 'model-highest.ckpt'))
            except:
                pass
        print('current_mAP = {:.2f}, highest_mAP = {:.2f}\n'.format(mAP_score, highest_mAP))

    #mlflow
    mlflow.log_metric("mAP_val", highest_mAP)
    #print("Uploading TensorFlow events as a run artifact.")
    #mlflow.log_artifacts(output_dir, artifact_path="events")
    


def validate_multi(val_loader, model, ema_model):
    print("starting validation")
    Sig = torch.nn.Sigmoid()
    preds_regular = []
    preds_ema = []
    targets = []
    for i, (input, target) in enumerate(val_loader):
        target = target
        target = target.max(dim=1)[0]
        # compute output
        with torch.no_grad():
            with autocast():
                output_regular = Sig(model(input.cuda())).cpu()
                output_ema = Sig(ema_model.module(input.cuda())).cpu()

        # for mAP calculation
        preds_regular.append(output_regular.cpu().detach())
        preds_ema.append(output_ema.cpu().detach())
        targets.append(target.cpu().detach())

    mAP_score_regular = mAP(torch.cat(targets).numpy(), torch.cat(preds_regular).numpy())
    mAP_score_ema = mAP(torch.cat(targets).numpy(), torch.cat(preds_ema).numpy())
    print("mAP score regular {:.2f}, mAP score EMA {:.2f}".format(mAP_score_regular, mAP_score_ema))
    return max(mAP_score_regular, mAP_score_ema)


if __name__ == '__main__':
    main()
