import warnings
from matplotlib import image
import pandas as pd
from sklearn.model_selection import train_test_split
import tqdm


import torch
torch.cuda.empty_cache()
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.optim import Adam, lr_scheduler
from tensorboardX import SummaryWriter

from config import device, epochs, lrate, wdecay, batch_size, getLoss, print_freq, tensorboard_freq, net, \
                    img_dir, csv_src, train_test_split_ratio, early_stop_tolerance, fine_tune_ratio, \
                    is_continue, best_ckpt_src, ckpt_src
from utils import group_move_to_device, LossMeter, get_logger, load_ckpt_continue_training

from models import TruckResnet18,TruckResnet34, TruckResnet50,  TruckResnet101, TruckResnet152, GoogLeNet, GoogLeNetPlus, ResNet, BasicBlock, ResNetLarge, Block, attention56
from data import TruckDataset
#from attentionVGG import ProjectorBlock, SpatialAttn

"""
Input Dimension Validation: 

GoogLeNet: N x 3 x 224 x 224 -> N x 1
TruckResnet18: N x 3 x 224 x 224 -> N x 1
TruckResnet50: N x 3 x 224 x 224 -> N x 1
TruckResnet34: N x 3 x 224 x 224 -> N x 1
TruckResnet101: N x 3 x 224 x 224 -> N x 1
TruckResnet152: N x 3 x 224 x 224 -> N x 1

"""

def train(cont=False):
    def loadData():
        data = pd.read_csv(csv_src)
        X = data[data.columns[0]].values
        y = data[data.columns[3]].values

        return X, y

    # For tensorboard tracking
    logger = get_logger()
    logger.info("(1) Initiating Training ... ")
    logger.info("Training on device: {}".format(device))
    writer = SummaryWriter()

    # Init model
    if net == "TruckResnet18":
        model = TruckResnet18()
    elif net == "TruckResnet34":
        model = TruckResnet34()
    elif net == "TruckResnet50":
        model = TruckResnet50()
    elif net == "TruckResnet101":
        model = TruckResnet101()
    elif net == "TruckResnet152":
        model = TruckResnet152()
    elif net == "ResNet":
        model = ResNet(BasicBlock, [3,4,6,3]) #ResNet18-> 2,2,2,2; ResNet20-> 2,2,3,2; ResNet22-> 2,3,3,2; ResNet24-> 2,3,4,2; 
    elif net == "ResNetLarge":
        model = ResNetLarge(Block, [3,3,3,3]) # ResNet38-> 3,3,3,3; ResNet44-> 3,4,4,3; ResNet50-> 3,4,6,3; ResNet101-> 3,4,23,3; ResNet152-> 3,8,36,3				        #ResNet26->3,3,3,3; ResNet28-> 3,3,4,3; ResNet30-> 3,4,4,3; ResNet32-> 3,4,5,3; ResNet34-> 3,4,6,3
    elif net == "GoogLeNet":
        model = GoogLeNet()
    elif net == "GoogLeNetPlus":
        model = GoogLeNetPlus()
    elif net == "CBAM":
        model = CBAM()
    elif net == "attention":
        #model = attention92()
        model = attention56()

    # Schedule learning rate
    optim = Adam(model.parameters(), lr=lrate, weight_decay=wdecay)
    pytorch_total_params = sum(p.numel() for p in model.parameters())
    print(pytorch_total_params)
    scheduler = lr_scheduler.MultiStepLR(optim, milestones=[int(epochs * fine_tune_ratio)], gamma=0.1)

    cur_epoch = 0
    best_mse = float('inf')
    epochs_since_improvement = 0

    # For continued training only (Must have best_ckpt_1.pth file)
    #if cont:
    #    model, optim, cur_epoch, best_mse = load_ckpt_continue_training(best_ckpt_src, model, optim, logger)
    #    logger.info("Current best loss (mse): {0}".format(best_mse))
    #    with warnings.catch_warnings():
    #        warnings.simplefilter("ignore")
    #        for i in range(cur_epoch):
    #            scheduler.step()
    #else:
    model = nn.DataParallel(model)
    model = model.to(device)

    logger.info("(2) Model Initiated ... ")
    logger.info("Training model: {}".format(net))

    # Dataset and DataLoaders
    img_src_lst, angles = loadData()
    X_train, X_valid, y_train, y_valid = train_test_split(img_src_lst, angles, test_size=1 - train_test_split_ratio, random_state=0, shuffle=True)
    train_dataset = TruckDataset(X=X_train, y=y_train)
    valid_dataset = TruckDataset(X=X_valid, y=y_valid)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    valid_loader = DataLoader(valid_dataset, batch_size=batch_size, shuffle=False)

    logger.info("(3) Dataset Initiated. Training Started. ")

    # Loop over epochs
    epoch_bar = tqdm.tqdm(total=epochs, desc="Epoch", position=cur_epoch, leave=True)
    for e in range(epochs - cur_epoch):
        epoch = e + cur_epoch
        # Training.
        model.train()
        trainLossMeter = LossMeter()
        train_batch_bar = tqdm.tqdm(total=len(train_loader), desc="TrainBatch", position=0, leave=True)
        for batch_num, (leftImg, centerImg, rightImg, leftAng, centerAng, rightAng) in enumerate(train_loader):

            leftImg, centerImg, rightImg, leftAng, centerAng, rightAng = group_move_to_device([leftImg, centerImg, rightImg, leftAng, centerAng, rightAng])

            optim.zero_grad()
            for (img, y_train) in [[leftImg, leftAng], [centerImg, centerAng], [rightImg, rightAng]]:
                y_pred = model(img)
                y_train = y_train.unsqueeze(1) # Shape N x 1
                loss = getLoss(y_pred, y_train)

                # Backward Propagation, Update weight and metrics
                loss.backward()
                optim.step()

                # Update loss
                trainLossMeter.update(loss.item())

            # Print status
            if (batch_num+1) % print_freq == 0:
                status = 'Epoch: [{0}][{1}/{2}]\t' \
                    'Loss {loss.val:.4f} ({loss.avg:.4f})\t'.format(epoch+1, batch_num+1, len(train_loader), loss=trainLossMeter)
                logger.info(status)

            # Log loss to tensorboard 
            if (batch_num+1) % tensorboard_freq == 0:
                writer.add_scalar('Train_Loss_{0}'.format(tensorboard_freq), 
                                trainLossMeter.avg, 
                                epoch * (len(train_loader) / tensorboard_freq) + (batch_num+1) / tensorboard_freq)
            train_batch_bar.update(1)

        writer.add_scalar('Train_Loss_epoch', trainLossMeter.avg, epoch)

        # Validation.
        model.eval()
        validLossMeter = LossMeter()
        valid_batch_bar = tqdm.tqdm(total=len(valid_loader), desc="ValidBatch", position=0, leave=True)
        with torch.no_grad():
            for batch_num, (leftImg, centerImg, rightImg, leftAng, centerAng, rightAng) in enumerate(valid_loader):

                leftImg, centerImg, rightImg, leftAng, centerAng, rightAng = group_move_to_device([leftImg, centerImg, rightImg, leftAng, centerAng, rightAng])

                for (img, y_train) in [[leftImg, leftAng], [centerImg, centerAng], [rightImg, rightAng]]:
                    y_pred = model(img)
                    y_train = y_train.unsqueeze(1) # Shape N x 1
                    loss = getLoss(y_pred, y_train)

                    # Update loss
                    validLossMeter.update(loss.item())

                # Print status
                if (batch_num+1) % print_freq == 0:
                    status = 'Validation: [{0}][{1}/{2}]\t' \
                        'Loss {loss.val:.4f} ({loss.avg:.4f})\t'.format(epoch+1, batch_num+1, len(valid_loader), loss=validLossMeter)
                    logger.info(status)

                # Log loss to tensorboard 
                if (batch_num+1) % tensorboard_freq == 0:
                    writer.add_scalar('Valid_Loss_{0}'.format(tensorboard_freq), 
                                    validLossMeter.avg, 
                                    epoch * (len(valid_loader) / tensorboard_freq) + (batch_num+1) / tensorboard_freq)
                valid_batch_bar.update(1)
        valid_loss = validLossMeter.avg
        writer.add_scalar('Valid_Loss_epoch', valid_loss, epoch)
        logger.info("Validation Loss of epoch [{0}/{1}]: {2}\n".format(epoch+1, epochs, valid_loss))    
    
        # Update optim scheduler
        scheduler.step()

        # Save checkpoint (Only saves on best)
        is_best = valid_loss < best_mse
        best_loss = min(valid_loss, best_mse)
        if not is_best:
            epochs_since_improvement += 1
            logger.info("Epochs since last improvement: %d\n" % (epochs_since_improvement,))
            #if epochs_since_improvement == early_stop_tolerance:
                #break # Early stopping.
        else:
            epochs_since_improvement = 0
            state = {
                'epoch': epoch,
                'loss': best_loss,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optim.state_dict(),
            }
            torch.save(state, ckpt_src)
            logger.info("Checkpoint updated.")
            best_mse = best_loss
        epoch_bar.update(1)
    writer.close()

if __name__ == "__main__":
    train(cont=is_continue)
    
    #pytorch_total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
