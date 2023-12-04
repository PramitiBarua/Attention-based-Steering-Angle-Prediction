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

from models import TruckResnet18,TruckResnet34, TruckResnet50,  TruckResnet101, TruckResnet152, GoogLeNet, GoogLeNetPlus, ResNet, BasicBlock, ResNetLarge, Block
from data import TruckDataset
#from attentionVGG import ProjectorBlock, SpatialAttn
output_file = open("output.txt", "w")
def loadData():

    data = pd.read_csv(csv_src)
    X = data[data.columns[0]].values
    y = data[data.columns[3]].values

    return X, y
#checkpoint_path = "/users/PCS0269/nswetha/Steering Angle Prediction/checkpoints/ResNetLarge/best_ckpt.pth"
# Dataset and DataLoaders
img_src_lst, angles = loadData()
X_train, X_valid, y_train, y_valid = train_test_split(img_src_lst, angles, test_size=1 - train_test_split_ratio, random_state=0, shuffle=True)
train_dataset = TruckDataset(X=X_train, y=y_train)
valid_dataset = TruckDataset(X=X_valid, y=y_valid)
#train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
valid_loader = DataLoader(valid_dataset, batch_size=batch_size, shuffle=False)

#logger.info("(3) Dataset Initiated. Training Started. ")

logger = get_logger()
logger.info("(1) Initiating Training ... ")
logger.info("Training on device: {}".format(device))
def test(model):
    
    
    writer = SummaryWriter()
    #model.load_state_dict(torch.load(checkpoint_path), strict=False)
    model = model.to(device)
    # Validation.
    model.eval()
    validLossMeter = LossMeter()
    epsilon = 3
    valid_batch_bar = tqdm.tqdm(total=len(valid_loader), desc="ValidBatch", position=0, leave=True)
    #with torch.no_grad():
    for batch_num, (leftImg, centerImg, rightImg, leftAng, centerAng, rightAng) in enumerate(valid_loader):
        

        leftImg, centerImg, rightImg, leftAng, centerAng, rightAng = group_move_to_device([leftImg, centerImg, rightImg, leftAng, centerAng, rightAng])

        for (img, y_train) in [[leftImg, leftAng], [centerImg, centerAng], [rightImg, rightAng]]:
            img.requires_grad = True
            y_pred = model(img)
            y_train = y_train.unsqueeze(1) # Shape N x 1
            loss = getLoss(y_pred, y_train)
                
                  # Backward pass
            model.zero_grad()
            loss.backward()
        
                # Generate adversarial image
            img_grad = img.grad.data
            sign_img_grad = img_grad.sign()
            img_adv = img + epsilon * sign_img_grad
            img_adv = torch.clamp(img_adv, 0, 1)  # Assuming the images are normalized between 0 and 1
        
                # Predict on adversarial image
            y_pred_adv = model(img_adv)
        
                # Calculate loss
            loss_adv = getLoss(y_pred_adv, y_train)

                # Update loss
            validLossMeter.update(loss_adv.item())

            # Print status
        if (batch_num+1) % print_freq == 0:
            
            status = 'Validation: [{0}/{1}]\tLoss {loss.val:.4f} ({loss.avg:.4f})\t'.format(batch_num+1, len(valid_loader), loss=validLossMeter)
            logger.info(status)

            # Log loss to tensorboard 
        if (batch_num+1) % tensorboard_freq == 0:
            
            writer.add_scalar('Valid_Loss_{0}'.format(tensorboard_freq), validLossMeter.avg, (batch_num+1) / tensorboard_freq)
        valid_batch_bar.update(1)
    valid_loss = validLossMeter.avg
    logger.info(valid_loss)
    #writer.add_scalar('Valid_Loss_epoch', valid_loss, epoch)
    #logger.info("Validation Loss of epoch [{0}/{1}]: {2}\n".format(epoch+1, epochs, valid_loss)) 

#output_file.close()   

if __name__ == "__main__":
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
        model = ResNet(BasicBlock, [3,4,6,3])
    elif net == "ResNetLarge":
        model = ResNetLarge(Block, [3,4,5,3])
    elif net == "GoogLeNet":
        model = GoogLeNet()
    elif net == "GoogLeNetPlus":
        model = GoogLeNetPlus()
    elif net == "CBAM":
        model = CBAM()
    elif net == "attention":
        model = attention56()
    else:
        raise ValueError("Invalid net value: {}".format(net))
    model = ResNetLarge(Block, [3,4,5,3])
    checkpoint_path = "/users/PCS0269/nswetha/Steering Angle Prediction/checkpoints/ResNetLarge/best_ckpt.pth"
    def load_ckpt_continue_training(ck_path, model, logger):
        model = model.to(device)
    
        checkpoint = torch.load(ck_path, map_location=torch.device(device))
        for key in list(checkpoint['model_state_dict'].keys()):
            checkpoint['model_state_dict'][key.replace('module.', '')] = checkpoint['model_state_dict'].pop(key)
        model.load_state_dict(checkpoint['model_state_dict'])
        #optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        #model = nn.DataParallel(model)
        
        #logger.info("Continue training mode, from epoch {0}. Checkpoint loaded.".format(checkpoint['epoch']))
        logger.info(checkpoint['epoch'])
        logger.info(checkpoint['loss'])
        return model

    # Call the test function with the model
    model=load_ckpt_continue_training(checkpoint_path,model,logger)
    
    

    # Call the test function with the model
    test(model)
