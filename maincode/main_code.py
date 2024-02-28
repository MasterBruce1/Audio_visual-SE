from torchsummaryX import summary
import os
import argparse
import time
from torch.optim import Adam
import torch
from mode_code import *
from datapath_manage import speaker_list_t,speaker_list_v, speaker_list_s, number_list_t, number_list_v, number_list_s, datapath_manage, dataset
from data_custom import *
from torch.utils.data import DataLoader

parser = argparse.ArgumentParser()

########## training ##########
# parser.add_argument('--retrain', action='store_true', help='to train a new model or to retrain an existing model.')
# parser.add_argument('--dataset', type=str, default='AV_enh', help='options: AV_enh')
parser.add_argument('--model', type=str, default='LAVSE', help='options: LAVSE')
# parser.add_argument('--loss', type=str, default='MSE', help='option: MSE')
# parser.add_argument('--opt', type=str, default='Adam', help='option: Adam')
# parser.add_argument('--keeptrain', action='store_true', help='continue training of a trained model. remember to set --fromepoch.')
# parser.add_argument('--fromepoch', type=int, default=0, help='the last epoch already trained.')
# parser.add_argument('--epochs', type=int, default=500, help='the last epoch wanted to be trained.')
# parser.add_argument('--train_batch_size', type=int, default=1, help='the batch size wanted to be trained.')
# parser.add_argument('--frame_seq', type=int, default=5, help='the frames amount of model input.')
# parser.add_argument('--loss_coefficient', type=float, default=0.001, help='loss = noisy_loss + loss_coefficient * lip_loss')
# parser.add_argument('--learning_rate', type=float, default=1e-5, help='learning rate of the optimizer.')
######### testing ##########
# parser.add_argument('--retest', action='store_true', help='to test or retest an existing model.')
# parser.add_argument('--testnomodel', action='store_true', help='generate wav files which have the same length(frames) as test wav files for scoring.')
# parser.add_argument('--test_batch_size', type=int, default=1, help='the batch size wanted to be tested.')
########## scoring ##########
# parser.add_argument('--rescore', action='store_true', help='to rescore test wavs. scoring will automatically start after testing even if this argument is not triggered.')

args = parser.parse_args()

# retrain = args.retrain
# dataset = args.dataset
model_name = args.model
# loss_name = args.loss
# opt_name = args.opt
# keeptrain = args.keeptrain
# fromepoch = args.fromepoch
# epochs = args.epochs
# train_batch_size = args.train_batch_size
# frame_seq = args.frame_seq
# loss_coefficient = args.loss_coefficient
# lr = args.learning_rate
# retest = args.retest
# testnomodel = args.testnomodel
# test_batch_size = args.test_batch_size
# rescore = args.rescore

if __name__ == '__main__':

    # ********** starting **********
    #print('\n********** starting **********\n')

    #print('The %s model with dataset of %s.\n' % (model_name, dataset))
    #start_time = time.time()

    # ********** check cuda **********
    print('\n********** check cuda **********\n')
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f'Using device: {device}')

    # ********** define datapath **********
    print('\n********** define datapath **********\n')
    train_datapath, val_datapath, test_datapath = datapath_manage(dataset)
    #print(train_datapath)
    train_dataset = data_custom(name = 'train', data_path = train_datapath)
    val_dataset = data_custom(name = 'val', data_path = val_datapath)
    train_loader = DataLoader(train_dataset, batch_size=3, shuffle=False)
    val_loader = DataLoader(val_dataset, batch_size=1, shuffle=False)
    # Initialize the autoencoder, optimizer, and loss function
    model = LAVSE().to(device)
    criterion = torch.nn.MSELoss()
    optimizer = Adam(model.parameters(), lr = 0.001)
    #optimizer = optim.Adam(model.parameters(), lr=0.001)
    #criterion = nn.MSELoss()

    # ********** define train function **********
    def train(model, train_dataset, val_dataset, criterion,optimizer,num_epochs = 10000):
        # Initialize variables to track epoch loss
        for epoch in range(num_epochs):
            model.train()
            total_audio_loss = 0.0
            total_visual_loss = 0.0
            print(f'Starting epoch {epoch+1}/{num_epochs}')
            #print(len(train_dataset))
            for i, batch in enumerate(train_dataset):
                batch = {k: v.to(device) for k, v in batch.items()}  # Move batch to device
                clean_t, noisy_t, image_t = batch['clean'], batch['noisy'], batch['image']
                optimizer.zero_grad()  # Zero the gradients
                reconstructed_audio, reconstructed_image = model(noisy_t, image_t)
                audio_loss = criterion(reconstructed_audio, clean_t)
                visual_loss = criterion(reconstructed_image, image_t)
                # Combine losses if necessary, or handle them separately
                total_loss = audio_loss + visual_loss
                # Backward pass
                total_loss.backward()
            
                # Optimize
                optimizer.step()

                # Update running loss totals
                total_audio_loss += audio_loss.item()
                total_visual_loss += visual_loss.item()
            
                # if (i+1) % 10 == 0:
                #     print(f'Epoch [{epoch+1}/{num_epochs}], Step [{i+1}/{len(train_dataset)}], Loss: ...')
            # Print average losses per epoch
            avg_audio_loss = total_audio_loss / len(train_dataset)
            avg_visual_loss = total_visual_loss / len(train_dataset)
            print(f'Epoch {epoch+1}/{num_epochs}, Average Audio Loss: {avg_audio_loss:.4f}, Average Visual Loss: {avg_visual_loss:.4f}')

             # Validation loop
            model.eval()  # Set model to evaluation mode
            total_val_audio_loss = 0.0
            total_val_visual_loss = 0.0
            with torch.no_grad():  # Disable gradient computation
                for i, batch in enumerate(val_dataset):
                    batch = {k: v.to(device) for k, v in batch.items()}
                    clean_v, noisy_v, image_v = batch['clean'], batch['noisy'], batch['image']
                    #print(clean_v.shape)
                    #print(noisy_v.shape)
                    #print(image_v.shape)
                    reconstructed_audio, reconstructed_image = model(noisy_v, image_v)
                    
                    audio_val_loss = criterion(reconstructed_audio, clean_v)
                    visual_val_loss = criterion(reconstructed_image, image_v)
                    
                    total_val_audio_loss += audio_val_loss.item()
                    total_val_visual_loss += visual_val_loss.item()
                
            avg_val_audio_loss = total_val_audio_loss / len(val_dataset)
            avg_val_visual_loss = total_val_visual_loss / len(val_dataset)
            print(f'Epoch {epoch+1}, Validation Audio Loss: {avg_val_audio_loss:.4f}, Validation Visual Loss: {avg_val_visual_loss:.4f}')
            
        print('Training complete.')

    #********** start training **********
    train(model, train_loader, val_loader,criterion, optimizer)

    #********** start test **********
    torch.save(model.state_dict(), 'D:/20241.1-2024.8.31/Micro+photodiode+denoise/LAVSE/LAVSE_model.pth')
    
    

